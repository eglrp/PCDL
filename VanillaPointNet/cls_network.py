import tensorflow as tf
from preprocess import compute_group
import sys
sys.path.append('../ExtendedOperator')
import indices_pool_grad

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev=1e-1, wd=None, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
      use_xavier: bool, whether to use xavier initializer

    Returns:
      Variable Tensor
    """
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def batch_norm_template(inputs, is_training, moments_dims, beta, gamma, running_mean, running_var):
    batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
    mean, var = tf.cond(is_training,
                        lambda: (batch_mean, batch_var),
                        lambda: (running_mean, running_var))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)

    return normed,batch_mean,batch_var


def _cross_entropy_loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def leaky_relu(f,leak=0.1):
        return tf.nn.leaky_relu(f,leak)

class Network:

    def _declare_parameter(self, weight_shape, index, name='mlp',bn=False):
        with tf.variable_scope(name+str(index)):
            self.params['{}{}_weight'.format(name,index)]=_variable_with_weight_decay('weight',weight_shape)
            self.params['{}{}_bias'.format(name,index)]=_variable_on_cpu('bias',[weight_shape[-1]],tf.constant_initializer(0))
            if bn:
                self._declare_bn_parameter(weight_shape[-1],index,name)

    def _declare_bn_parameter(self, num_channels, index, name='mlp'):
        with tf.device('/cpu:0'):
            self.params['{}{}_beta'.format(name,index)]=tf.Variable(
                tf.constant(0.0, shape=[num_channels]), name='beta', trainable=True)
            self.params['{}{}_gamma'.format(name,index)]=tf.Variable(
                tf.constant(1.0, shape=[num_channels]), name='gamma', trainable=True)

            self.bn_cache['{}{}_mean'.format(name,index)]=tf.Variable(
                tf.constant(0.0, shape=[num_channels]), name='mean', trainable=False)
            self.bn_cache['{}{}_var'.format(name,index)]=tf.Variable(
                tf.constant(1.0, shape=[num_channels]), name='var', trainable=False)
            self.bn_cache['{}{}_batch_means'.format(name,index)]=[]
            self.bn_cache['{}{}_batch_vars'.format(name,index)]=[]

    def _declare_tnet_parameters(self,feats_dim=64):
        self._declare_parameter([1,self.input_dim,1,64],1,'xyz_trans_mlp',self.bn)
        self._declare_parameter([1,1,64,128],2,'xyz_trans_mlp',self.bn)
        self._declare_parameter([1,1,128,1024],3,'xyz_trans_mlp',self.bn)
        self._declare_parameter([1024, 512], 1, 'xyz_trans_fc', self.bn)
        self._declare_parameter([512, 256], 2, 'xyz_trans_fc', self.bn)

        with tf.variable_scope('transform_XYZ'):
            with tf.device('/cpu:0'):
                weights = tf.get_variable('weights', [256, self.input_dim * self.input_dim],
                                          initializer=tf.constant_initializer(0.0),
                                          dtype=tf.float32)
                biases = tf.get_variable('bias', [self.input_dim * self.input_dim],
                                         initializer=tf.constant_initializer(0.0),
                                         dtype=tf.float32)
            biases += tf.constant(np.eye(self.input_dim).flatten(), dtype=tf.float32)
            self.params['xyz_trans_final1_weight']=weights
            self.params['xyz_trans_final1_bias']=biases

        self._declare_parameter([1,1,feats_dim,64],1,'feat_trans_mlp',self.bn)
        self._declare_parameter([1,1,64,128],2,'feat_trans_mlp',self.bn)
        self._declare_parameter([1,1,128,1024],3,'feat_trans_mlp',self.bn)
        self._declare_parameter([1024, 512], 1, 'feat_trans_fc', self.bn)
        self._declare_parameter([512, 256], 2, 'feat_trans_fc', self.bn)

        with tf.variable_scope('transform_feat'):
            with tf.device('/cpu:0'):
                weights = tf.get_variable('weights', [256, feats_dim * feats_dim],
                                          initializer=tf.constant_initializer(0.0),
                                          dtype=tf.float32)
                biases = tf.get_variable('bias', [feats_dim * feats_dim],
                                         initializer=tf.constant_initializer(0.0),
                                         dtype=tf.float32)
            biases += tf.constant(np.eye(feats_dim).flatten(), dtype=tf.float32)
            self.params['feat_trans_final1_weight']=weights
            self.params['feat_trans_final1_bias']=biases

    def _declare_all_parameters(self):
        self._declare_parameter([1, self.input_dim, 1, 64], 1, 'mlp',self.bn)
        self._declare_parameter([1, 1, 64, 64], 2, 'mlp',self.bn)
        self._declare_parameter([1, 1, 64, 64], 3, 'mlp',self.bn)
        self._declare_parameter([1, 1, 64, 128], 4, 'mlp',self.bn)
        self._declare_parameter([1, 1, 128, self.final_dim], 5, 'mlp', self.bn)

        self._declare_parameter([self.final_dim, 512], 1, 'fc', self.bn)
        self._declare_parameter([512, 256], 2, 'fc',self.bn)
        self._declare_parameter([256, self.num_classes], 3, 'fc', False)

        if self.use_trans:
            self._declare_tnet_parameters()

    def _declare_mlp_layer(self,input,index,tower_name,name,activation_fn=tf.nn.relu,bn=False,is_training=None):
        weight=self.params['{}{}_weight'.format(name,index)]
        bias=self.params['{}{}_bias'.format(name,index)]
        with tf.name_scope(name+str(index)):
            mlp = tf.nn.conv2d(input, weight, (1, 1, 1, 1), 'VALID')
            mlp = tf.nn.bias_add(mlp,bias)
            mlp = activation_fn(mlp)
            #############################
            # self.ops['{}_mlp{}'.format(tower_name,index)]=mlp
            if bn:
                gamma=self.params['{}{}_gamma'.format(name,index)]
                beta=self.params['{}{}_beta'.format(name,index)]
                running_mean=self.bn_cache['{}{}_mean'.format(name,index)]
                running_var=self.bn_cache['{}{}_var'.format(name,index)]
                mlp,batch_mean,batch_var=batch_norm_template(
                    mlp,is_training,[0,1,2],beta,gamma,running_mean,running_var)
                self.bn_cache['{}{}_batch_means'.format(name,index)].append(batch_mean)
                self.bn_cache['{}{}_batch_vars'.format(name,index)].append(batch_var)

        self.ops['{}_{}{}'.format(tower_name,name,index)]=mlp
        return mlp

    def _declare_fc_layer(self,input,index,tower_name,name,activation_fn=tf.nn.relu,bn=False,is_training=None):
        weight=self.params['{}{}_weight'.format(name,index)]
        bias=self.params['{}{}_bias'.format(name,index)]
        with tf.name_scope(name+str(index)):
            fc=tf.nn.bias_add(tf.matmul(input,weight),bias)
            if activation_fn is not None:
                fc=activation_fn(fc)
            if bn:
                gamma=self.params['{}{}_gamma'.format(name,index)]
                beta=self.params['{}{}_beta'.format(name,index)]
                running_mean=self.bn_cache['{}{}_mean'.format(name,index)]
                running_var=self.bn_cache['{}{}_var'.format(name,index)]
                fc,batch_mean,batch_var=batch_norm_template(
                    fc,is_training,[0],beta,gamma,running_mean,running_var)
                self.bn_cache['{}{}_batch_means'.format(name,index)].append(batch_mean)
                self.bn_cache['{}{}_batch_vars'.format(name,index)].append(batch_var)

        self.ops['{}_{}{}'.format(tower_name,name,index)]=fc
        return fc

    def _declare_pooling(self,input,tower_name,name='pool'):
        with tf.name_scope(name):
            feature_pool=tf.reduce_max(input,axis=1)
            feature_pool=tf.squeeze(feature_pool,axis=(1))

        self.ops['{}_{}'.format(tower_name,name)]=feature_pool
        return feature_pool

    def _declare_tnet(self,input,is_training,input_dim,tower_name,name='xyz_trans',active_fn=tf.nn.relu):
        trans_mlp1=self._declare_mlp_layer(input,1,tower_name,'{}_mlp'.format(name),active_fn,self.bn,is_training)
        trans_mlp2=self._declare_mlp_layer(trans_mlp1,2,tower_name,'{}_mlp'.format(name),active_fn,self.bn,is_training)
        trans_mlp3=self._declare_mlp_layer(trans_mlp2,3,tower_name,'{}_mlp'.format(name),active_fn,self.bn,is_training)
        trans_pool=self._declare_pooling(trans_mlp3,tower_name,'{}_pool'.format(name))
        trans_fc1=self._declare_fc_layer(trans_pool,1,tower_name,'{}_fc'.format(name),active_fn,self.bn,is_training)
        trans_fc2=self._declare_fc_layer(trans_fc1,2,tower_name,'{}_fc'.format(name),active_fn,self.bn,is_training)
        trans_matrix=self._declare_fc_layer(trans_fc2,1,tower_name,'{}_final'.format(name),None,False)
        trans_matrix=tf.reshape(trans_matrix,[tf.shape(input)[0],input_dim,input_dim])
        return trans_matrix

    def _renew_running_mean_var(self, attached_op, bn_decay):
        op_list=[]
        with tf.name_scope('update_mean_var'):
            for name in self.bn_layer_names:
                with tf.name_scope('{}_update'.format(name)):
                    batch_vars=[tf.expand_dims(var,axis=0) for var in self.bn_cache['{}_batch_vars'.format(name)]]
                    batch_means=[tf.expand_dims(mean,axis=0) for mean in self.bn_cache['{}_batch_means'.format(name)]]
                    running_var=self.bn_cache['{}_var'.format(name)]
                    running_mean=self.bn_cache['{}_mean'.format(name)]
                    with tf.control_dependencies([attached_op]):
                        var_ass=tf.assign_sub(running_var,(1-bn_decay)*
                                              (running_var-tf.reduce_mean(tf.concat(batch_vars,axis=0),axis=0)))
                        mean_ass=tf.assign_sub(running_mean,(1-bn_decay)*
                                              (running_mean-tf.reduce_mean(tf.concat(batch_means,axis=0),axis=0)))
                    op_list+=[var_ass,mean_ass]
                    tf.summary.scalar(name+'_mean',tf.reduce_mean(running_mean))
                    tf.summary.scalar(name+'_mean_tower1',tf.reduce_mean(self.bn_cache['{}_batch_means'.format(name)][0]))
                    tf.summary.scalar(name+'_var',tf.reduce_mean(running_var))
                    tf.summary.scalar(name+'_var_tower1',tf.reduce_mean(self.bn_cache['{}_batch_vars'.format(name)][0]))
        return tf.group(*op_list)

    def __init__(self,input_dim,num_classes,use_bn,final_dim=1024,use_trans=False):
        self.params={}
        self.ops={}
        self.bn=use_bn
        self.input_dim=input_dim
        self.final_dim=final_dim
        self.num_classes=num_classes
        self.use_trans=use_trans
        if self.bn:
            self.bn_cache={}
            self.bn_layer_names=['mlp{}'.format(i) for i in range(1,6)]
            self.bn_layer_names+=['fc{}'.format(i) for i in range(1,3)]
            if self.use_trans:
                self.bn_layer_names+=['xyz_trans_mlp{}'.format(i) for i in range(1,4)]
                self.bn_layer_names+=['xyz_trans_fc{}'.format(i) for i in range(1,3)]
                self.bn_layer_names+=['feat_trans_mlp{}'.format(i) for i in range(1,4)]
                self.bn_layer_names+=['feat_trans_fc{}'.format(i) for i in range(1,3)]

        # declare parameter
        self._declare_all_parameters()

    def inference(self, input, tower_name, is_training, active_fn=leaky_relu):
        if self.use_trans:
            trans_matrix=self._declare_tnet(tf.expand_dims(input,axis=3),is_training,self.input_dim,tower_name,'xyz_trans',active_fn)
            input=tf.matmul(input,trans_matrix)

        input=tf.expand_dims(input,axis=3)
        mlp1=self._declare_mlp_layer(input, 1, tower_name,'mlp', active_fn, bn=self.bn, is_training=is_training)
        mlp2=self._declare_mlp_layer(mlp1, 2, tower_name,'mlp', active_fn, bn=self.bn, is_training=is_training)

        if self.use_trans:
            trans_matrix=self._declare_tnet(mlp2,is_training,64,tower_name,'feat_trans',active_fn)
            mlp2=tf.matmul(tf.squeeze(mlp2,axis=2),trans_matrix)
            mlp2=tf.expand_dims(mlp2,axis=2)

        mlp3=self._declare_mlp_layer(mlp2, 3, tower_name,'mlp', active_fn, bn=self.bn, is_training=is_training)
        mlp4=self._declare_mlp_layer(mlp3, 4, tower_name,'mlp', active_fn, bn=self.bn, is_training=is_training)
        mlp5=self._declare_mlp_layer(mlp4, 5, tower_name,'mlp', active_fn, bn=self.bn, is_training=is_training)

        self.ops['points_feats']=tf.squeeze(mlp5,axis=2)
        feature_pool=self._declare_pooling(mlp5,tower_name)
        self.ops['global_feats']=feature_pool

        fc1=self._declare_fc_layer(feature_pool,1,tower_name,'fc',bn=self.bn,is_training=is_training)
        fc2=self._declare_fc_layer(fc1,2,tower_name,'fc',bn=self.bn,is_training=is_training)
        fc2=tf.cond(is_training,
                    lambda: tf.nn.dropout(fc2,0.7),
                    lambda: fc2)
        fc3=self._declare_fc_layer(fc2,3,tower_name,'fc',None,bn=False)

        return fc3

    def declare_train_net(self,inputs,labels,is_training,gpu_num,
                          init_lr,lr_decay_rate,lr_decay_epoch,
                          init_bn,bn_decay_rate,bn_decay_epoch,bn_clip,
                          batch_size,total_size,):
        with tf.device('/cpu:0'):
            tower_losses=[]
            tower_logits=[]
            tower_gradients=[]

            global_step=tf.get_variable('gloabel_step',[],tf.int64,tf.constant_initializer(0),trainable=False)

            # lr
            epoch_batch_num=total_size/batch_size
            lr=tf.train.exponential_decay(init_lr,global_step,lr_decay_epoch*epoch_batch_num,
                                          lr_decay_rate,staircase=True)
            lr=tf.maximum(lr,1e-5)
            tf.summary.scalar('learning_rate',lr)

            # bn_decay
            bn_momentum=tf.train.exponential_decay(init_bn,global_step,bn_decay_epoch*
                                                   epoch_batch_num,bn_decay_rate,staircase=True)
            bn_decay = tf.minimum(bn_clip, 1 - bn_momentum)
            tf.summary.scalar('bn_decay',bn_decay)

            opt=tf.train.AdamOptimizer(lr)
            # opt=tf.train.GradientDescentOptimizer(lr)

            for i in xrange(gpu_num):
                with tf.device('/gpu:{}'.format(i)):
                    tower_name='tower_{}'.format(i)
                    with tf.name_scope(tower_name):
                        logits=self.inference(inputs[i], tower_name, is_training, active_fn=leaky_relu)
                        losses=_cross_entropy_loss(logits,labels[i])
                        gradients=opt.compute_gradients(losses)

                        tower_logits.append(logits)
                        tower_losses.append(losses)
                        tower_gradients.append(gradients)

            # for tg in tower_gradients:
            #     for g in tg:
            #         print g

            self.ops['loss']=tf.add_n(tower_losses)
            tf.summary.scalar('loss_val',self.ops['loss'])

            all_logits=tf.concat(tower_logits,axis=0)
            all_labels=tf.concat(labels,axis=0)
            correct_num=tf.equal(tf.argmax(all_logits,axis=1),tf.cast(all_labels,tf.int64))
            accuracy=tf.reduce_mean(tf.cast(correct_num,tf.float32))
            self.ops['logits']=all_logits
            self.ops['accuracy']=accuracy
            tf.summary.scalar('accuracy',accuracy)

            grads=_average_gradients(tower_gradients)
            apply_grad_op=opt.apply_gradients(grads,global_step=global_step)
            self.ops['apply_grad']=apply_grad_op
            self.ops['global_step']=global_step

            if self.bn:
                self.ops['apply_grad']=self._renew_running_mean_var(apply_grad_op,bn_decay)


    def inference_normal(self, input, tower_name, is_training, active_fn=leaky_relu):
        if self.use_trans:
            trans_matrix=self._declare_tnet(tf.expand_dims(input,axis=3),is_training,self.input_dim,tower_name,'xyz_trans',active_fn)
            input=tf.matmul(input,trans_matrix)

        input=tf.expand_dims(input,axis=3)
        mlp1=self._declare_mlp_layer(input, 1, tower_name,'mlp', active_fn, bn=self.bn, is_training=is_training)
        mlp2=self._declare_mlp_layer(mlp1, 2, tower_name,'mlp', active_fn, bn=self.bn, is_training=is_training)

        if self.use_trans:
            trans_matrix=self._declare_tnet(mlp2,is_training,64,tower_name,'feat_trans',active_fn)
            mlp2=tf.matmul(tf.squeeze(mlp2,axis=2),trans_matrix)
            mlp2=tf.expand_dims(mlp2,axis=2)

        mlp3=self._declare_mlp_layer(mlp2, 3, tower_name,'mlp', active_fn, bn=self.bn, is_training=is_training)
        mlp4=self._declare_mlp_layer(mlp3, 4, tower_name,'mlp', active_fn, bn=self.bn, is_training=is_training)
        mlp5=self._declare_mlp_layer(mlp4, 5, tower_name,'mlp', active_fn, bn=self.bn, is_training=is_training)

        self.ops['points_feats']=tf.squeeze(mlp5,axis=2)
        # whether feed in normal out points
        mlp5=tf.cond(is_training,
                     lambda : tf.split(mlp5,2, axis=1)[0],
                     lambda : mlp5)
        feature_pool=self._declare_pooling(mlp5,tower_name)
        self.ops['global_feats']=feature_pool

        fc1=self._declare_fc_layer(feature_pool,1,tower_name,'fc',bn=self.bn,is_training=is_training)
        fc2=self._declare_fc_layer(fc1,2,tower_name,'fc',bn=self.bn,is_training=is_training)
        # fc2=tf.cond(is_training,
        #             lambda: tf.nn.dropout(fc2,0.7),
        #             lambda: fc2)
        fc3=self._declare_fc_layer(fc2,3,tower_name,'fc',None,bn=False)

        return fc3

    def normal_constraint_loss(self, indices, original_feats,constraint_ratio):
        # weights n k t 2
        neighbor_size=tf.shape(indices)[2]
        original_feats, out_feats=tf.split(original_feats, 2, axis=1)                           # n,k,f
        neighbor_feats=tf.gather_nd(original_feats, indices)                                    # n,k,t,f
        tile_feats=tf.tile(tf.expand_dims(original_feats, axis=2), [1, 1, neighbor_size, 1])    # n k t f
        normal_diff=tf.abs(original_feats - out_feats)                                          # n,k,f
        tangent_diff=tf.reduce_max(tf.abs(tile_feats-neighbor_feats),axis=2)                    # n,k,f
        constraint_loss=tf.maximum(tangent_diff-normal_diff,0)                                  # n,k,f
        constraint_loss=tf.reduce_min(tangent_diff-normal_diff,axis=2)
        # loss=tf.reduce_min(constraint_loss,axis=2)                                            # n,k
        loss=tf.reduce_mean(tangent_diff-normal_diff)
        tf.summary.scalar('normal_loss',loss)
        tf.add_to_collection('losses',loss*constraint_ratio)


    def declare_train_net_normal(self,inputs,labels,nidxs,constraint_ratio,
                                 is_training,gpu_num,
                                 init_lr,lr_decay_rate,lr_decay_epoch,
                                 init_bn,bn_decay_rate,bn_decay_epoch,bn_clip,
                                 batch_size,total_size,):
                with tf.device('/cpu:0'):
                    tower_losses=[]
                    tower_logits=[]
                    tower_gradients=[]

                    global_step=tf.get_variable('gloabel_step',[],tf.int64,tf.constant_initializer(0),trainable=False)

                    # lr
                    epoch_batch_num=total_size/batch_size
                    lr=tf.train.exponential_decay(init_lr,global_step,lr_decay_epoch*epoch_batch_num,
                                                  lr_decay_rate,staircase=True)
                    lr=tf.maximum(lr,1e-5)
                    tf.summary.scalar('learning_rate',lr)

                    # bn_decay
                    bn_momentum=tf.train.exponential_decay(init_bn,global_step,bn_decay_epoch*
                                                           epoch_batch_num,bn_decay_rate,staircase=True)
                    bn_decay = tf.minimum(bn_clip, 1 - bn_momentum)
                    tf.summary.scalar('bn_decay',bn_decay)

                    # opt=tf.train.AdamOptimizer(lr)
                    opt=tf.train.GradientDescentOptimizer(init_lr)

                    for i in xrange(gpu_num):
                        with tf.device('/gpu:{}'.format(i)):
                            tower_name='tower_{}'.format(i)
                            with tf.name_scope(tower_name):
                                logits=self.inference_normal(inputs[i], tower_name, is_training, active_fn=leaky_relu)
                                _cross_entropy_loss(logits,labels[i])
                                self.normal_constraint_loss(nidxs[i],self.ops['points_feats'],constraint_ratio)
                                losses=tf.add_n(tf.get_collection('losses'))
                                gradients=opt.compute_gradients(losses)

                                tower_logits.append(logits)
                                tower_losses.append(losses)
                                tower_gradients.append(gradients)

                    self.ops['loss']=tf.add_n(tower_losses)
                    tf.summary.scalar('loss_val',self.ops['loss'])

                    all_logits=tf.concat(tower_logits,axis=0)
                    all_labels=tf.concat(labels,axis=0)
                    correct_num=tf.equal(tf.argmax(all_logits,axis=1),tf.cast(all_labels,tf.int64))
                    accuracy=tf.reduce_mean(tf.cast(correct_num,tf.float32))
                    self.ops['logits']=all_logits
                    self.ops['accuracy']=accuracy
                    self.ops['correct_num']=tf.reduce_sum(tf.cast(correct_num,tf.float32))
                    self.ops['batch_num']=tf.shape(all_logits)[0]
                    tf.summary.scalar('accuracy',accuracy)

                    grads=_average_gradients(tower_gradients)
                    apply_grad_op=opt.apply_gradients(grads,global_step=global_step)
                    self.ops['apply_grad']=apply_grad_op
                    self.ops['global_step']=global_step

                    if self.bn:
                        self.ops['apply_grad']=self._renew_running_mean_var(apply_grad_op,bn_decay)

import numpy as np
if __name__=="__main__":
    batch_size=30
    point_num=2048
    gpu_num=2

    lr_init=1e-1
    lr_decay_epoch=5
    lr_decay_rate=0.9

    bn_init=0.95
    bn_decay_epoch=5
    bn_decay_rate=0.9
    bn_clip=0.99

    points=[]
    labels=[]
    for i in xrange(gpu_num):
        points.append(tf.placeholder(tf.float32,[batch_size,point_num,3]))
        labels.append(tf.placeholder(tf.int64,[batch_size]))
    is_training=tf.placeholder(tf.bool)

    net=Network(3,40,True,1024,True)
    net.declare_train_net(points,labels,is_training,gpu_num,
                          lr_init,lr_decay_rate,lr_decay_epoch,
                          bn_init,bn_decay_rate,bn_decay_epoch,bn_clip,
                          batch_size,300)

    fixed_points=np.random.uniform(-1, 1, [batch_size, point_num, 3])
    fixed_labels=np.random.randint(0,40,[batch_size])


    saver=tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    for i in range(100000):
        feed_dict={}
        for k in range(gpu_num):
            feed_dict[points[k]]=fixed_points
            feed_dict[labels[k]]=fixed_labels
        feed_dict[is_training]=True

        acc,_=sess.run([net.ops['accuracy'],net.ops['apply_grad']],feed_dict)

        if i%50==0:
            print 'step {} acc {:5}'.format(i,acc)


