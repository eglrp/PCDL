import argparse
import time
import numpy as np
import os

import tensorflow as tf

from autoencoder_network import concat_pointnet_encoder_v2,fc_voxel_decoder_v2,voxel_filling_loss
from modelnet.data_util import read_modelnet_v2,rotate,exchange_dims_zy,jitter_point_cloud
from s3dis.voxel_util import points2voxel_gpu_modelnet,voxel2points,points2covars_gpu
from s3dis.draw_util import output_points

from train_util import log_str,average_gradients
from provider import ProviderV2

import random

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=2, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')

parser.add_argument('--lr_init', type=float, default=1e-3, help='')
parser.add_argument('--lr_clip', type=float, default=1e-5, help='')
parser.add_argument('--decay_rate', type=float, default=0.5, help='')
parser.add_argument('--decay_epoch', type=int, default=50, help='')
parser.add_argument('--num_classes', type=int, default=40, help='')


parser.add_argument('--log_step', type=int, default=20, help='')
parser.add_argument('--train_dir', type=str, default='train/modelnet_voxel', help='')
parser.add_argument('--save_dir', type=str, default='model/modelnet_voxel', help='')
parser.add_argument('--log_file', type=str, default='modelnet_voxel.log', help='')

parser.add_argument('--train_epoch_num', type=int, default=500, help='')

parser.add_argument('--dump_dir', type=str, default='unsupervise/modelnet_voxel', help='')
parser.add_argument('--dump_num', type=int, default=5, help='')
parser.add_argument('--split_num', type=int, default=30, help='')

FLAGS = parser.parse_args()


def tower_loss(points, covars, labels, true_state, is_training, num_classes, voxel_num, reuse=False):
    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
        points_covars=tf.concat([points,covars],axis=2)                                                           # [n,k,16]
        global_feats,_=concat_pointnet_encoder_v2(points_covars,is_training, reuse, final_dim=1024,use_bn=False)  # [n,1024]
        # get reconstructed voxel
        voxel_state,_=fc_voxel_decoder_v2(global_feats,is_training,voxel_num, False, reuse,use_bn=False)
        # logits=model_classifier(global_feats,num_classes,reuse, is_training)

    # softmax loss
    # labels=tf.reshape(labels,[-1,1])
    # labels=tf.squeeze(labels,axis=1)
    # loss=tf.losses.sparse_softmax_cross_entropy(labels,logits)
    # tf.add_to_collection(tf.GraphKeys.LOSSES,loss)
    # tf.summary.scalar(loss.op.name,loss)

    # filling loss
    recon_loss=voxel_filling_loss(voxel_state, true_state)
    tf.add_to_collection(tf.GraphKeys.LOSSES,recon_loss)
    tf.summary.scalar(recon_loss.op.name,recon_loss)

    return recon_loss,voxel_state


def train_ops(points, covars, labels, true_state, is_training, num_classes, voxel_num, epoch_batch_num):
    ops={}
    with tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        decay_steps=epoch_batch_num*FLAGS.decay_epoch
        lr=tf.train.exponential_decay(FLAGS.lr_init,global_step,decay_steps,FLAGS.decay_rate,staircase=True)
        lr=tf.maximum(FLAGS.lr_clip,lr)
        tf.summary.scalar('learning rate',lr)

        opt=tf.train.AdamOptimizer(lr)

        with tf.name_scope('split_data'):
            tower_points=tf.split(points, FLAGS.num_gpus)
            tower_covars=tf.split(covars,FLAGS.num_gpus)
            tower_labels=tf.split(labels,FLAGS.num_gpus)
            tower_true_state=tf.split(true_state,FLAGS.num_gpus)

        reuse=False
        tower_grads=[]
        tower_recon_losses=[]
        tower_voxel_state=[]
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('tower_{}'.format(i)):
                    recon_loss,voxel_state\
                        =tower_loss(tower_points[i], tower_covars[i],tower_labels[i],
                                    tower_true_state[i], is_training, num_classes,voxel_num,reuse)

                    grad=opt.compute_gradients(recon_loss)
                    tower_grads.append(grad)

                    tower_recon_losses.append(recon_loss)
                    tower_voxel_state.append(voxel_state)

                    reuse=True

        avg_grad=average_gradients(tower_grads)
        update_op=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_op):
            apply_grad_op=opt.apply_gradients(avg_grad,global_step=global_step)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge(summaries)

        total_recon_loss_op=tf.add_n(tower_recon_losses)/FLAGS.num_gpus

        voxel_state_op=tf.concat(tower_voxel_state,axis=0)

        ops['total_recon_loss']=total_recon_loss_op
        ops['apply_grad']=apply_grad_op
        ops['summary']=summary_op
        ops['global_step']=global_step
        ops['voxel_state']=voxel_state_op

    return ops


def read_fn(model,filename):
    points, nidxs, labels=read_modelnet_v2(filename)
    points=exchange_dims_zy(points)
    if model=='train':
        points=rotate(points)
        # points=jitter_point_cloud(points)
    else:
        # new_points=np.empty([points.shape[0]*4,points.shape[1],3],dtype=np.float32)
        # new_points[:points.shape[0]]=points
        # new_points[points.shape[0]:points.shape[0]*2]=rotate(points,np.pi/2.0)
        # new_points[points.shape[0]*2:points.shape[0]*3]=rotate(points,np.pi)
        # new_points[points.shape[0]*3:]=rotate(points,np.pi*3.0/2.0)
        #
        # labels=np.tile(labels,[4,1])
        # nidxs=np.tile(nidxs,[4,1,1])
        # points=new_points
        pass

    voxel_state = points2voxel_gpu_modelnet(points, FLAGS.split_num, 1)
    covars= points2covars_gpu(points, nidxs, 16, 0)
    labels=labels.flatten()

    return points, covars, labels, voxel_state


def batch_fn(file_data, cur_idx, data_indices, require_size):
    end_idx = min(cur_idx + require_size, file_data[0].shape[0])
    batch_data=[item[data_indices[cur_idx:end_idx]] for item in file_data]
    return batch_data , end_idx - cur_idx


def unpack_feats_labels(batch,num_gpus):
    if batch[0].shape[0]%num_gpus!=0:
        left_num=(batch[0].shape[0]/num_gpus+1)*num_gpus-batch[0].shape[0]
        left_idx = np.random.randint(0, batch[0].shape[0], left_num)
        for i in enumerate(batch):
            batch[i]=np.concatenate([batch[i],batch[i][left_idx]],axis=0)

    return batch


def output_gen_points(pts, voxels, gen_state, file_idx, epoch_num, dump_dir):
    if pts is not  None:
        pts[:, :] += 1.0
        pts[:, :] /= 2.0
        fn = os.path.join(dump_dir, '{}_{}_points.txt'.format(epoch_num, file_idx))
        output_points(fn, pts)

    if voxels is not None:
        true_state_pts = voxel2points(voxels)
        fn = os.path.join(dump_dir, '{}_{}_state_true.txt'.format(epoch_num, file_idx))
        output_points(fn, true_state_pts)

    if gen_state is not None:
        gen_state[gen_state < 0.0] = 0.0
        gen_state[gen_state > 1.0] = 1.0
        pred_state_pts = voxel2points(gen_state)
        fn = os.path.join(dump_dir, '{}_{}_state_pred.txt'.format(epoch_num, file_idx))
        output_points(fn, pred_state_pts)


def train_one_epoch(ops,pls,sess,summary_writer,trainset,epoch_num,feed_dict):
    total_model=0
    begin_time=time.time()
    total_recon_losses=[]
    for i,feed_in in enumerate(trainset):
        points_list, covars_list, labels_list, voxel_state_list=\
            unpack_feats_labels(feed_in,FLAGS.num_gpus)

        feed_dict[pls['points']]=points_list
        feed_dict[pls['covars']]=covars_list
        feed_dict[pls['is_training']]=True
        feed_dict[pls['voxel_state']]=voxel_state_list
        total_model+=points_list.shape[0]

        _,recon_loss_val=sess.run(
            [ops['apply_grad'],ops['total_recon_loss']],feed_dict)
        total_recon_losses.append(recon_loss_val)

        if i % FLAGS.log_step==0:
            summary,global_step,gen_state=sess.run(
                [ops['summary'],ops['global_step'],ops['voxel_state']],feed_dict)

            log_str('epoch {} step {} recon_loss {:.5} | {:.5} examples/s'.format(
                epoch_num,i,np.mean(np.asarray(total_recon_losses)),
                float(total_model)/(time.time()-begin_time)
            ),FLAGS.log_file)

            summary_writer.add_summary(summary,global_step)
            total_model=0
            begin_time=time.time()
            total_recon_losses=[]


def test_one_epoch(ops,pls,sess,saver,testset,epoch_num,feed_dict):
    total_model=0
    begin_time=time.time()
    total_recon_losses=[]
    left_size=FLAGS.dump_num
    for i,feed_in in enumerate(testset):
        points_list, covars_list, labels_list, voxel_state_list=\
            unpack_feats_labels(feed_in,FLAGS.num_gpus)

        feed_dict[pls['points']]=points_list
        feed_dict[pls['covars']]=covars_list
        feed_dict[pls['is_training']]=True
        feed_dict[pls['voxel_state']]=voxel_state_list
        total_model+=points_list.shape[0]

        recon_loss_val,gen_state=sess.run([ops['total_recon_loss'],ops['voxel_state']],feed_dict)
        total_recon_losses.append(recon_loss_val)
        if left_size>0:
            for k in range(min(left_size,points_list.shape[0])):
                output_gen_points(points_list[k],voxel_state_list[k],gen_state[k],left_size,epoch_num,FLAGS.dump_dir)
                left_size-=1

    log_str('test epoch {} recon_loss {:.5} | {:.5} examples/s'.format(
        epoch_num,np.mean(np.asarray(total_recon_losses)),
        float(total_model)/(time.time()-begin_time)
    ),FLAGS.log_file)

    checkpoint_path = os.path.join(FLAGS.save_dir, 'unsupervise{}.ckpt'.format(epoch_num))
    saver.save(sess,checkpoint_path)


def train():
    pt_num=2048
    voxel_num=FLAGS.split_num*FLAGS.split_num*FLAGS.split_num

    # train_list,test_list=get_train_test_split()
    train_list=['data/ModelNetTrain/nidxs/ply_data_train{}.h5'.format(i) for i in xrange(5)]
    train_list+=['data/ModelNetTrain/voxel/ply_data_test_voxel0.h5']
    test_list=['data/ModelNetTrain/voxel/ply_data_test_voxel1.h5']

    train_provider = ProviderV2(train_list,'train',FLAGS.batch_size*FLAGS.num_gpus,batch_fn,read_fn,2)
    test_provider = ProviderV2(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,batch_fn,read_fn,2)

    try:
        pls={}
        pls['points']=tf.placeholder(tf.float32,[None,pt_num,3],'points')
        pls['covars']=tf.placeholder(tf.float32,[None,pt_num,9],'covars')
        pls['labels']=tf.placeholder(tf.int64,[None],'labels')
        pls['is_training']=tf.placeholder(tf.bool,name='is_training')
        pls['voxel_state']=tf.placeholder(tf.float32,[None,voxel_num],'voxel_state')

        ops=train_ops(pls['points'],pls['covars'],
                      pls['labels'],pls['voxel_state'],pls['is_training'],
                      FLAGS.num_classes,voxel_num,
                      10000/(FLAGS.batch_size*FLAGS.num_gpus))

        feed_dict = {}
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=500)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,graph=sess.graph)

        for epoch_num in xrange(FLAGS.train_epoch_num):
            train_one_epoch(ops,pls,sess,summary_writer,train_provider,epoch_num,feed_dict)
            test_one_epoch(ops,pls,sess,saver,test_provider,epoch_num,feed_dict)

    finally:
        train_provider.close()
        test_provider.close()


def test_data_iter():
    train_list=['data/ModelNetTrain/nidxs/ply_data_train{}.h5'.format(i) for i in xrange(5)]
    train_list+=['data/ModelNetTrain/voxel/ply_data_test_voxel0.h5']
    test_list=['data/ModelNetTrain/voxel/ply_data_test_voxel1.h5']

    train_provider = ProviderV2(train_list,'train',FLAGS.batch_size*FLAGS.num_gpus,batch_fn,read_fn,2)
    # test_provider = ProviderV2(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,batch_fn,read_fn,2)

    begin=time.time()
    for data in train_provider:
        print 'cost {} s'.format(time.time()-begin)
        begin=time.time()



if __name__=="__main__":
    train()