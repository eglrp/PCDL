import tensorflow as tf
import tensorflow.contrib.framework as framework


def graph_pool(feats,nidxs):
    '''

    :param feats: n,k,f
    :param nidxs: n,k,t,2
    :return:
    '''
    feats=tf.gather_nd(feats,nidxs)               # n,k,t,f
    pooled_feats=tf.reduce_max(feats,axis=3)      # n,k,f
    # pooled_feats=tf.nn.relu(pooled_feats)       # n,k,f todo: Is the relu here useful?
    return pooled_feats


def revise_nidxs(nidxs):
    '''
    :param nidxs: n,k,t
    :return: n,k,t,2
    '''
    n,k,t=tf.shape(nidxs)[0],tf.shape(nidxs)[1],tf.shape(nidxs)[2]
    nidxs=tf.expand_dims(nidxs,axis=3)                     # n,k,t,1
    batch_idxs=tf.range(n,dtype=tf.int32)
    batch_idxs=tf.expand_dims(batch_idxs,axis=1)           # n,1,1,1
    batch_idxs=tf.expand_dims(batch_idxs,axis=2)           # n,1,1,1
    batch_idxs=tf.expand_dims(batch_idxs,axis=3)           # n,1,1,1
    batch_idxs=tf.tile(batch_idxs,[1,k,t,1])               # n,k,t,1
    nidxs=tf.concat([batch_idxs,nidxs],axis=3)

    return nidxs


def folding_net_encoder(points, covariance, nidxs, reuse=False):
    '''
    :param points: n,k,3
    :param covariance: n,k,9
    :param nidxs:  n,k,t
    :return:
    '''
    with tf.name_scope('encoder'):
        points_covar=tf.concat([points,covariance],axis=2)
        points_covar=tf.expand_dims(points_covar,axis=2)    # n,k,1,12
        nidxs=revise_nidxs(nidxs)     # n,k,t,2

        with framework.arg_scope([tf.contrib.layers.conv2d],kernel_size=[1,1],stride=1,
                                 padding='VALID',activation_fn=tf.nn.relu,reuse=reuse,
                                 ):
            with tf.name_scope('point_perceptron'):
                point_mlp1=tf.contrib.layers.conv2d(points_covar,num_outputs=64,scope='point_mlp1')       # n,k,1,64
                point_mlp2=tf.contrib.layers.conv2d(point_mlp1,num_outputs=64,scope='point_mlp2')
                point_mlp3=tf.contrib.layers.conv2d(point_mlp2,num_outputs=64,scope='point_mlp3')         # n,k,1,64

            with tf.name_scope('graph'):
                point_mlp3=tf.squeeze(point_mlp3,axis=2)    # n,k,64
                local_feats1=graph_pool(point_mlp3,nidxs)
                local_feats1=tf.expand_dims(local_feats1,axis=2)    # n,k,1,64
                local_feats1=tf.contrib.layers.conv2d(local_feats1, num_outputs=256, scope='graph_mlp1')

                local_feats1=tf.squeeze(local_feats1,axis=2)
                local_feats2=graph_pool(local_feats1,nidxs)
                local_feats2=tf.expand_dims(local_feats2,axis=2)    # n,k,1,256
                local_feats2=tf.contrib.layers.conv2d(local_feats2, num_outputs=1024,
                                                      activation_fn=None, scope='graph_mlp2')

            with tf.name_scope('global_perceptron'):
                local_feats2=tf.squeeze(local_feats2,axis=2)              # n,k,1024
                global_feats=tf.reduce_max(local_feats2,axis=1)           # n,1024

                global_mlp1=tf.contrib.layers.fully_connected(global_feats,num_outputs=1024,
                                                              reuse=reuse,scope='global_mlp1')

                global_mlp2=tf.contrib.layers.fully_connected(global_mlp1,num_outputs=512,reuse=reuse,
                                                              activation_fn=None,scope='global_mlp2')

    return global_mlp2


def vanilla_pointnet_encoder(points, reuse=False, trainable=True):
    '''
    :param points: n,k,6 xyzrgb
    :param reuse:
    :return:
    '''
    points=tf.expand_dims(points,axis=2)    # n,k,1,6
    with tf.name_scope('encoder'):
        with framework.arg_scope([tf.contrib.layers.conv2d],kernel_size=[1,1],stride=1,
                                 padding='VALID',activation_fn=tf.nn.relu,reuse=reuse,trainable=trainable
                                 ):
            point_mlp1 = tf.contrib.layers.conv2d(points, num_outputs=64, scope='point_mlp1')
            point_mlp2 = tf.contrib.layers.conv2d(point_mlp1, num_outputs=64, scope='point_mlp2')
            point_mlp3 = tf.contrib.layers.conv2d(point_mlp2, num_outputs=64, scope='point_mlp3')
            point_mlp4 = tf.contrib.layers.conv2d(point_mlp3, num_outputs=128, scope='point_mlp4')
            point_mlp5 = tf.contrib.layers.conv2d(point_mlp4, num_outputs=512, scope='point_mlp5')
            point_mlp6 = tf.contrib.layers.conv2d(point_mlp5, num_outputs=1024, scope='point_mlp6',activation_fn=None)
            # point_mlp5 = tf.contrib.layers.conv2d(point_mlp4, num_outputs=1024, scope='point_mlp5')
            # point_mlp6 = tf.contrib.layers.conv2d(point_mlp5, num_outputs=512, scope='point_mlp6',activation_fn=None)

        point_mlp6=tf.squeeze(point_mlp6,axis=2)
        codewords=tf.reduce_max(point_mlp6,axis=1)

    return codewords,point_mlp6


def concat_pointnet_encoder(points, reuse=False, trainable=True, final_dim=512):
    '''
    :param points: n,k,6 xyzrgb
    :param reuse:
    :param trainable:
    :param final_dim:
    :return:
    '''
    points = tf.expand_dims(points, axis=2)  # n,k,1,6
    with tf.name_scope('encoder'):
        with framework.arg_scope([tf.contrib.layers.conv2d], kernel_size=[1, 1], stride=1,
                                 padding='VALID', activation_fn=tf.nn.relu, reuse=reuse, trainable=trainable
                                 ):
            point_mlp1 = tf.contrib.layers.conv2d(points, num_outputs=64, scope='point_mlp1')
            point_mlp2 = tf.contrib.layers.conv2d(point_mlp1, num_outputs=64, scope='point_mlp2')
            point_mlp3 = tf.contrib.layers.conv2d(point_mlp2, num_outputs=64, scope='point_mlp3')
            point_mlp3 = tf.concat([point_mlp3, points], axis=3)

            point_mlp4 = tf.contrib.layers.conv2d(point_mlp3, num_outputs=128, scope='point_mlp4')
            point_mlp5 = tf.contrib.layers.conv2d(point_mlp4, num_outputs=1024, scope='point_mlp5')
            point_mlp5 = tf.concat([point_mlp5, points], axis=3)

            point_mlp6 = tf.contrib.layers.conv2d(point_mlp5, num_outputs=final_dim, scope='point_mlp6',
                                                  activation_fn=None)

        point_mlp6 = tf.squeeze(point_mlp6, axis=2)
        codewords = tf.reduce_max(point_mlp6, axis=1)

    return codewords, point_mlp6


def concat_pointnet_encoder_v2(points, is_training, reuse=False, trainable=True, final_dim=512, use_bn=True):
    '''
    :param points: n,k,6 xyzrgb
    :param reuse:
    :param trainable:
    :param final_dim:
    :param use_bn:
    :return:
    '''
    points = tf.expand_dims(points, axis=2)  # n,k,1,6
    normalizer_params={'scale':False,'is_training':is_training,'reuse':reuse}
    bn=tf.contrib.layers.batch_norm if use_bn else None
    with tf.name_scope('encoder'):
        with framework.arg_scope([tf.contrib.layers.conv2d], kernel_size=[1, 1], stride=1,
                                 padding='VALID', activation_fn=tf.nn.relu, reuse=reuse, trainable=trainable,
                                 normalizer_fn=bn):

            normalizer_params['scope']='point_mlp1_bn'
            point_mlp1 = tf.contrib.layers.conv2d(
                points, num_outputs=64, scope='point_mlp1',normalizer_params=normalizer_params)
            point_mlp1 = tf.concat([point_mlp1, points], axis=3)

            normalizer_params['scope']='point_mlp2_bn'
            point_mlp2 = tf.contrib.layers.conv2d(
                point_mlp1, num_outputs=64, scope='point_mlp2',normalizer_params=normalizer_params)
            point_mlp2 = tf.concat([point_mlp2, points], axis=3)

            normalizer_params['scope']='point_mlp3_bn'
            point_mlp3 = tf.contrib.layers.conv2d(
                point_mlp2, num_outputs=64, scope='point_mlp3',normalizer_params=normalizer_params)
            point_mlp3 = tf.concat([point_mlp3, points], axis=3)

            normalizer_params['scope']='point_mlp4_bn'
            point_mlp4 = tf.contrib.layers.conv2d(
                point_mlp3, num_outputs=128, scope='point_mlp4',normalizer_params=normalizer_params)
            point_mlp4 = tf.concat([point_mlp4, points], axis=3)

            normalizer_params['scope']='point_mlp5_bn'
            point_mlp5 = tf.contrib.layers.conv2d(
                point_mlp4, num_outputs=final_dim, scope='point_mlp5',activation_fn=None,normalizer_fn=None)

        point_mlp5 = tf.squeeze(point_mlp5, axis=2)     # n,k,f
        codewords = tf.reduce_max(point_mlp5, axis=1)   # n,f

    return codewords, point_mlp5


def all_concat_pointnet_encoder(points, is_training, reuse=False, trainable=True, final_dim=1024, use_bn=True):
    '''
    used for 1_19 overall acc 89.4% mean acc 87.0% epoch 102
    :param points: n,k,6 xyzrgb
    :param reuse:
    :param trainable:
    :param final_dim:
    :param use_bn:
    :return:
    '''
    points = tf.expand_dims(points, axis=2)  # n,k,1,6
    normalizer_params={'scale':False,'is_training':is_training,'reuse':reuse}
    bn=tf.contrib.layers.batch_norm if use_bn else None
    with tf.name_scope('encoder'):
        with framework.arg_scope([tf.contrib.layers.conv2d], kernel_size=[1, 1], stride=1,
                                 padding='VALID', activation_fn=tf.nn.relu, reuse=reuse, trainable=trainable,
                                 normalizer_fn=bn):

            normalizer_params['scope']='point_mlp1_bn'
            point_mlp1 = tf.contrib.layers.conv2d(
                points, num_outputs=64, scope='point_mlp1',normalizer_params=normalizer_params)
            point_mlp1 = tf.concat([point_mlp1, points], axis=3)

            normalizer_params['scope']='point_mlp2_bn'
            point_mlp2 = tf.contrib.layers.conv2d(
                point_mlp1, num_outputs=64, scope='point_mlp2',normalizer_params=normalizer_params)
            point_mlp2 = tf.concat([point_mlp2, point_mlp1], axis=3)

            normalizer_params['scope']='point_mlp3_bn'
            point_mlp3 = tf.contrib.layers.conv2d(
                point_mlp2, num_outputs=64, scope='point_mlp3',normalizer_params=normalizer_params)
            point_mlp3 = tf.concat([point_mlp3, point_mlp2], axis=3)

            normalizer_params['scope']='point_mlp4_bn'
            point_mlp4 = tf.contrib.layers.conv2d(
                point_mlp3, num_outputs=128, scope='point_mlp4',normalizer_params=normalizer_params)
            point_mlp4 = tf.concat([point_mlp4, point_mlp3], axis=3)

            normalizer_params['scope']='point_mlp5_bn'
            point_mlp5 = tf.contrib.layers.conv2d(
                point_mlp4, num_outputs=final_dim, scope='point_mlp5',activation_fn=None,normalizer_fn=None)

        point_mlp5 = tf.squeeze(point_mlp5, axis=2)
        codewords = tf.reduce_max(point_mlp5, axis=1)

    return codewords, point_mlp5


def folding_net_decoder(codewords, grids, decode_dim=3, reuse=False):
    '''
    :param codewords:   n,f f=512
    :param grids:       m, 2 or 3
    :return:
    '''
    with tf.name_scope('decoder'):
        m=tf.shape(grids)[0]
        n=tf.shape(codewords)[0]

        codewords=tf.expand_dims(codewords,axis=1)  # n,1,f
        codewords=tf.tile(codewords,[1,m,1])        # n,m,f

        grids=tf.expand_dims(grids,axis=0)      # 1,m,2
        grids=tf.tile(grids,[n,1,1])            # n,m,2

        with framework.arg_scope([tf.contrib.layers.conv2d],kernel_size=[1,1],stride=1,
                                 padding='VALID',activation_fn=tf.nn.relu,reuse=reuse,
                                 ):
            with tf.name_scope('folding_stage1'):
                folding_feats1 = tf.concat([codewords, grids], axis=2)          # n,m,f+2
                folding_feats1 = tf.expand_dims(folding_feats1, axis=2)         # n,m,1,f+2

                folding_mlp1=tf.contrib.layers.conv2d(folding_feats1,num_outputs=512,scope='folding_mlp1')
                folding_mlp2=tf.contrib.layers.conv2d(folding_mlp1,num_outputs=512,scope='folding_mlp2')
                folding_points1=tf.contrib.layers.conv2d(folding_mlp2,num_outputs=decode_dim,
                                                         scope='folding_points1',activation_fn=None)    # n,m,1,decode_dim

            with tf.name_scope('folding_stage2'):
                folding_points1=tf.squeeze(folding_points1,axis=2)
                folding_feats2=tf.concat([codewords,folding_points1],axis=2)     # n,m,f+3
                folding_feats2=tf.expand_dims(folding_feats2,axis=2)

                folding_mlp3=tf.contrib.layers.conv2d(folding_feats2,num_outputs=512,scope='folding_mlp3')
                folding_mlp4=tf.contrib.layers.conv2d(folding_mlp3,num_outputs=512,scope='folding_mlp4')
                folding_points2=tf.contrib.layers.conv2d(folding_mlp4,num_outputs=decode_dim,
                                                         scope='folding_points2',activation_fn=None)    # n,m,1,decode_dim

        folding_points2=tf.squeeze(folding_points2,axis=2)

    return folding_points2


def fc_voxel_decoder(codewords, voxel_num=27000, generate_color=False, reuse=False):
    '''
    :param codewords: n,512
    :param reuse:
    :param voxel_num:
    :param decode_dim:
    :return:
    '''
    with tf.name_scope('decoder'):
        with framework.arg_scope([tf.contrib.layers.fully_connected],activation_fn=tf.nn.relu,reuse=reuse):
            fc1=tf.contrib.layers.fully_connected(codewords,num_outputs=512,scope='fc1')
            fc2=tf.contrib.layers.fully_connected(fc1,num_outputs=512,scope='fc2')
            fc3=tf.contrib.layers.fully_connected(fc2,num_outputs=512,scope='fc3')
            fc4=tf.contrib.layers.fully_connected(fc3,num_outputs=512,scope='fc4')
            fc5=tf.contrib.layers.fully_connected(fc4,num_outputs=512,scope='fc5')
            if not generate_color:
                voxel_state=tf.contrib.layers.fully_connected(fc5,num_outputs=voxel_num,scope='voxel_state',activation_fn=None)
                voxel_state=tf.sigmoid(voxel_state)
                return voxel_state
            else:
                voxels=tf.contrib.layers.fully_connected(fc5,num_outputs=voxel_num*4,scope='voxel_state',activation_fn=None)
                voxels=tf.reshape(voxels,[-1,voxel_num,4])
                voxel_state,voxel_color=tf.split(voxels,[1,3],axis=2)

                voxel_state=tf.squeeze(voxel_state,axis=2)
                voxel_state=tf.sigmoid(voxel_state)

                return voxel_state,voxel_color


def fc_voxel_decoder_v2(codewords, is_training, voxel_num=27000, generate_color=False, reuse=False, use_bn=True):
    '''
    :param codewords: n,512
    :param reuse:
    :param voxel_num:
    :param decode_dim:
    :return:
    '''
    bn=tf.contrib.layers.batch_norm if use_bn else None
    normalizer_params={'scale':False,'is_training':is_training,'reuse':reuse}
    with tf.name_scope('decoder'):
        with framework.arg_scope([tf.contrib.layers.fully_connected],
                                 activation_fn=tf.nn.relu,reuse=reuse,
                                 normalizer_fn=bn):

            normalizer_params['scope']='fc1_bn'
            fc1=tf.contrib.layers.fully_connected(
                codewords,num_outputs=1024,scope='fc1',normalizer_params=normalizer_params)

            normalizer_params['scope']='fc2_bn'
            fc2=tf.contrib.layers.fully_connected(
                fc1,num_outputs=1024,scope='fc2',normalizer_params=normalizer_params)

            normalizer_params['scope']='fc3_bn'
            fc3=tf.contrib.layers.fully_connected(
                fc2,num_outputs=1024,scope='fc3',normalizer_params=normalizer_params)

            if not generate_color:
                voxel_state=tf.contrib.layers.fully_connected(
                    fc3,num_outputs=voxel_num,scope='voxel_state',activation_fn=None,normalizer_params=None)
                voxel_state=tf.sigmoid(voxel_state)
                return voxel_state,fc3
            else:
                voxels=tf.contrib.layers.fully_connected(
                    fc3,num_outputs=voxel_num*4,scope='voxel_state',activation_fn=None,normalizer_params=None)

                voxels=tf.reshape(voxels,[-1,voxel_num,4])
                voxel_state,voxel_color=tf.split(voxels,[1,3],axis=2)

                voxel_state=tf.squeeze(voxel_state,axis=2)
                voxel_state=tf.sigmoid(voxel_state)

                return voxel_state,voxel_color,fc3


def voxel_filling_loss(voxel_state, true_state):
    '''
    :param voxel_state:  n,voxel_num
    :param true_state:  n,voxel_num [0,1]
    :return:
    '''
    with tf.name_scope('filling_loss'):
        voxel_fill_loss=-tf.log(true_state*voxel_state+(1-true_state)*(1-voxel_state)+1e-7)     # n,voxel_num
        filling_loss=tf.reduce_mean(tf.reduce_mean(voxel_fill_loss,axis=1),name='filling_loss')

    return filling_loss


def voxel_color_loss(voxel_color, true_state, true_color):
    '''
    :param voxel_color: n,voxel_num,3
    :param true_state: n,voxel_num
    :param true_color:
    :return:
    '''
    with tf.name_scope('color_loss'):
        color_diff=tf.reduce_sum(tf.squared_difference(voxel_color,true_color),axis=2)            # n, voxel_num
        color_loss=tf.reduce_sum(true_state * color_diff, axis=1)         # n
        color_loss=tf.div(color_loss, tf.reduce_sum(true_state, axis=1))  # n/n
        color_loss=tf.reduce_mean(color_loss,name='color_loss')           # 1

    return color_loss


def chamfer_loss(ref_pts,gen_pts):
    '''
    :param ref_pts: n,k1,3
    :param gen_pts:  n,k2,3
    :return:
    '''
    with tf.name_scope('chamfer_loss'):
        k1=tf.shape(ref_pts)[1]
        k2=tf.shape(gen_pts)[1]

        ref_pts=tf.expand_dims(ref_pts,axis=1)
        ref_pts=tf.tile(ref_pts,[1,k2,1,1])
        gen_pts=tf.expand_dims(gen_pts,axis=2)
        gen_pts=tf.tile(gen_pts,[1,1,k1,1])

        dists=tf.squared_difference(ref_pts,gen_pts)    # n,k1,k2,3
        dists=tf.reduce_sum(dists,axis=3)               # n,k1,k2

        ref_dists=tf.sqrt(tf.reduce_min(dists,axis=2))        # n
        ref_dists=tf.reduce_mean(ref_dists,axis=1)            # 1

        gen_dists=tf.sqrt(tf.reduce_min(dists,axis=1))        # n
        gen_dists=tf.reduce_mean(gen_dists,axis=1)            # 1

        chamfer_dists=tf.maximum(ref_dists,gen_dists)         # 1

    return tf.reduce_mean(chamfer_dists,name='chamfer_dist_loss')


def chamfer_color_loss(ref_pts,gen_pts):
    '''
    :param ref_pts: n,k1,6  xyz rgb
    :param gen_pts:  n,k2,6 xyz rgb
    :return:
    '''
    k1=tf.shape(ref_pts)[1]
    k2=tf.shape(gen_pts)[1]
    n=tf.shape(gen_pts)[0]

    ref_pts,ref_clrs=tf.split(ref_pts,2,axis=2)
    gen_pts,gen_clrs=tf.split(gen_pts,2,axis=2)

    ref_pts=tf.expand_dims(ref_pts,axis=1)
    ref_pts=tf.tile(ref_pts,[1,k2,1,1])
    gen_pts=tf.expand_dims(gen_pts,axis=2)
    gen_pts=tf.tile(gen_pts,[1,1,k1,1])

    dists=tf.squared_difference(ref_pts,gen_pts)    # n,k2,k1,3
    dists=tf.reduce_sum(dists,axis=3)               # n,k2,k1

    ##########################
    ref_idxs=tf.expand_dims(tf.cast(tf.argmin(dists,axis=1),dtype=tf.int32),axis=2) # n,k1,1
    gen_idxs=tf.expand_dims(tf.cast(tf.argmin(dists,axis=2),dtype=tf.int32),axis=2) # n,k2,1

    batch_idxs=tf.expand_dims(tf.range(0,n),axis=1) # n,1
    batch_idxs=tf.expand_dims(batch_idxs,axis=2) # n,1,1
    ref_tile=tf.tile(batch_idxs,[1,k1,1])
    gen_tile=tf.tile(batch_idxs,[1,k2,1])

    ref_idxs=tf.concat([ref_tile,ref_idxs],axis=2)  # n,k1,2
    gen_idxs=tf.concat([gen_tile,gen_idxs],axis=2)  # n,k2,2

    ref_color_diff=tf.squared_difference(tf.gather_nd(gen_clrs,ref_idxs),ref_clrs)          # n,k1,3
    ref_color_diff=tf.reduce_mean(tf.sqrt(tf.reduce_sum(ref_color_diff,axis=2)),axis=1)     # n

    gen_color_diff=tf.squared_difference(tf.gather_nd(ref_clrs,gen_idxs),gen_clrs)           # n,k2,3
    gen_color_diff=tf.reduce_mean(tf.sqrt(tf.reduce_sum(gen_color_diff,axis=2)),axis=1)      # n

    color_loss=tf.reduce_mean(tf.maximum(ref_color_diff,gen_color_diff),name='color_loss')

    ###########################
    ref_dists=tf.sqrt(tf.reduce_min(dists,axis=1))        # n
    ref_dists=tf.reduce_mean(ref_dists,axis=1)            # 1

    gen_dists=tf.sqrt(tf.reduce_min(dists,axis=2))        # n
    gen_dists=tf.reduce_mean(gen_dists,axis=1)            # 1

    chamfer_dists=tf.maximum(ref_dists,gen_dists)         # 1

    dist_loss=tf.reduce_mean(chamfer_dists,name='dist_loss')

    return dist_loss,color_loss
