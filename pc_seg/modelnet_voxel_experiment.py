import tensorflow as tf
from autoencoder_network import fc_voxel_decoder_v2,voxel_filling_loss,concat_pointnet_encoder_v2
from modelnet.data_util import read_modelnet_v2,rotate
from s3dis.voxel_util import points2voxel_gpu_modelnet,voxel2points,points2covars_gpu
from s3dis.draw_util import output_points
import os
import numpy as np


def operators(points,covars,true_state,is_training,voxel_num,reuse=False):
    points_covars=tf.concat([points,covars],axis=2)                                                           # [n,k,16]
    global_feats,_=concat_pointnet_encoder_v2(points_covars,is_training, reuse, final_dim=1024,use_bn=False)  # [n,1024]
    # get reconstructed voxel
    voxel_state,_=fc_voxel_decoder_v2(global_feats, is_training, voxel_num, False, reuse, use_bn=False)

    # filling loss
    recon_loss=voxel_filling_loss(voxel_state, true_state)
    tf.add_to_collection(tf.GraphKeys.LOSSES,recon_loss)
    tf.summary.scalar(recon_loss.op.name,recon_loss)

    return recon_loss, voxel_state

def network(split_num=30):
    voxel_num=split_num**3

    pls={}
    pls['points']=tf.placeholder(tf.float32,[None,2048,3])
    pls['covars']=tf.placeholder(tf.float32,[None,2048,9])
    pls['states']=tf.placeholder(tf.float32,[None,voxel_num])
    pls['is_training']=tf.placeholder(tf.bool,name='is_training')

    global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)
    lr = tf.train.exponential_decay(1e-3, global_step, 1000, 0.75, staircase=True)
    lr = tf.maximum(1e-5, lr)
    loss,gen_state=operators(pls['points'],pls['covars'],pls['states'],pls['is_training'],voxel_num)
    opt=tf.train.AdamOptimizer(lr)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op=opt.minimize(loss,global_step=global_step)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ops={}
    ops['train']=train_op
    ops['gen_state']=gen_state
    ops['loss']=loss
    ops['lr']=lr

    return sess,pls,ops


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


if __name__=="__main__":
    train_num=10
    train_epoch=50000
    split_num=30

    output_epoch=1000
    log_epoch=30
    dump_dir='unsupervise/modelnet_voxel_experiment'

    sess,pls,ops=network(split_num)

    points, nidxs, labels=read_modelnet_v2('data/ModelNetTrain/nidxs/ply_data_train0.h5')
    points=points[:train_num]
    nidxs=nidxs[:train_num]
    labels=labels[:train_num]

    output_points(dump_dir+'/points.txt',points[0])

    for i in range(train_epoch):
        rot_points=rotate(points)
        # rot_points=np.copy(points)
        voxels=points2voxel_gpu_modelnet(rot_points,split_num,0)
        covars=points2covars_gpu(rot_points,nidxs,16,0)

        feed_dict={}
        feed_dict[pls['points']]=rot_points
        feed_dict[pls['covars']]=covars
        feed_dict[pls['states']]=voxels
        feed_dict[pls['is_training']]=True

        _,loss,gen_state,lr=sess.run([ops['train'],ops['loss'],ops['gen_state'],ops['lr']],feed_dict)

        if i % log_epoch==0:
            print 'loss {} lr {}'.format(loss,lr)

        if i % output_epoch==0:
            output_gen_points(None,voxels[0],gen_state[0],0,i,dump_dir)



