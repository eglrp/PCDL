import argparse
import time
import numpy as np
import os

import tensorflow as tf
import random

from autoencoder_network import vanilla_pointnet_encoder,chamfer_loss,folding_net_decoder
from s3dis.draw_util import output_points
from s3dis.block_util import read_block_v2
from s3dis.data_util import get_train_test_split

from train_util import log_str,average_gradients
from provider import ProviderV2

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=4, help='')
parser.add_argument('--batch_size', type=int, default=8, help='')

parser.add_argument('--lr_init', type=float, default=1e-4, help='')
parser.add_argument('--decay_rate', type=float, default=0.9, help='')
parser.add_argument('--decay_epoch', type=int, default=10, help='')

parser.add_argument('--log_step', type=int, default=100, help='')
parser.add_argument('--train_dir', type=str, default='train', help='')
parser.add_argument('--save_dir', type=str, default='model', help='')
parser.add_argument('--log_file', type=str, default='train.log', help='')

parser.add_argument('--dump_dir', type=str, default='unsupervise', help='')
parser.add_argument('--dump_num', type=int, default=10, help='')

parser.add_argument('--train_epoch_num', type=int, default=500, help='')
FLAGS = parser.parse_args()


def tower_loss(points,covars,grids,color_ratio,reuse=False,decode_dim=6):
    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
        points_covars=tf.concat([points,covars],axis=2)
        codewords=vanilla_pointnet_encoder(points_covars, reuse)
        gen_points=folding_net_decoder(codewords, grids, decode_dim, reuse)

    # ref_points,_=tf.split(points,2,axis=2)
    loss=chamfer_loss(points,gen_points)

    # loss=tf.add(dist_loss,color_ratio*color_loss,name='total_loss')
    tf.add_to_collection(tf.GraphKeys.LOSSES,loss)

    tf.summary.scalar(loss.op.name,loss)
    # tf.summary.scalar(dist_loss.op.name,dist_loss)
    # tf.summary.scalar(color_loss.op.name,color_loss)

    return loss,gen_points


def train_ops(points, covars, grids, epoch_batch_num):
    ops={}
    with tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        decay_steps=epoch_batch_num*FLAGS.decay_epoch
        lr=tf.train.exponential_decay(FLAGS.lr_init,global_step,decay_steps,FLAGS.decay_rate,staircase=True)

        decay_steps=epoch_batch_num*15
        reverse_color_ratio=tf.train.exponential_decay(0.99,global_step,decay_steps,0.9,staircase=True)

        color_ratio=tf.constant(1.0,tf.float32)-reverse_color_ratio

        tf.summary.scalar('learning rate',lr)
        tf.summary.scalar('color_ratio',color_ratio)

        opt=tf.train.AdamOptimizer(lr)

        tower_points=tf.split(points, FLAGS.num_gpus)
        tower_covars=tf.split(covars,FLAGS.num_gpus)

        reuse=False
        tower_grads=[]
        tower_losses=[]
        tower_gen_pts=[]
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('tower_{}'.format(i)):
                    loss,gen_pts=tower_loss(tower_points[i], tower_covars[i], grids, color_ratio, reuse, 3)
                    # print tf.trainable_variables()
                    grad=opt.compute_gradients(loss,tf.trainable_variables())
                    tower_grads.append(grad)
                    tower_losses.append(loss)
                    tower_gen_pts.append(gen_pts)

                    reuse=True

        avg_grad=average_gradients(tower_grads)
        update_op=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_op):
            apply_grad_op=opt.apply_gradients(avg_grad,global_step=global_step)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge(summaries)

        total_loss_op=tf.add_n(tower_losses)

        gen_pts_op=tf.concat(tower_gen_pts,axis=0)

        ops['total_loss']=total_loss_op
        ops['apply_grad']=apply_grad_op
        ops['gen_pts']=gen_pts_op
        ops['summary']=summary_op
        ops['global_step']=global_step

    return ops


def batch_fn(file_data, cur_idx, data_indices, require_size):
    points,covars=file_data
    end_idx=min(cur_idx + require_size, points.shape[0])

    return [points[data_indices[cur_idx:end_idx],:],
           covars[data_indices[cur_idx:end_idx],:]],end_idx-cur_idx


def unpack_feats_labels(batch,num_gpus):
    points_list, covars_list = batch
    if points_list.shape[0]%num_gpus!=0:
        left_num=(points_list.shape[0]/num_gpus+1)*num_gpus-points_list.shape[0]
        left_idx = np.random.randint(0, points_list.shape[0], left_num)
        points_list=np.concatenate([points_list,points_list[left_idx,:]],axis=0)
        covars_list=np.concatenate([covars_list,covars_list[left_idx,:]],axis=0)

    return points_list, covars_list


def generate_grids():
    x=np.arange(-8,8)
    y=np.arange(-8,8)
    z=np.arange(0,16)
    X,Y,Z=np.meshgrid(x,y,z)
    grids=np.concatenate([X[:,:,:,None],Y[:,:,:,None],Z[:,:,:,None]],axis=3)
    grids=np.reshape(grids,[-1,3])
    grids=np.asarray(grids,dtype=np.float32)/8.0
    return grids


def train_one_epoch(ops,pls,sess,summary_writer,trainset,epoch_num,feed_dict):
    total=0
    begin_time=time.time()
    for i,feed_in in enumerate(trainset):
        points_list, covars_list=unpack_feats_labels(feed_in,FLAGS.num_gpus)

        feed_dict[pls['points']]=points_list[:,:,:3]
        feed_dict[pls['covars']]=covars_list
        total+=points_list.shape[0]

        sess.run(ops['apply_grad'],feed_dict)

        if i % FLAGS.log_step==0:
            total_loss,summary,global_step=sess.run(
                [ops['total_loss'],ops['summary'],ops['global_step']],feed_dict)

            log_str('epoch {} step {} loss {:.5} | {:.5} examples/s'.format(
                epoch_num,i,total_loss/FLAGS.num_gpus,float(total)/(time.time()-begin_time)
            ),FLAGS.log_file)

            summary_writer.add_summary(summary,global_step)
            total=0
            begin_time=time.time()


def test_one_epoch(ops,pls,sess,saver,testset,epoch_num,feed_dict):
    total=0
    begin_time=time.time()
    test_loss=[]
    left_size=FLAGS.dump_num
    for i,feed_in in enumerate(testset):
        points_list, covars_list=unpack_feats_labels(feed_in,FLAGS.num_gpus)

        feed_dict[pls['points']]=points_list[:,:,:3]
        feed_dict[pls['covars']]=covars_list
        total+=points_list.shape[0]

        loss,gen_pts=sess.run([ops['total_loss'],ops['gen_pts']],feed_dict)
        test_loss.append(loss/FLAGS.num_gpus)

        # output generated points
        if left_size>0 and random.random()<0.3:
            idx=np.random.randint(0,points_list.shape[0],dtype=np.int)
            # colors=np.asarray(points_list[idx,:,3:]*128+128,dtype=np.int)
            # fn=os.path.join(FLAGS.dump_dir,'{}_{}_true.txt'.format(epoch_num,left_size))
            # output_points(fn,points_list[idx,:,:3],colors)
            fn=os.path.join(FLAGS.dump_dir,'{}_{}_true.txt'.format(epoch_num,left_size))
            output_points(fn,points_list[idx,:,:3])

            # colors=np.asarray(gen_pts[idx,:,3:]*128+128,dtype=np.int)
            # colors[colors>255]=255
            # colors[colors<0]=0
            # fn=os.path.join(FLAGS.dump_dir,'{}_{}_recon.txt'.format(epoch_num,left_size))
            # output_points(fn,gen_pts[idx,:,:3],colors)
            fn=os.path.join(FLAGS.dump_dir,'{}_{}_recon.txt'.format(epoch_num,left_size))
            output_points(fn,gen_pts[idx,:,:3])
            left_size-=1

    test_loss=np.mean(np.asarray(test_loss))
    log_str('epoch {} test_loss {} cost {} s'.format(epoch_num,test_loss,time.time()-begin_time),FLAGS.log_file)

    checkpoint_path = os.path.join(FLAGS.save_dir, 'unsupervise{}.ckpt'.format(epoch_num))
    saver.save(sess,checkpoint_path)


def train():
    pt_num=4096

    # train_list,test_list = prepare_train_test_v2()
    # total_size=len(train_list)

    # train_list=['data/S3DIS/folding/block_train/block_train{}.h5'.format(i) for i in range(22)]
    # test_list=['data/S3DIS/folding/block_train/block_test{}.h5'.format(i) for i in range(9)]
    train_list,test_list=get_train_test_split()
    train_list=['data/S3DIS/folding/block_v2/'+fn+'.h5' for fn in train_list]
    test_list=['data/S3DIS/folding/block_v2/'+fn+'.h5' for fn in test_list]

    read_fn=lambda fn: read_block_v2(fn)[:2]

    train_provider = ProviderV2(train_list,'train',FLAGS.batch_size*FLAGS.num_gpus,batch_fn,read_fn,2)
    test_provider = ProviderV2(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,batch_fn,read_fn,2)

    try:
        pls={}
        pls['points']=tf.placeholder(tf.float32,[None,pt_num,3],'points')
        pls['covars']=tf.placeholder(tf.float32,[None,pt_num,9],'covars')
        pls['grids']=tf.placeholder(tf.float32,[4096,3],'grids')
        ops=train_ops(pls['points'],pls['covars'],pls['grids'],22000/(FLAGS.batch_size*FLAGS.num_gpus))

        # compute grids
        feed_dict = {}
        feed_dict[pls['grids']]=generate_grids()

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
    train_list,test_list=get_train_test_split()
    train_list=['data/S3DIS/folding/block_v2/'+fn+'.h5' for fn in train_list]
    read_fn=lambda fn: read_block_v2(fn)[:2]

    train_provider = ProviderV2(train_list,'train',20,batch_fn,read_fn,2)
    for data in train_provider:
        print data[0].shape,data[1].shape

    for data in train_provider:
        print data[0].shape, data[1].shape

    train_provider.close()

if __name__=="__main__":
    train()