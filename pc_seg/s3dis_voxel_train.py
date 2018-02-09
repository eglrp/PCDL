import argparse
import time
import numpy as np
import os

import tensorflow as tf
import random

from autoencoder_network import vanilla_pointnet_encoder,fc_voxel_decoder,voxel_color_loss,voxel_filling_loss
from s3dis.draw_util import output_points
from s3dis.block_util import read_block_v2
from s3dis.data_util import get_train_test_split
from s3dis.voxel_util import points2voxel_color_gpu, voxel2points

from train_util import log_str,average_gradients
from provider import ProviderV2

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=4, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')

parser.add_argument('--lr_init', type=float, default=1e-4, help='')
parser.add_argument('--decay_rate', type=float, default=0.9, help='')
parser.add_argument('--decay_epoch', type=int, default=10, help='')

parser.add_argument('--log_step', type=int, default=20, help='')
parser.add_argument('--train_dir', type=str, default='train/voxel', help='')
parser.add_argument('--save_dir', type=str, default='model/voxel', help='')
parser.add_argument('--log_file', type=str, default='train_voxel.log', help='')

parser.add_argument('--dump_dir', type=str, default='unsupervise/voxel', help='')
parser.add_argument('--dump_num', type=int, default=10, help='')
parser.add_argument('--split_num', type=int, default=30, help='')

parser.add_argument('--train_epoch_num', type=int, default=500, help='')
FLAGS = parser.parse_args()


def tower_loss(points, covars, true_color, true_state, voxel_num, reuse=False):
    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
        points=tf.concat([points,covars],axis=2)
        codewords,_=vanilla_pointnet_encoder(points, reuse)
        voxel_state,voxel_color=fc_voxel_decoder(codewords, voxel_num, True, reuse)

    color_loss=voxel_color_loss(voxel_color, true_state, true_color)
    filling_loss=voxel_filling_loss(voxel_state, true_state)
    loss=tf.add(color_loss,filling_loss,name='total_loss')
    tf.add_to_collection(tf.GraphKeys.LOSSES,loss)

    tf.summary.scalar(color_loss.op.name,color_loss)
    tf.summary.scalar(filling_loss.op.name,filling_loss)
    tf.summary.scalar(loss.op.name,loss)

    return loss,voxel_state,voxel_color


def train_ops(points, covars, true_color, true_state, voxel_num, epoch_batch_num):
    ops={}
    with tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        decay_steps=epoch_batch_num*FLAGS.decay_epoch
        lr=tf.train.exponential_decay(FLAGS.lr_init,global_step,decay_steps,FLAGS.decay_rate,staircase=True)
        tf.summary.scalar('learning rate',lr)

        opt=tf.train.AdamOptimizer(lr)

        tower_points=tf.split(points, FLAGS.num_gpus)
        tower_covars=tf.split(covars,FLAGS.num_gpus)
        tower_true_color=tf.split(true_color,FLAGS.num_gpus)
        tower_true_state=tf.split(true_state,FLAGS.num_gpus)

        reuse=False
        tower_grads=[]
        tower_losses=[]
        tower_voxel_state=[]
        tower_voxel_color=[]
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('tower_{}'.format(i)):
                    loss,voxel_state,voxel_color=tower_loss(tower_points[i], tower_covars[i],
                                                            tower_true_color[i], tower_true_state[i],
                                                            voxel_num, reuse)
                    # print tf.trainable_variables()
                    grad=opt.compute_gradients(loss,tf.trainable_variables())
                    tower_grads.append(grad)
                    tower_losses.append(loss)
                    tower_voxel_state.append(voxel_state)
                    tower_voxel_color.append(voxel_color)

                    reuse=True

        avg_grad=average_gradients(tower_grads)
        update_op=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_op):
            apply_grad_op=opt.apply_gradients(avg_grad,global_step=global_step)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge(summaries)

        total_loss_op=tf.add_n(tower_losses)

        voxel_color_op=tf.concat(tower_voxel_color,axis=0)
        voxel_state_op=tf.concat(tower_voxel_state,axis=0)

        ops['total_loss']=total_loss_op
        ops['apply_grad']=apply_grad_op
        ops['voxel_color']=voxel_color_op
        ops['voxel_state']=voxel_state_op
        ops['summary']=summary_op
        ops['global_step']=global_step

    return ops


def read_fn(model,filename):
    points,covars=read_block_v2(filename)[:2]
    voxel_state,voxel_color=points2voxel_color_gpu(points,FLAGS.split_num,1)

    return points, covars, voxel_state, voxel_color


def batch_fn(file_data, cur_idx, data_indices, require_size):
    points, covars, voxel_state, voxel_color = file_data
    end_idx = min(cur_idx + require_size, points.shape[0])

    return [points[data_indices[cur_idx:end_idx], :, :],
            covars[data_indices[cur_idx:end_idx], :, :],
            voxel_state[data_indices[cur_idx:end_idx], :],
            voxel_color[data_indices[cur_idx:end_idx], :, :]
            ], end_idx - cur_idx


def unpack_feats_labels(batch,num_gpus):
    points_list, covars_list, voxel_state_list, voxel_color_list = batch
    if points_list.shape[0]%num_gpus!=0:
        left_num=(points_list.shape[0]/num_gpus+1)*num_gpus-points_list.shape[0]
        left_idx = np.random.randint(0, points_list.shape[0], left_num)
        points_list=np.concatenate([points_list,points_list[left_idx,:]],axis=0)
        covars_list=np.concatenate([covars_list,covars_list[left_idx,:]],axis=0)
        voxel_state_list=np.concatenate([voxel_state_list,voxel_state_list[left_idx,:]],axis=0)
        voxel_color_list=np.concatenate([voxel_color_list,voxel_color_list[left_idx,:]],axis=0)

    return points_list, covars_list, voxel_state_list, voxel_color_list


def train_one_epoch(ops,pls,sess,summary_writer,trainset,epoch_num,feed_dict):
    total=0
    begin_time=time.time()
    for i,feed_in in enumerate(trainset):
        points_list, covars_list, voxel_state_list, voxel_color_list=\
            unpack_feats_labels(feed_in,FLAGS.num_gpus)

        feed_dict[pls['points']]=points_list
        feed_dict[pls['covars']]=covars_list
        feed_dict[pls['voxel_state']]=voxel_state_list
        feed_dict[pls['voxel_color']]=voxel_color_list
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
        points_list, covars_list, voxel_state_list, voxel_color_list=\
            unpack_feats_labels(feed_in,FLAGS.num_gpus)

        feed_dict[pls['points']]=points_list
        feed_dict[pls['covars']]=covars_list
        feed_dict[pls['voxel_state']]=voxel_state_list
        feed_dict[pls['voxel_color']]=voxel_color_list
        total+=points_list.shape[0]

        loss,gen_state,gen_color=sess.run([ops['total_loss'],ops['voxel_state'],ops['voxel_color']],feed_dict)
        test_loss.append(loss/FLAGS.num_gpus)

        # output generated voxels
        for i in range(3):
            if left_size>0 and random.random()<0.9:
                idx=np.random.randint(0,points_list.shape[0],dtype=np.int)

                pts=points_list[idx,:,:]
                pts[:,:2]+=0.5
                pts[:,3:]+=1.0
                pts[:,3:]*=127
                fn=os.path.join(FLAGS.dump_dir,'{}_{}_points.txt'.format(epoch_num,left_size))
                output_points(fn,pts)

                true_state_pts=voxel2points(voxel_state_list[idx])
                fn=os.path.join(FLAGS.dump_dir,'{}_{}_state_true.txt'.format(epoch_num,left_size))
                output_points(fn,true_state_pts)

                gen_state[idx][gen_state[idx]<0.0]=0.0
                gen_state[idx][gen_state[idx]>1.0]=1.0
                pred_state_pts=voxel2points(gen_state[idx])
                fn=os.path.join(FLAGS.dump_dir,'{}_{}_state_pred.txt'.format(epoch_num,left_size))
                output_points(fn,pred_state_pts)

                true_color_pts=voxel2points(voxel_color_list[idx])
                fn = os.path.join(FLAGS.dump_dir, '{}_{}_color_true.txt'.format(epoch_num, left_size))
                output_points(fn, true_color_pts)

                gen_color[idx][gen_color[idx]<0.0]=0.0
                gen_color[idx][gen_color[idx]>1.0]=1.0
                pred_color_pts=voxel2points(gen_color[idx])
                fn = os.path.join(FLAGS.dump_dir, '{}_{}_color_pred.txt'.format(epoch_num, left_size))
                output_points(fn, pred_color_pts)

                left_size-=1

    test_loss=np.mean(np.asarray(test_loss))
    log_str('epoch {} test_loss {} cost {} s'.format(epoch_num,test_loss,time.time()-begin_time),FLAGS.log_file)

    checkpoint_path = os.path.join(FLAGS.save_dir, 'unsupervise{}.ckpt'.format(epoch_num))
    saver.save(sess,checkpoint_path)


def train():
    pt_num=4096
    voxel_num=FLAGS.split_num*FLAGS.split_num*FLAGS.split_num

    train_list,test_list=get_train_test_split()
    train_list+=test_list
    train_list=['data/S3DIS/folding/block_v2/{}.h5'.format(fn) for fn in train_list]
    test_list=['data/S3DIS/folding/block_v2/{}.h5'.format(fn) for fn in test_list]

    train_provider = ProviderV2(train_list,'train',FLAGS.batch_size*FLAGS.num_gpus,batch_fn,read_fn,2)
    test_provider = ProviderV2(test_list[:5],'test',FLAGS.batch_size*FLAGS.num_gpus,batch_fn,read_fn,2)

    try:
        pls={}
        pls['points']=tf.placeholder(tf.float32,[None,pt_num,6],'points')
        pls['covars']=tf.placeholder(tf.float32,[None,pt_num,9],'covars')
        pls['voxel_state']=tf.placeholder(tf.float32,[None,voxel_num],'voxel_state')
        pls['voxel_color']=tf.placeholder(tf.float32,[None,voxel_num,3],'voxel_color')
        ops=train_ops(pls['points'],pls['covars'],pls['voxel_color'],pls['voxel_state'],
                      voxel_num,26000/(FLAGS.batch_size*FLAGS.num_gpus))

        # compute grids
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
    train_list,test_list=get_train_test_split()
    train_list+=test_list
    train_list=['data/S3DIS/folding/block_v2/{}.h5'.format(fn) for fn in train_list]
    # test_list=['data/S3DIS/folding/block_v2/{}.h5'.format(fn) for fn in test_list]

    train_provider = ProviderV2(train_list,'train',32,batch_fn,read_fn,2)

    begin=time.time()
    for data in train_provider:
        for item in data:
            print item.shape
        print 'cost {} s'.format(time.time()-begin)
        begin=time.time()
        # print data[0].shape,data[1].shape

    begin=time.time()
    for data in train_provider:
        print 'cost {} s'.format(time.time()-begin)
        begin=time.time()
        # print data[0].shape, data[1].shape

    train_provider.close()

if __name__=="__main__":
    train()