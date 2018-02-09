import argparse
import time
import numpy as np
import os

import tensorflow as tf

from autoencoder_network import concat_pointnet_encoder_v2,fc_voxel_decoder_v2,voxel_filling_loss
from classify_network import model_classifier_v2
from modelnet.data_util import read_modelnet_v2,compute_acc,rotate,exchange_dims_zy,jitter_point_cloud
from s3dis.voxel_util import points2voxel_gpu_modelnet,voxel2points,points2covars_gpu
from s3dis.draw_util import output_points

from train_util import log_str,average_gradients,unpack_feats_labels
from provider import ProviderV2

import random

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=2, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')

parser.add_argument('--lr_init', type=float, default=1e-3, help='')
parser.add_argument('--lr_clip', type=float, default=1e-5, help='')
parser.add_argument('--lr_point_ratio', type=float, default=0.1, help='')
parser.add_argument('--decay_rate', type=float, default=0.5, help='')
parser.add_argument('--decay_epoch', type=int, default=50, help='')
parser.add_argument('--num_classes', type=int, default=40, help='')

parser.add_argument('--recon_decay_rate', type=float, default=0.5, help='')
parser.add_argument('--recon_decay_epoch', type=int, default=100, help='')

parser.add_argument('--log_step', type=int, default=40, help='')
parser.add_argument('--train_dir', type=str, default='train/modelnet_voxel_label_sync', help='')
parser.add_argument('--save_dir', type=str, default='model/modelnet_voxel_label_sync', help='')
parser.add_argument('--log_file', type=str, default='modelnet_voxel_label_sync.log', help='')

parser.add_argument('--train_epoch_num', type=int, default=500, help='')

parser.add_argument('--dump_dir', type=str, default='unsupervise/modelnet_voxel_label_sync', help='')
parser.add_argument('--dump_num', type=int, default=10, help='')
parser.add_argument('--split_num', type=int, default=30, help='')

FLAGS = parser.parse_args()


def tower_loss(points, covars, labels, true_state, is_training, num_classes, voxel_num, reuse=False):
    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
        points_covars=tf.concat([points,covars],axis=2)                                                                         # [n,k,16]
        global_feats,_=concat_pointnet_encoder_v2(points_covars, is_training, reuse, final_dim=1024,use_bn=False)               # [n,1024]
        # get reconstructed voxel
        voxel_state,global_feats_v2=fc_voxel_decoder_v2(global_feats, is_training, voxel_num, False, reuse,use_bn=False)
        logits=model_classifier_v2(global_feats_v2,num_classes,is_training,reuse,use_bn=False)

    # softmax loss
    labels=tf.reshape(labels,[-1,1])
    labels=tf.squeeze(labels,axis=1)
    loss=tf.losses.sparse_softmax_cross_entropy(labels,logits)
    tf.add_to_collection(tf.GraphKeys.LOSSES,loss)
    tf.summary.scalar(loss.op.name,loss)

    # filling loss
    recon_loss=voxel_filling_loss(voxel_state, true_state)
    tf.add_to_collection(tf.GraphKeys.LOSSES,recon_loss)
    tf.summary.scalar(recon_loss.op.name,recon_loss)

    return loss,recon_loss,logits,voxel_state


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

        decay_steps=epoch_batch_num*FLAGS.recon_decay_epoch
        recon_loss_ratio=tf.train.exponential_decay(1.0,global_step,decay_steps,FLAGS.recon_decay_rate,staircase=True)
        tf.summary.scalar('reconstruction loss ratio',recon_loss_ratio)


        opt=tf.train.AdamOptimizer(lr)

        with tf.name_scope('split_data'):
            tower_points=tf.split(points, FLAGS.num_gpus)
            tower_covars=tf.split(covars,FLAGS.num_gpus)
            tower_labels=tf.split(labels,FLAGS.num_gpus)
            tower_true_state=tf.split(true_state,FLAGS.num_gpus)

        reuse=False
        tower_grads=[]
        tower_recon_grads=[]
        tower_losses,tower_recon_losses=[],[]
        tower_logits=[]
        tower_voxel_state=[]
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('tower_{}'.format(i)):
                    loss,recon_loss,logits,voxel_state\
                        =tower_loss(tower_points[i], tower_covars[i],tower_labels[i],
                                    tower_true_state[i], is_training, num_classes,voxel_num,reuse)

                    grad=opt.compute_gradients(loss+(recon_loss*recon_loss_ratio))
                    tower_grads.append(grad)

                    all_var=tf.trainable_variables()
                    recon_var=[var for var in all_var if var.name.startswith('point_mlp') or \
                                var.name.startswith('fc') or var.name.startswith('voxel_state')]
                    recon_grad=opt.compute_gradients(recon_loss*recon_loss_ratio,var_list=recon_var)
                    tower_recon_grads.append(recon_grad)

                    tower_losses.append(loss)
                    tower_recon_losses.append(recon_loss)
                    tower_logits.append(logits)
                    tower_voxel_state.append(voxel_state)
                    update_op=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                    reuse=True

        avg_grad=average_gradients(tower_grads)
        avg_recon_grad=average_gradients(tower_recon_grads)

        with tf.control_dependencies(update_op):
            apply_grad_op=opt.apply_gradients(avg_grad,global_step=global_step)

        apply_recon_grad_op=opt.apply_gradients(avg_recon_grad)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge(summaries)

        total_loss_op=tf.add_n(tower_losses)/FLAGS.num_gpus
        total_recon_loss_op=tf.add_n(tower_recon_losses)/FLAGS.num_gpus

        logits_op=tf.concat(tower_logits,axis=0)
        preds_op=tf.argmax(logits_op,axis=1)
        correct_num_op=tf.reduce_sum(tf.cast(tf.equal(preds_op,labels),tf.float32))

        voxel_state_op=tf.concat(tower_voxel_state,axis=0)

        ops['total_loss']=total_loss_op
        ops['total_recon_loss']=total_recon_loss_op
        ops['apply_grad']=apply_grad_op
        ops['apply_recon_grad']=apply_recon_grad_op
        ops['logits']=logits_op
        ops['preds']=preds_op
        ops['correct_num']=correct_num_op
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
        new_points=np.empty([points.shape[0]*4,points.shape[1],3],dtype=np.float32)
        new_points[:points.shape[0]]=points
        new_points[points.shape[0]:points.shape[0]*2]=rotate(points,np.pi/2.0)
        new_points[points.shape[0]*2:points.shape[0]*3]=rotate(points,np.pi)
        new_points[points.shape[0]*3:]=rotate(points,np.pi*3.0/2.0)

        labels=np.tile(labels,[4,1])
        nidxs=np.tile(nidxs,[4,1,1])
        points=new_points

    voxel_state =points2voxel_gpu_modelnet(points, FLAGS.split_num, 1)
    covars=points2covars_gpu(points, nidxs, 16, 0)

    return points, covars, labels, voxel_state


def batch_fn(file_data, cur_idx, data_indices, require_size):
    end_idx = min(cur_idx + require_size, file_data[0].shape[0])
    batch_data=[item[data_indices[cur_idx:end_idx]] for item in file_data]
    return batch_data , end_idx - cur_idx


def train_one_epoch(ops,pls,sess,summary_writer,trainset,testset,epoch_num,feed_dict):
    total_recon_losses=[]
    begin_time=time.time()
    total_model=0
    for i,feed_in in enumerate(testset):
        points_list, covars_list, labels_list, voxel_state_list=\
            unpack_feats_labels(feed_in,FLAGS.num_gpus)

        feed_dict[pls['points']]=points_list
        feed_dict[pls['covars']]=covars_list
        feed_dict[pls['labels']]=labels_list[:,0]
        feed_dict[pls['is_training']]=True
        feed_dict[pls['voxel_state']]=voxel_state_list

        _,recon_loss_val=sess.run([ops['apply_recon_grad'],ops['total_recon_loss']],feed_dict)
        total_recon_losses.append(recon_loss_val)
        total_model+=points_list.shape[0]

    log_str('epoch {} testset reconstruction loss {:.5} | {:.5} examples/s'.format(
        epoch_num,
        np.mean(np.asarray(total_recon_losses)),
        float(total_model) / (time.time() - begin_time)
    ), FLAGS.log_file)

    total_correct,total_model=0,0
    begin_time=time.time()
    total_recon_losses,total_losses=[],[]
    for i,feed_in in enumerate(trainset):
        points_list, covars_list, labels_list, voxel_state_list=\
            unpack_feats_labels(feed_in,FLAGS.num_gpus)

        feed_dict[pls['points']]=points_list
        feed_dict[pls['covars']]=covars_list
        feed_dict[pls['labels']]=labels_list[:,0]
        feed_dict[pls['is_training']]=True
        feed_dict[pls['voxel_state']]=voxel_state_list
        total_model+=points_list.shape[0]

        _,loss_val,recon_loss_val,correct_num=sess.run(
            [ops['apply_grad'],ops['total_loss'],ops['total_recon_loss'],ops['correct_num']],feed_dict)
        total_losses.append(loss_val)
        total_recon_losses.append(recon_loss_val)
        total_correct+=correct_num

        if i % FLAGS.log_step==0:
            summary,global_step=sess.run(
                [ops['summary'],ops['global_step']],feed_dict)

            log_str('epoch {} step {} loss {:.5} recon_loss {:.5} acc {:.5} | {:.5} examples/s'.format(
                epoch_num,i,np.mean(np.asarray(total_losses)),
                np.mean(np.asarray(total_recon_losses)),
                float(total_correct)/total_model,
                float(total_model)/(time.time()-begin_time)
            ),FLAGS.log_file)

            summary_writer.add_summary(summary,global_step)
            total_correct,total_model=0,0
            begin_time=time.time()
            total_recon_losses,total_losses=[],[]


def output_gen_points(points_list,voxel_state_list,gen_state, left_size,epoch_num):
    idx = np.random.randint(0, points_list.shape[0], dtype=np.int)

    pts = points_list[idx, :, :]
    pts[:, :2] += 0.5
    pts[:, 3:] += 1.0
    pts[:, 3:] *= 127
    fn = os.path.join(FLAGS.dump_dir, '{}_{}_points.txt'.format(epoch_num, left_size))
    output_points(fn, pts)

    true_state_pts = voxel2points(voxel_state_list[idx])
    fn = os.path.join(FLAGS.dump_dir, '{}_{}_state_true.txt'.format(epoch_num, left_size))
    output_points(fn, true_state_pts)

    gen_state[idx][gen_state[idx] < 0.0] = 0.0
    gen_state[idx][gen_state[idx] > 1.0] = 1.0
    pred_state_pts = voxel2points(gen_state[idx])
    fn = os.path.join(FLAGS.dump_dir, '{}_{}_state_pred.txt'.format(epoch_num, left_size))
    output_points(fn, pred_state_pts)


def test_one_epoch(ops,pls,sess,saver,testset,epoch_num,feed_dict):
    total=0
    begin_time=time.time()
    test_loss,test_recon_loss=[],[]
    all_preds,all_labels=[],[]
    left_size=FLAGS.dump_num
    for i,feed_in in enumerate(testset):
        points_list, covars_list, labels_list, voxel_state_list= unpack_feats_labels(feed_in,FLAGS.num_gpus)

        feed_dict[pls['points']]=points_list
        feed_dict[pls['covars']]=covars_list
        feed_dict[pls['labels']]=labels_list[:,0]
        feed_dict[pls['is_training']]=False
        feed_dict[pls['voxel_state']]=voxel_state_list

        total+=points_list.shape[0]

        loss,recon_loss,preds,gen_state=sess.run([ops['total_loss'],ops['total_recon_loss'],
                                                  ops['preds'],ops['voxel_state']],feed_dict)
        test_loss.append(loss)
        test_recon_loss.append(recon_loss)

        all_preds.append(preds.flatten())
        all_labels.append(labels_list.flatten())
        if left_size > 0 and random.random() < 0.3:
            output_gen_points(points_list,voxel_state_list,gen_state,left_size,epoch_num)
            left_size-=1

    all_preds=np.concatenate(all_preds,axis=0)
    all_labels=np.concatenate(all_labels,axis=0)

    test_loss=np.mean(np.asarray(test_loss))
    test_recon_loss=np.mean(np.asarray(test_recon_loss))

    acc,macc,oacc=compute_acc(all_labels,all_preds,FLAGS.num_classes)

    log_str('mean acc {:.5} overall acc {:.5} loss {:.5} recon_loss {:.5} cost {:.3} s'.format(
        macc, oacc, test_loss, test_recon_loss, time.time()-begin_time
    ),FLAGS.log_file)

    checkpoint_path = os.path.join(FLAGS.save_dir, 'unsupervise{}.ckpt'.format(epoch_num))
    saver.save(sess,checkpoint_path)


def train():
    pt_num=2048
    voxel_num=FLAGS.split_num*FLAGS.split_num*FLAGS.split_num

    # train_list,test_list=get_train_test_split()
    train_list=['data/ModelNetTrain/nidxs/ply_data_train{}.h5'.format(i) for i in xrange(5)]
    test_list=['data/ModelNetTrain/nidxs/ply_data_test{}.h5'.format(i) for i in xrange(2)]

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
                      8000/(FLAGS.batch_size*FLAGS.num_gpus))

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
            train_one_epoch(ops,pls,sess,summary_writer,train_provider,test_provider,epoch_num,feed_dict)
            test_one_epoch(ops,pls,sess,saver,test_provider,epoch_num,feed_dict)

    finally:
        train_provider.close()
        test_provider.close()


def test_data_iter():
    train_list=['data/ModelNetTrain/nidxs/ply_data_train{}.h5'.format(i) for i in xrange(5)]
    test_list=['data/ModelNetTrain/nidxs/ply_data_test{}.h5'.format(i) for i in xrange(2)]

    train_provider = ProviderV2(train_list,'test',FLAGS.batch_size*FLAGS.num_gpus,batch_fn,read_fn,2)
    # test_provider = ProviderV2(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,batch_fn,read_fn,2)

    begin=time.time()
    for data in train_provider:
        for item in data:
            print item.shape
        break
        # true_state_pts = voxel2points(data[3][0])
        # fn = os.path.join(FLAGS.dump_dir, 'state_true.txt')
        # output_points(fn, true_state_pts)
        # print np.min(data[0],axis=(0,1))
        # print 'cost {} s'.format(time.time()-begin)
        # break

    train_provider.close()


if __name__=="__main__":
    train()