import argparse
import time
import numpy as np
import os

import tensorflow as tf

from autoencoder_network import concat_pointnet_encoder,fc_voxel_decoder,voxel_color_loss,voxel_filling_loss
from classify_network import segmentation_classifier
from s3dis.block_util import read_block_v2
from s3dis.data_util import get_train_test_split,compute_iou,get_class_names
from s3dis.voxel_util import points2voxel_color_gpu,voxel2points
from s3dis.draw_util import output_points

from train_util import log_str,average_gradients
from provider import ProviderV2

import random

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=4, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')

parser.add_argument('--lr_init', type=float, default=1e-3, help='')
parser.add_argument('--lr_clip', type=float, default=1e-5, help='')
parser.add_argument('--decay_rate', type=float, default=0.9, help='')
parser.add_argument('--decay_epoch', type=int, default=10, help='')
parser.add_argument('--num_classes', type=int, default=13, help='')

parser.add_argument('--recon_decay_rate', type=float, default=0.5, help='')
parser.add_argument('--recon_decay_epoch', type=int, default=20, help='')

# saver.restore(sess,'model/folding_label/unsupervise5.ckpt')
parser.add_argument('--log_step', type=int, default=20, help='')
parser.add_argument('--train_dir', type=str, default='train/voxel_label_sync', help='')
parser.add_argument('--save_dir', type=str, default='model/voxel_label_sync', help='')
parser.add_argument('--log_file', type=str, default='voxel_label_sync.log', help='')
# parser.add_argument('--pretrain_model',type=str, default='model/voxel/unsupervise380.ckpt',help='')


parser.add_argument('--eval',type=bool, default=False,help='')
parser.add_argument('--eval_model',type=str, default='model/folding_label/unsupervise152.ckpt',help='')

parser.add_argument('--train_epoch_num', type=int, default=500, help='')

parser.add_argument('--dump_dir', type=str, default='unsupervise/voxel_label_sync', help='')
parser.add_argument('--dump_num', type=int, default=10, help='')
parser.add_argument('--split_num', type=int, default=30, help='')

FLAGS = parser.parse_args()


def tower_loss(points, covars, rpoints, labels, true_state, true_color, num_classes, voxel_num, reuse=False):
    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
        points_coord,point_colors=tf.split(points,2,axis=2)                                        # n,k,3
        points_covars=tf.concat([points,covars],axis=2)                                            # [n,k,16]
        global_feats,point_feats=concat_pointnet_encoder(points_covars, reuse)                     # [n,1024],[n,k,1024]

        # get reconstructed voxel
        voxel_state,voxel_color=fc_voxel_decoder(global_feats, voxel_num, True, reuse)

        # get logits
        k=tf.shape(point_feats)[1]                                                                 # k
        global_feats=tf.tile(tf.expand_dims(global_feats,axis=1),[1,k,1])                          # n,k,1024
        point_all_feats=tf.concat([global_feats,point_feats,covars,rpoints,point_colors],axis=2)   # n,k,1024*2+9+3+3
        logits=segmentation_classifier(point_all_feats, num_classes, reuse)                        # n,k,num_classes

    # softmax loss
    flatten_logits=tf.reshape(logits,[-1,num_classes])
    labels=tf.reshape(labels,[-1,1])
    labels=tf.squeeze(labels,axis=1)
    loss=tf.losses.sparse_softmax_cross_entropy(labels,flatten_logits)
    tf.add_to_collection(tf.GraphKeys.LOSSES,loss)
    tf.summary.scalar(loss.op.name,loss)

    # color_loss and state loss
    color_loss=voxel_color_loss(voxel_color, true_state, true_color)
    filling_loss=voxel_filling_loss(voxel_state, true_state)
    recon_loss=tf.add(color_loss,filling_loss,name='total_recon_loss')
    tf.add_to_collection(tf.GraphKeys.LOSSES,recon_loss)

    tf.summary.scalar(color_loss.op.name,color_loss)
    tf.summary.scalar(filling_loss.op.name,filling_loss)
    tf.summary.scalar(recon_loss.op.name,recon_loss)

    return loss,recon_loss,logits,voxel_state,voxel_color


def train_ops(points, covars, rpoints, labels, true_state, true_color, num_classes, voxel_num, epoch_batch_num):
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
            tower_rpoints=tf.split(rpoints,FLAGS.num_gpus)
            tower_labels=tf.split(labels,FLAGS.num_gpus)
            tower_true_state=tf.split(true_state,FLAGS.num_gpus)
            tower_true_color=tf.split(true_color,FLAGS.num_gpus)

        reuse=False
        tower_grads=[]
        tower_recon_grads=[]
        tower_losses,tower_recon_losses=[],[]
        tower_logits=[]
        tower_voxel_state=[]
        tower_voxel_color=[]
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('tower_{}'.format(i)):
                    loss,recon_loss,logits,voxel_state,voxel_color\
                        =tower_loss(tower_points[i], tower_covars[i],
                                    tower_rpoints[i],tower_labels[i],
                                    tower_true_state[i],tower_true_color[i],
                                    num_classes,voxel_num,reuse)

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
                    tower_voxel_color.append(voxel_color)

                    reuse=True

        avg_grad=average_gradients(tower_grads)
        avg_recon_grad=average_gradients(tower_recon_grads)
        update_op=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_op):
            apply_grad_op=opt.apply_gradients(avg_grad,global_step=global_step)

        apply_recon_grad_op=opt.apply_gradients(avg_recon_grad)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge(summaries)

        total_loss_op=tf.add_n(tower_losses)/FLAGS.num_gpus
        total_recon_loss_op=tf.add_n(tower_recon_losses)/FLAGS.num_gpus

        logits_op=tf.concat(tower_logits,axis=0)
        preds_op=tf.argmax(logits_op,axis=2)
        correct_num_op=tf.reduce_sum(tf.cast(tf.equal(preds_op,labels),tf.float32))

        voxel_color_op=tf.concat(tower_voxel_color,axis=0)
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
        ops['voxel_color']=voxel_color_op
        ops['voxel_state']=voxel_state_op

    return ops


def read_fn(model,filename):
    points, covars, rpoints, labels=read_block_v2(filename)
    voxel_state, voxel_color=points2voxel_color_gpu(points,FLAGS.split_num,1)

    return points, covars, rpoints, labels, voxel_state, voxel_color


def batch_fn(file_data, cur_idx, data_indices, require_size):
    points, covars, rpoints, labels, voxel_state, voxel_color = file_data
    end_idx = min(cur_idx + require_size, points.shape[0])

    return [points[data_indices[cur_idx:end_idx], :, :],
            covars[data_indices[cur_idx:end_idx], :, :],
            rpoints[data_indices[cur_idx:end_idx], :, :],
            labels[data_indices[cur_idx:end_idx], :],
            voxel_state[data_indices[cur_idx:end_idx], :],
            voxel_color[data_indices[cur_idx:end_idx], :, :]
            ], end_idx - cur_idx


def unpack_feats_labels(batch,num_gpus):
    points_list, covars_list, rpoints_list, \
    labels_list, voxel_state_list, voxel_color_list = batch
    if points_list.shape[0]%num_gpus!=0:
        left_num=(points_list.shape[0]/num_gpus+1)*num_gpus-points_list.shape[0]
        left_idx = np.random.randint(0, points_list.shape[0], left_num)
        points_list=np.concatenate([points_list,points_list[left_idx]],axis=0)
        covars_list=np.concatenate([covars_list,covars_list[left_idx]],axis=0)
        rpoints_list=np.concatenate([rpoints_list,rpoints_list[left_idx]],axis=0)
        labels_list=np.concatenate([labels_list,labels_list[left_idx]],axis=0)
        voxel_state_list=np.concatenate([voxel_state_list,voxel_state_list[left_idx]],axis=0)
        voxel_color_list=np.concatenate([voxel_color_list,voxel_color_list[left_idx]],axis=0)

    return points_list, covars_list, rpoints_list, labels_list, voxel_state_list, voxel_color_list


def train_one_epoch(ops,pls,sess,summary_writer,trainset,epoch_num,feed_dict):
    total_correct,total_block,total_points=0,0,0
    begin_time=time.time()
    total_recon_losses,total_losses=[],[]
    for i,feed_in in enumerate(trainset):
        points_list, covars_list, rpoints_list, labels_list, voxel_state_list, voxel_color_list=\
            unpack_feats_labels(feed_in,FLAGS.num_gpus)

        feed_dict[pls['points']]=points_list
        feed_dict[pls['covars']]=covars_list
        feed_dict[pls['rpoints']]=rpoints_list
        feed_dict[pls['labels']]=labels_list
        feed_dict[pls['voxel_state']]=voxel_state_list
        feed_dict[pls['voxel_color']]=voxel_color_list
        total_block+=points_list.shape[0]
        total_points+=points_list.shape[0]*points_list.shape[1]

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
                float(total_correct)/total_points,
                float(total_block)/(time.time()-begin_time)
            ),FLAGS.log_file)

            summary_writer.add_summary(summary,global_step)
            total_correct,total_block,total_points=0,0,0
            begin_time=time.time()
            total_recon_losses,total_losses=[],[]


def output_gen_points(points_list,voxel_state_list,voxel_color_list,gen_state,gen_color, left_size,epoch_num):
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

    true_color_pts = voxel2points(voxel_color_list[idx])
    fn = os.path.join(FLAGS.dump_dir, '{}_{}_color_true.txt'.format(epoch_num, left_size))
    output_points(fn, true_color_pts)

    gen_color[idx][gen_color[idx] < 0.0] = 0.0
    gen_color[idx][gen_color[idx] > 1.0] = 1.0
    pred_color_pts = voxel2points(gen_color[idx])
    pred_color_pts = pred_color_pts[pred_state_pts[:,3]>127,:]
    fn = os.path.join(FLAGS.dump_dir, '{}_{}_color_pred.txt'.format(epoch_num, left_size))
    output_points(fn, pred_color_pts)


def test_one_epoch(ops,pls,sess,saver,testset,epoch_num,feed_dict):
    total=0
    begin_time=time.time()
    test_loss,test_recon_loss=[],[]
    all_preds,all_labels=[],[]
    left_size=FLAGS.dump_num
    for i,feed_in in enumerate(testset):
        points_list, covars_list, rpoints_list, labels_list, voxel_state_list, voxel_color_list=\
            unpack_feats_labels(feed_in,FLAGS.num_gpus)

        feed_dict[pls['points']]=points_list
        feed_dict[pls['covars']]=covars_list
        feed_dict[pls['rpoints']]=rpoints_list
        feed_dict[pls['labels']]=labels_list
        feed_dict[pls['voxel_state']]=voxel_state_list
        feed_dict[pls['voxel_color']]=voxel_color_list

        total+=points_list.shape[0]

        _,loss,recon_loss,preds,gen_color,gen_state=sess.run([ops['apply_recon_grad'],ops['total_loss'],
                                                              ops['total_recon_loss'],ops['preds'],
                                                              ops['voxel_color'],ops['voxel_state']],feed_dict)
        test_loss.append(loss)
        test_recon_loss.append(recon_loss)

        all_preds.append(preds.flatten())
        all_labels.append(labels_list.flatten())
        if left_size > 0 and random.random() < 0.3:
            output_gen_points(points_list,voxel_state_list,voxel_color_list,gen_state,gen_color,left_size,epoch_num)
            left_size-=1

    all_preds=np.concatenate(all_preds,axis=0)
    all_labels=np.concatenate(all_labels,axis=0)

    test_loss=np.mean(np.asarray(test_loss))
    test_recon_loss=np.mean(np.asarray(test_recon_loss))

    iou, miou, oiou, acc, macc, oacc=compute_iou(all_labels,all_preds)

    if not FLAGS.eval:
        log_str('mean iou {:.5} overall iou {:.5} loss {:.5} recon_loss {:.5} \n mean acc {:.5} overall acc {:.5} cost {:.3} s'.format(
            miou, oiou, test_loss, test_recon_loss, macc, oacc, time.time()-begin_time
        ),FLAGS.log_file)

        checkpoint_path = os.path.join(FLAGS.save_dir, 'unsupervise{}.ckpt'.format(epoch_num))
        saver.save(sess,checkpoint_path)
    else:
        print 'mean iou {:.5} overall iou {:5} loss {:5} \n mean acc {:5} overall acc {:5} cost {:3} s'.format(
               miou, oiou, test_loss, macc, oacc, time.time()-begin_time)
        names=get_class_names()
        for name,iou_val in zip(names,iou):
            print '{} : {}'.format(name,iou_val)


def train():
    pt_num=4096
    voxel_num=FLAGS.split_num*FLAGS.split_num*FLAGS.split_num

    train_list,test_list=get_train_test_split()
    train_list=['data/S3DIS/folding/block_v2/'+fn+'.h5' for fn in train_list]
    test_list=['data/S3DIS/folding/block_v2/'+fn+'.h5' for fn in test_list]

    train_provider = ProviderV2(train_list,'train',FLAGS.batch_size*FLAGS.num_gpus,batch_fn,read_fn,2)
    test_provider = ProviderV2(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,batch_fn,read_fn,2)

    try:
        pls={}
        pls['points']=tf.placeholder(tf.float32,[None,pt_num,6],'points')
        pls['covars']=tf.placeholder(tf.float32,[None,pt_num,9],'covars')
        pls['rpoints']=tf.placeholder(tf.float32,[None,pt_num,3],'rpoints')
        pls['labels']=tf.placeholder(tf.int64,[None,pt_num],'labels')
        pls['voxel_state']=tf.placeholder(tf.float32,[None,voxel_num],'voxel_state')
        pls['voxel_color']=tf.placeholder(tf.float32,[None,voxel_num,3],'voxel_color')
        ops=train_ops(pls['points'],pls['covars'],pls['rpoints'],
                      pls['labels'],pls['voxel_state'],pls['voxel_color'],
                      FLAGS.num_classes,voxel_num,
                      26000/(FLAGS.batch_size*FLAGS.num_gpus))

        feed_dict = {}
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=500)
        # saver.restore(sess,'model/voxel_label_sync/unsupervise9.ckpt')
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,graph=sess.graph)

        for epoch_num in xrange(FLAGS.train_epoch_num):
            train_one_epoch(ops,pls,sess,summary_writer,train_provider,epoch_num,feed_dict)
            test_one_epoch(ops,pls,sess,saver,test_provider,epoch_num,feed_dict)

    finally:
        train_provider.close()
        test_provider.close()


def eval():
    pt_num=4096
    voxel_num=FLAGS.split_num*FLAGS.split_num*FLAGS.split_num

    train_list,test_list=get_train_test_split()
    test_list=['data/S3DIS/folding/block_v2/'+fn+'.h5' for fn in test_list]
    test_provider = ProviderV2(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,batch_fn,read_block_v2,2)

    try:
        pls={}
        pls['points']=tf.placeholder(tf.float32,[None,pt_num,6],'points')
        pls['covars']=tf.placeholder(tf.float32,[None,pt_num,9],'covars')
        pls['rpoints']=tf.placeholder(tf.float32,[None,pt_num,3],'rpoints')
        pls['labels']=tf.placeholder(tf.int64,[None,pt_num],'labels')
        pls['voxel_state']=tf.placeholder(tf.float32,[None,voxel_num],'voxel_state')
        pls['voxel_color']=tf.placeholder(tf.float32,[None,voxel_num,3],'voxel_color')
        ops=train_ops(pls['points'],pls['covars'],pls['rpoints'],
                      pls['labels'],pls['voxel_state'],pls['voxel_color'],
                      FLAGS.num_classes,voxel_num,
                      22000/(FLAGS.batch_size*FLAGS.num_gpus))

        feed_dict = {}
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        saver = tf.train.Saver(max_to_keep=500)
        saver.restore(sess,FLAGS.eval_model)

        test_one_epoch(ops,pls,sess,saver,test_provider,0,feed_dict)

    finally:
        test_provider.close()


def test_data_iter():
    train_list,test_list=get_train_test_split()
    train_list=['data/S3DIS/folding/block_v2/'+fn+'.h5' for fn in train_list]
    # test_list=['data/S3DIS/folding/block_v2/'+fn+'.h5' for fn in test_list]

    train_provider = ProviderV2(train_list,'train',FLAGS.batch_size*FLAGS.num_gpus,batch_fn,read_block_v2,2)
    max_label=0
    for data in train_provider:
        # print data[0].shape,data[1].shape
        max_label=max(np.max(data[3]),max_label)

    print max_label
    train_provider.close()

if __name__=="__main__":
    if FLAGS.eval:  eval()
    else: train()