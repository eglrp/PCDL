import argparse
import time
import numpy as np
import os

import tensorflow as tf

from autoencoder_network import all_concat_pointnet_encoder
from classify_network import model_classifier_v2
from modelnet.data_util import read_modelnet_v2,compute_acc,rotate,exchange_dims_zy,get_classes_name
from s3dis.voxel_util import points2covars_gpu

from train_util import log_str,average_gradients
from provider import ProviderV2

import random

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=2, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')

parser.add_argument('--lr_init', type=float, default=1e-3, help='')
parser.add_argument('--lr_clip', type=float, default=1e-5, help='')
parser.add_argument('--decay_rate', type=float, default=0.5, help='')
parser.add_argument('--decay_epoch', type=int, default=25, help='')
parser.add_argument('--num_classes', type=int, default=40, help='')

parser.add_argument('--log_step', type=int, default=50, help='')
parser.add_argument('--train_dir', type=str, default='train/modelnet_label', help='')
parser.add_argument('--save_dir', type=str, default='model/modelnet_label', help='')
parser.add_argument('--log_file', type=str, default='modelnet_label.log', help='')


parser.add_argument('--eval', type=bool, default=True, help='')
parser.add_argument('--eval_model', type=str, default='model/modelnet_label/unsupervise102.ckpt', help='')

parser.add_argument('--confusion_matrix', type=bool, default=True, help='')
parser.add_argument('--confusion_matrix_path', type=str, default='test_result/', help='')
parser.add_argument('--output_error_models', type=bool, default=True, help='')
parser.add_argument('--output_error_path', type=str, default='test_result/', help='')

parser.add_argument('--train_epoch_num', type=int, default=500, help='')

FLAGS = parser.parse_args()


def tower_loss(points, covars, labels, is_training, num_classes, reuse=False):
    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
        points_covars=tf.concat([points,covars],axis=2)                                              # [n,k,16]
        global_feats,_=all_concat_pointnet_encoder(points_covars, is_training, reuse, final_dim=1024) # [n,1024]
        # get reconstructed voxel
        logits=model_classifier_v2(global_feats,num_classes,is_training,reuse)

    # softmax loss
    labels=tf.reshape(labels,[-1,1])
    labels=tf.squeeze(labels,axis=1)
    loss=tf.losses.sparse_softmax_cross_entropy(labels,logits)
    tf.add_to_collection(tf.GraphKeys.LOSSES,loss)
    tf.summary.scalar(loss.op.name,loss)

    return loss,logits


def train_ops(points, covars, labels, is_training, num_classes, epoch_batch_num):
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

        reuse=False
        tower_grads=[]
        tower_losses=[]
        tower_logits=[]
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('tower_{}'.format(i)):
                    loss,logits\
                        =tower_loss(tower_points[i], tower_covars[i],
                                    tower_labels[i],is_training, num_classes,reuse)

                    grad=opt.compute_gradients(loss)
                    tower_grads.append(grad)

                    tower_losses.append(loss)
                    tower_logits.append(logits)
                    tower_update_op=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                    reuse=True

        avg_grad=average_gradients(tower_grads)
        # update_op=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # print tower_update_op

        with tf.control_dependencies(tower_update_op):
            apply_grad_op=opt.apply_gradients(avg_grad,global_step=global_step)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge(summaries)

        total_loss_op=tf.add_n(tower_losses)/FLAGS.num_gpus

        logits_op=tf.concat(tower_logits,axis=0)
        preds_op=tf.argmax(logits_op,axis=1)
        correct_num_op=tf.reduce_sum(tf.cast(tf.equal(preds_op,labels),tf.float32))

        ops['total_loss']=total_loss_op
        ops['apply_grad']=apply_grad_op
        ops['logits']=logits_op
        ops['preds']=preds_op
        ops['correct_num']=correct_num_op
        ops['summary']=summary_op
        ops['global_step']=global_step

    return ops


def read_fn(model,filename):
    points, nidxs, labels=read_modelnet_v2(filename)
    points=exchange_dims_zy(points)

    if model=='train':
        points=rotate(points)
        # points=jitter_point_cloud(points)
    else:
        # new_points=np.empty([points.shape[0]*4,points.shape[1],3])
        # new_points[:points.shape[0]]=points
        # new_points[points.shape[0]:points.shape[0]*2]=rotate(points,np.pi/2.0)
        # new_points[points.shape[0]*2:points.shape[0]*3]=rotate(points,np.pi)
        # new_points[points.shape[0]*3:]=rotate(points,np.pi*3.0/2.0)
        # labels=np.tile(labels,[4,1])
        # nidxs=np.tile(nidxs,[4,1,1])
        # points=new_points
        pass

    covars=points2covars_gpu(points, nidxs, 16, 0)

    return points, covars, labels


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


def train_one_epoch(ops,pls,sess,summary_writer,trainset,epoch_num,feed_dict):
    total_correct,total_model=0,0
    begin_time=time.time()
    total_losses=[]
    for i,feed_in in enumerate(trainset):
        points_list, covars_list, labels_list=\
            unpack_feats_labels(feed_in,FLAGS.num_gpus)

        feed_dict[pls['points']]=points_list
        feed_dict[pls['covars']]=covars_list
        feed_dict[pls['labels']]=labels_list[:,0]
        feed_dict[pls['is_training']]=True

        total_model+=points_list.shape[0]

        _,loss_val,correct_num=sess.run(
            [ops['apply_grad'],ops['total_loss'],ops['correct_num']],feed_dict)
        total_losses.append(loss_val)
        total_correct+=correct_num

        if i % FLAGS.log_step==0:
            summary,global_step=sess.run(
                [ops['summary'],ops['global_step']],feed_dict)

            log_str('epoch {} step {} loss {:.5} acc {:.5} | {:.5} examples/s'.format(
                epoch_num,i,np.mean(np.asarray(total_losses)),
                float(total_correct)/total_model,
                float(total_model)/(time.time()-begin_time)
            ),FLAGS.log_file)

            summary_writer.add_summary(summary,global_step)
            total_correct,total_model=0,0
            begin_time=time.time()
            total_recon_losses,total_losses=[],[]


def test_one_epoch(ops,pls,sess,saver,testset,epoch_num,feed_dict):
    total=0
    begin_time=time.time()
    test_loss=[]
    all_preds,all_labels=[],[]
    all_error_models,all_error_preds,all_error_gts=[],[],[]
    for i,feed_in in enumerate(testset):
        points_list, covars_list, labels_list= unpack_feats_labels(feed_in,FLAGS.num_gpus)

        feed_dict[pls['points']]=points_list
        feed_dict[pls['covars']]=covars_list
        feed_dict[pls['labels']]=labels_list[:,0]
        feed_dict[pls['is_training']]=False

        total+=points_list.shape[0]

        loss,preds=sess.run([ops['total_loss'],ops['preds']],feed_dict)
        test_loss.append(loss)

        preds=preds.flatten()
        labels_list=labels_list.flatten()
        all_preds.append(preds)
        all_labels.append(labels_list)

        mask=preds!=labels_list
        all_error_models.append(points_list[mask])  # n,k,3
        all_error_preds.append(preds[mask])         # n
        all_error_gts.append(labels_list[mask])     # n

    all_preds=np.concatenate(all_preds,axis=0)
    all_labels=np.concatenate(all_labels,axis=0)

    test_loss=np.mean(np.asarray(test_loss))

    acc,macc,oacc=compute_acc(all_labels,all_preds,FLAGS.num_classes)

    if not FLAGS.eval:
        log_str('mean acc {:.5} overall acc {:.5} loss {:.5} cost {:.3} s'.format(
            macc, oacc, test_loss, time.time()-begin_time
        ),FLAGS.log_file)
        checkpoint_path = os.path.join(FLAGS.save_dir, 'unsupervise{}.ckpt'.format(epoch_num))
        saver.save(sess,checkpoint_path)
    else:

        print 'mean acc {:.5} overall acc {:.5} loss {:.5} cost {:.3} s'.format(
            macc, oacc, test_loss, time.time()-begin_time
        )
        names=get_classes_name()
        for name,accuracy in zip(names,acc):
            print '{} : {}'.format(name,accuracy)

        if FLAGS.confusion_matrix:
            from s3dis.draw_util import plot_confusion_matrix
            plot_confusion_matrix(all_preds,all_labels,names,save_path=FLAGS.confusion_matrix_path)

        if FLAGS.output_error_models:
            from s3dis.draw_util import output_points
            all_error_models=np.concatenate(all_error_models,axis=0)
            all_error_preds=np.concatenate(all_error_preds,axis=0)
            all_error_gts=np.concatenate(all_error_gts,axis=0)
            error_num=all_error_gts.shape[0]
            assert np.sum(all_labels!=all_preds)==error_num
            for k in xrange(error_num):
                output_points(FLAGS.output_error_path+'{}_{}_{}.txt'.format(
                    names[all_error_gts[k]],names[all_error_preds[k]],k),all_error_models[k])


def train():
    pt_num=2048

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

        ops=train_ops(pls['points'],pls['covars'],
                      pls['labels'],pls['is_training'],FLAGS.num_classes,
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
            train_one_epoch(ops,pls,sess,summary_writer,train_provider,epoch_num,feed_dict)
            test_one_epoch(ops,pls,sess,saver,test_provider,epoch_num,feed_dict)

    finally:
        train_provider.close()
        test_provider.close()


def eval():
    pt_num = 2048

    # train_list,test_list=get_train_test_split()
    # train_list = ['data/ModelNetTrain/nidxs/ply_data_train{}.h5'.format(i) for i in xrange(5)]
    test_list = ['data/ModelNetTrain/nidxs/ply_data_test{}.h5'.format(i) for i in xrange(2)]

    # train_provider = ProviderV2(train_list, 'train', FLAGS.batch_size * FLAGS.num_gpus, batch_fn, read_fn, 2)
    test_provider = ProviderV2(test_list, 'test', FLAGS.batch_size * FLAGS.num_gpus, read_fn)

    try:
        pls = {}
        pls['points'] = tf.placeholder(tf.float32, [None, pt_num, 3], 'points')
        pls['covars'] = tf.placeholder(tf.float32, [None, pt_num, 9], 'covars')
        pls['labels'] = tf.placeholder(tf.int64, [None], 'labels')
        pls['is_training'] = tf.placeholder(tf.bool, name='is_training')

        ops = train_ops(pls['points'], pls['covars'],
                        pls['labels'], pls['is_training'], FLAGS.num_classes,
                        8000 / (FLAGS.batch_size * FLAGS.num_gpus))

        feed_dict = {}
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=500)
        saver.restore(sess,FLAGS.eval_model)

        test_one_epoch(ops, pls, sess, saver, test_provider, 0, feed_dict)

    finally:
        # train_provider.close()
        test_provider.close()


if __name__=="__main__":
    if FLAGS.eval: eval()
    else: train()