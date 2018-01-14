import argparse
import functools
import time

import tensorflow as tf

from point_network import inference
from provider import Provider, ProviderMultiGPUWrapper
from s3dis.point_util import *

from train_util import log_str,average_gradients

# FLAGS = tf.app.flags.FLAGS

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=2, help='')
parser.add_argument('--batch_size', type=int, default=4096, help='')
parser.add_argument('--num_classes', type=int, default=13, help='')

parser.add_argument('--lr_init', type=float, default=1e-3, help='')
parser.add_argument('--decay_rate', type=float, default=0.9, help='')
parser.add_argument('--decay_epoch', type=int, default=10, help='')

parser.add_argument('--log_step', type=int, default=100, help='')
parser.add_argument('--train_dir', type=str, default='train', help='')
parser.add_argument('--save_dir', type=str, default='model', help='')
parser.add_argument('--log_file', type=str, default='train.log', help='')

parser.add_argument('--train_epoch_num', type=int, default=500, help='')
FLAGS = parser.parse_args()


def tower_loss(feats,labels,num_classes,is_trainging,reuse=False):
    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
        logits=inference(feats,is_trainging,num_classes,reuse)

    loss=tf.losses.sparse_softmax_cross_entropy(labels,logits)

    preds=tf.argmax(logits,axis=1)
    preds=tf.cast(preds,dtype=tf.int64)
    correct_mask=tf.equal(preds,tf.squeeze(labels))
    acc=tf.reduce_mean(tf.cast(correct_mask,dtype=tf.float32),name='accuracy')

    # loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    # loss_averages_op = loss_averages.apply([loss])

    tf.summary.scalar(loss.op.name,loss)
    # tf.summary.scalar(loss.op.name+'_avg',loss_averages.average(loss))
    tf.summary.scalar(acc.op.name,acc)

    # with tf.control_dependencies([loss_averages_op]):
    #     loss=tf.identity(loss)

    return loss,logits


def unpack_feats_labels(batch,num_gpus):
    labels,feats=[],[]
    for i in xrange(num_gpus):
        feats.append(batch[i][0])
        labels.append(batch[i][1])

    labels=np.concatenate(labels,axis=0)
    feats=np.concatenate(feats,axis=0)
    if feats.shape[0]%num_gpus!=0:
        left_num=(feats.shape[0]/num_gpus+1)*num_gpus-feats.shape[0]
        left_idx=np.random.randint(0,feats.shape[0],left_num)
        feats=np.concatenate([feats,feats[left_idx,:]],axis=0)
        labels=np.concatenate([labels,labels[left_idx]],axis=0)

    return feats,labels


def train_ops(feats, labels, is_training, epoch_batch_num):
    ops={}
    with tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        decay_steps=epoch_batch_num*FLAGS.decay_epoch
        lr=tf.train.exponential_decay(FLAGS.lr_init,global_step,decay_steps,FLAGS.decay_rate,True)
        tf.summary.scalar('learning rate',lr)

        opt=tf.train.AdamOptimizer(lr)

        tower_feats=tf.split(feats, FLAGS.num_gpus)
        tower_labels=tf.split(labels,FLAGS.num_gpus)

        reuse=False
        tower_grads=[]
        tower_losses=[]
        tower_logits=[]
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('tower_{}'.format(i)):
                    loss,logits=tower_loss(tower_feats[i], tower_labels[i], FLAGS.num_classes, is_training, reuse)
                    # print tf.trainable_variables()
                    grad=opt.compute_gradients(loss,tf.trainable_variables())
                    tower_grads.append(grad)
                    tower_losses.append(loss)
                    tower_logits.append(logits)

                    batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    # print batchnorm_updates
                    reuse=True

        # todo: the batchnorm updates will copy to another gpu?
        avg_grad=average_gradients(tower_grads)
        with tf.control_dependencies(batchnorm_updates):
            apply_grad_op=opt.apply_gradients(avg_grad,global_step=global_step)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge(summaries)

        preds=tf.argmax(tf.concat(tower_logits,axis=0),axis=1)
        correct_pred_op=tf.reduce_sum(tf.cast(tf.equal(preds,labels),tf.float32))
        total_loss_op=tf.add_n(tower_losses)

        ops['correct_num']=correct_pred_op
        ops['total_loss']=total_loss_op
        ops['apply_grad']=apply_grad_op
        ops['summary']=summary_op
        ops['global_step']=global_step
        ops['preds']=preds

    return ops


def train_one_epoch(ops,pls,sess,summary_writer,trainset,epoch_num):
    feed_dict={}
    correct,total=0,0
    begin_time=time.time()
    for i,feed_in in enumerate(trainset):
        feats,labels=unpack_feats_labels(feed_in,FLAGS.num_gpus)

        feed_dict[pls['feats']]=feats
        feed_dict[pls['labels']]=labels.flatten()
        feed_dict[pls['is_training']]=True

        _,correct_num=sess.run([ops['apply_grad'],ops['correct_num']],feed_dict)
        correct+=correct_num
        total+=labels.shape[0]

        if i % FLAGS.log_step==0:
            total_loss,summary,global_step=sess.run(
                [ops['total_loss'],ops['summary'],ops['global_step']],feed_dict)

            accuracy=float(correct)/total

            log_str('epoch {} step {} loss {:.5} accuracy {:.5} | {:.5} examples/s'.format(
                epoch_num,i,total_loss,accuracy,float(total)/(time.time()-begin_time)
            ),FLAGS.log_file)

            summary_writer.add_summary(summary,global_step)

            correct,total=0,0
            begin_time=time.time()


def test_one_epoch(ops, pls, sess, saver, testset, epoch_num):
    feed_dict={}
    begin_time=time.time()
    all_preds,all_labels=[],[]
    for i,feed_in in enumerate(testset):
        feats,labels=unpack_feats_labels(feed_in,FLAGS.num_gpus)

        feed_dict[pls['feats']]=feats
        feed_dict[pls['is_training']]=False

        preds=sess.run(ops['preds'],feed_dict)

        all_preds.append(preds)
        all_labels.append(labels)

    all_preds=np.concatenate(all_preds,axis=0)
    all_labels=np.concatenate(all_labels,axis=0)

    iou, miou, oiou, acc, macc, oacc=compute_iou(all_labels,all_preds)

    log_str('mean iou {:.5} overall iou {:5} \n mean acc {:5} overall acc {:5} cost {} s'.format(
        miou,oiou,macc,oacc, time.time()-begin_time
    ),FLAGS.log_file)

    checkpoint_path = os.path.join(FLAGS.save_dir, 'model{}_{:.3}.ckpt'.format(epoch_num,miou))
    saver.save(sess,checkpoint_path)


def train():
    train_list, test_list = prepare_input_list('data/S3DIS/point/fpfh/', FLAGS.batch_size)
    fetch_data_with_batch = functools.partial(fetch_data, batch_size=FLAGS.batch_size)
    train_provider = Provider(train_list, 1, fetch_data_with_batch, 'train', 4, fetch_batch, max_worker_num=1)
    test_provider = Provider(test_list, 1, fetch_data_with_batch, 'test', 4, fetch_batch, max_worker_num=1)
    trainset=ProviderMultiGPUWrapper(FLAGS.num_gpus,train_provider)
    testset=ProviderMultiGPUWrapper(FLAGS.num_gpus,test_provider)

    try:
        pls={}
        pls['feats']=tf.placeholder(tf.float32,[None,39],'feats')
        pls['labels']=tf.placeholder(tf.int64,[None,],'labels')
        pls['is_training']=tf.placeholder(tf.bool,[],'is_training')
        ops=train_ops(pls['feats'],pls['labels'],pls['is_training'],train_provider.batch_num)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=500)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,graph=sess.graph)

        for epoch_num in xrange(FLAGS.train_epoch_num):
            test_one_epoch(ops,pls,sess,saver,testset,epoch_num)
            train_one_epoch(ops,pls,sess,summary_writer,trainset,epoch_num)

    finally:
        train_provider.close()
        test_provider.close()


if __name__=="__main__":
    train()






