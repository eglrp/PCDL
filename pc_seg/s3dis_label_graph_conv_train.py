import argparse
import time
import numpy as np
import os

import tensorflow as tf

from graph_conv_layer import graph_conv_encoder
from classify_network import segmentation_classifier_v2
from s3dis.block_util import read_block_v2
from s3dis.data_util import get_block_train_test_split,compute_iou,get_class_names,read_room_pkl
from s3dis.sample_util import flip,swap_xy,uniform_sample_block,random_rotate_sample_block,downsample_random_gpu

from train_util import log_str,average_gradients
from provider import ProviderV3
import random
from tf_ops import libPointUtil

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=4, help='')
parser.add_argument('--batch_size', type=int, default=1, help='')

parser.add_argument('--lr_init', type=float, default=1e-3, help='')
parser.add_argument('--lr_clip', type=float, default=1e-5, help='')
parser.add_argument('--decay_rate', type=float, default=0.5, help='')
parser.add_argument('--decay_epoch', type=int, default=50, help='')
parser.add_argument('--num_classes', type=int, default=13, help='')

parser.add_argument('--log_step', type=int, default=250, help='')
parser.add_argument('--train_dir', type=str, default='train/s3dis_graph', help='')
parser.add_argument('--save_dir', type=str, default='model/s3dis_graph', help='')
parser.add_argument('--log_file', type=str, default='s3dis_graph.log', help='')


parser.add_argument('--eval',type=bool, default=False, help='')
parser.add_argument('--eval_model',type=str, default='model/label/unsupervise80.ckpt',help='')

parser.add_argument('--train_epoch_num', type=int, default=500, help='')

FLAGS = parser.parse_args()

BLOCK_SIZE=1.0
BLOCK_STRIDE=0.5
SAMPLE_STRIDE_LOW=0.02
SAMPLE_STRIDE_HIGH=0.05
RESAMPLE_RATIO_LOW=0.8
RESAMPLE_RATIO_HIGH=1.0
NEIGHBOR_RADIUS=0.1


def tower_loss(points,labels,nidxs,nidxs_lens,nidxs_bgs,m,is_training,num_classes,reuse=False,pmiu=None):
    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
        points_coord,point_colors=tf.split(points,2,axis=1) # 9->3,3,3
        global_feats,point_feats=graph_conv_encoder\
            (points_coord,point_colors,nidxs,nidxs_lens,nidxs_bgs,m,pmiu,reuse,final_dim=512)

        global_feats=tf.expand_dims(global_feats,axis=0)
        point_feats=tf.expand_dims(point_feats,axis=0)
        point_colors=tf.expand_dims(point_colors,axis=0)

        k=tf.shape(point_feats)[1]
        global_feats=tf.tile(tf.expand_dims(global_feats,axis=1),[1,k,1])
        point_all_feats=tf.concat([global_feats, point_feats, point_colors],axis=2)

        logits=segmentation_classifier_v2(point_all_feats, point_colors, is_training, num_classes, reuse,use_bn=False)

    flatten_logits=tf.reshape(logits,[-1,num_classes])
    labels_flatten=tf.reshape(labels,[-1,1])
    labels_flatten=tf.squeeze(labels_flatten,axis=1)
    loss=tf.losses.sparse_softmax_cross_entropy(labels_flatten,flatten_logits)
    tf.summary.scalar(loss.op.name,loss)

    return loss,logits


def train_ops(points, labels, nidxs, nidxs_lens, nidxs_bgs, m, is_training, num_classes, epoch_batch_num, pmiu=None):
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

        reuse=False
        tower_grads=[]
        tower_losses=[]
        tower_logits=[]
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('tower_{}'.format(i)):
                    # print points[i],labels[i]
                    loss,logits=tower_loss(points[i], labels[i], nidxs[i],
                                           nidxs_lens[i], nidxs_bgs[i],m,
                                           is_training, num_classes, reuse)

                    tower_grads.append(opt.compute_gradients(loss))
                    tower_losses.append(loss)
                    tower_logits.append(tf.squeeze(logits,axis=0))
                    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    reuse=True

        avg_grad=average_gradients(tower_grads)

        with tf.control_dependencies(update_op):
            apply_grad_op=tf.group(opt.apply_gradients(avg_grad,global_step=global_step))

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge(summaries)

        total_loss_op=tf.add_n(tower_losses)/FLAGS.num_gpus

        logits_op=tf.concat(tower_logits,axis=0)
        preds_op=tf.argmax(logits_op,axis=1)

        flatten_labels=[]
        for i in xrange(FLAGS.num_gpus):
            flatten_labels.append(tf.squeeze(labels[i],axis=0,name='squeeze_labels_{}'.format(i)))

        flatten_labels=tf.concat(flatten_labels,axis=0)
        correct_num_op=tf.reduce_sum(tf.cast(tf.equal(preds_op,flatten_labels),tf.float32))

        ops['total_loss']=total_loss_op
        ops['apply_grad']=apply_grad_op
        ops['logits']=logits_op
        ops['preds']=preds_op
        ops['correct_num']=correct_num_op
        ops['summary']=summary_op
        ops['global_step']=global_step

    return ops


def point2ndixs(pts,radius):
    leafsize=15 if pts.shape[0]>100 else 1
    nidxs=libPointUtil.findNeighborRadiusGPU(pts,radius*radius,leafsize)
    nidxs_lens=[len(nidx) for nidx in nidxs]
    nidxs_lens=np.asarray(nidxs_lens,dtype=np.int32)
    nidxs=np.concatenate(nidxs,axis=0)

    csum=0
    nidxs_bgs=np.empty_like(nidxs_lens,dtype=np.int32)
    for i,lval in enumerate(nidxs_lens):
        nidxs_bgs[i]=csum
        csum+=lval

    return nidxs,nidxs_lens,nidxs_bgs


def read_fn(model,filename):
    points,labels=read_room_pkl(filename) # [n,6],[n,1]
    block_points_list,block_labels_list=[],[]
    if model=='train':
        # begin = time.time()
        sample_stride=np.random.uniform(SAMPLE_STRIDE_LOW,SAMPLE_STRIDE_HIGH)
        points, labels = downsample_random_gpu(points, labels, sample_stride)
        # print 'down sample cost {} s'.format(time.time() - begin)

        # rescale
        # begin = time.time()
        x_scale=np.random.uniform(0.9,1.1)
        y_scale=np.random.uniform(0.9,1.1)
        z_scale=np.random.uniform(0.9,1.1)
        points[:,0]*=x_scale
        points[:,1]*=y_scale
        points[:,2]*=z_scale
        # print 'scale cost {} s'.format(time.time() - begin)

        # begin = time.time()
        if random.random()<0.5:
            points=swap_xy(points)
        if random.random()<0.5:
            points=flip(points,axis=0)
        if random.random()<0.5:
            points=flip(points,axis=1)
        # print 'flip cost {} s'.format(time.time() - begin)

        # begin=time.time()
        points-=np.min(points,axis=0,keepdims=True)
        oblock_points_list,oblock_labels_list = uniform_sample_block(
            points,labels,BLOCK_SIZE,BLOCK_STRIDE,normalized=True)
        # print 'uniform sample cost {} s'.format(time.time()-begin)

        # begin=time.time()
        rblock_points_list,rblock_labels_list = random_rotate_sample_block(
            points,labels,BLOCK_SIZE,BLOCK_STRIDE,normalized=True)
        # print 'random sample cost {} s'.format(time.time()-begin)

        block_points_list+=rblock_points_list
        block_labels_list+=rblock_labels_list

        block_points_list+=oblock_points_list
        block_labels_list+=oblock_labels_list

    else:
        points, labels = downsample_random_gpu(points, labels, SAMPLE_STRIDE_LOW)
        oblock_points_list,oblock_labels_list = uniform_sample_block(
            points,labels,BLOCK_SIZE,BLOCK_SIZE,normalized=True)

        block_points_list+=oblock_points_list
        block_labels_list+=oblock_labels_list

    # normalize
    # begin = time.time()
    # room_max=np.max(points[:,:3],axis=0,keepdims=True)
    block_nidxs_list,block_nidxs_lens_list,block_nidxs_bgs_list=[],[],[]
    for block_idx in xrange(len(block_points_list)):
        # random downsample
        pt_num=block_points_list[block_idx].shape[0]

        if model =='train':
            random_down_ratio=np.random.uniform(RESAMPLE_RATIO_LOW,RESAMPLE_RATIO_HIGH)
            idxs=np.random.choice(pt_num,int(pt_num*random_down_ratio))
            block_points_list[block_idx]=block_points_list[block_idx][idxs]
            block_labels_list[block_idx]=block_labels_list[block_idx][idxs]
            pt_num=idxs.shape[0]

        # room_points=block_points_list[block_idx][:,:3]/room_max # n,3
        # room_points-=0.5
        # room_points/=0.5
        pts=np.ascontiguousarray(block_points_list[block_idx][:,:3],np.float32)
        nidxs, nidxs_lens, nidxs_bgs=point2ndixs(pts,NEIGHBOR_RADIUS)

        block_nidxs_list.append(nidxs)
        block_nidxs_lens_list.append(nidxs_lens)
        block_nidxs_bgs_list.append(nidxs_bgs)

        block_points_list[block_idx][:,:2]-=(np.min(block_points_list[block_idx][:,:2],axis=0,keepdims=True)+1.5)
        block_points_list[block_idx][:,:2]/=1.5
        block_points_list[block_idx][:,2]/=np.max(block_points_list[block_idx][:,2],axis=0,keepdims=True)
        block_points_list[block_idx][:,2]-=0.5
        block_points_list[block_idx][:,2]/=0.5

        if model =='train':
            block_points_list[block_idx][:,3:]+=np.random.uniform(-2.5,2.5,[pt_num,3])

        block_points_list[block_idx][:,3:]-=128
        block_points_list[block_idx][:,3:]/=128

        # block_points_list[block_idx]=np.concatenate([block_points_list[block_idx],room_points],axis=1) # n,9

    # print 'normalize cost {} s'.format(time.time()-begin)

    return block_points_list,block_labels_list,block_nidxs_list,block_nidxs_lens_list,block_nidxs_bgs_list


def unpack_feats_labels(batch,num_gpus):
    data_num=len(batch[0])
    if data_num%num_gpus!=0:
        left_num=(data_num/num_gpus+1)*num_gpus-data_num
        left_idx = np.random.randint(0, data_num, left_num)
        for i in xrange(len(batch)):
            for idx in left_idx:
                batch[i].append(batch[i][idx])
    return batch


def train_one_epoch(ops,pls,sess,summary_writer,trainset,epoch_num,feed_dict):
    epoch_begin=time.time()
    total_correct,total_block,total_points=0,0,0
    begin_time=time.time()
    total_losses=[]
    for i,feed_in in enumerate(trainset):
        points_list, labels_list, nidxs_list, nidxs_lens_list, nidxs_bgs_list=\
            unpack_feats_labels(feed_in,FLAGS.num_gpus)

        for k in xrange(FLAGS.num_gpus):
            feed_dict[pls['points'][k]]=points_list[k]
            feed_dict[pls['labels'][k]]=labels_list[k].flatten()
            feed_dict[pls['nidxs'][k]]=nidxs_list[k]
            feed_dict[pls['nidxs_lens'][k]]=nidxs_lens_list[k]
            feed_dict[pls['nidxs_bgs'][k]]=nidxs_bgs_list[k]

            total_points+=labels_list[k].shape[0]

        feed_dict[pls['is_training']]=True
        total_block+=FLAGS.num_gpus

        _,loss_val,correct_num=sess.run([ops['apply_grad'],ops['total_loss'],ops['correct_num']],feed_dict)
        total_losses.append(loss_val)
        total_correct+=correct_num

        if i % FLAGS.log_step==0:
            summary,global_step=sess.run(
                [ops['summary'],ops['global_step']],feed_dict)

            log_str('epoch {} step {} loss {:.5} acc {:.5} | {:.5} examples/s'.format(
                epoch_num,i,np.mean(np.asarray(total_losses)),
                float(total_correct)/total_points,
                float(total_block)/(time.time()-begin_time)
            ),FLAGS.log_file)

            summary_writer.add_summary(summary,global_step)
            total_correct,total_block,total_points=0,0,0
            begin_time=time.time()
            total_losses=[]

    log_str('epoch {} cost {} s'.format(epoch_num, time.time()-epoch_begin), FLAGS.log_file)


def test_one_epoch(ops,pls,sess,saver,testset,epoch_num,feed_dict):
    begin_time=time.time()
    test_loss=[]
    all_preds,all_labels=[],[]
    for i,feed_in in enumerate(testset):
        points_list, labels_list=unpack_feats_labels(feed_in,FLAGS.num_gpus)

        for k in xrange(FLAGS.num_gpus):
            feed_dict[pls['points'][k]]=np.expand_dims(points_list[k],axis=0)
            feed_dict[pls['labels'][k]]=np.expand_dims(labels_list[k].flatten(),axis=0)
            all_labels.append(labels_list[k].flatten())

        feed_dict[pls['is_training']] = False

        loss,preds=sess.run([ops['total_loss'],ops['preds']],feed_dict)
        test_loss.append(loss)

        all_preds.append(preds)

    all_preds=np.concatenate(all_preds,axis=0)
    all_labels=np.concatenate(all_labels,axis=0)

    test_loss=np.mean(np.asarray(test_loss))

    iou, miou, oiou, acc, macc, oacc = compute_iou(all_labels,all_preds)

    if not FLAGS.eval:
        log_str('mean iou {:.5} overall iou {:5} loss {:5} \n mean acc {:5} overall acc {:5} cost {:3} s'.format(
            miou, oiou, test_loss, macc, oacc, time.time()-begin_time
        ),FLAGS.log_file)

        checkpoint_path = os.path.join(FLAGS.save_dir, 'unsupervise{}.ckpt'.format(epoch_num))
        saver.save(sess,checkpoint_path)
    else:
        print 'mean iou {:.5} overall iou {:5} loss {:5} \n mean acc {:5} overall acc {:5} cost {:3} s'.format(
               miou, oiou, test_loss, macc, oacc, time.time()-begin_time)
        names=get_class_names()
        for name,iou_val in zip(names,iou):
            print '{} : {}'.format(name,iou_val)


def neighbor_anchors():
    interval=2*np.pi/8
    pmiu=[]
    for va in np.arange(-np.pi/2,np.pi/2+interval,interval):
        for ha in np.arange(0,2*np.pi,interval):
            v=np.asarray([np.cos(va)*np.cos(ha),
                         np.cos(va)*np.sin(ha),
                         np.sin(va)],dtype=np.float32)
            pmiu.append(v)

    return np.asarray(pmiu)


def train():
    train_list,test_list=get_block_train_test_split()
    train_list=['data/S3DIS/room_block_10_10/'+fn for fn in train_list]
    test_list=['data/S3DIS/room_block_10_10/'+fn for fn in test_list]

    train_provider = ProviderV3(train_list,'train',FLAGS.batch_size*FLAGS.num_gpus,read_fn)
    test_provider = ProviderV3(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,read_fn)

    try:
        pls={}
        pls['points'],pls['labels'],pls['nidxs'],pls['nidxs_lens'],pls['nidxs_bgs']=[],[],[],[],[]
        for i in xrange(FLAGS.num_gpus):
            pls['points'].append(tf.placeholder(tf.float32,[None,6],'points{}'.format(i)))
            pls['labels'].append(tf.placeholder(tf.int64,[None],'labels{}'.format(i)))
            pls['nidxs'].append(tf.placeholder(tf.int32,[None],'nidxs{}'.format(i)))
            pls['nidxs_lens'].append(tf.placeholder(tf.int32,[None],'nidxs_lens{}'.format(i)))
            pls['nidxs_bgs'].append(tf.placeholder(tf.int32,[None],'nidxs_bgs{}'.format(i)))

        pmiu=neighbor_anchors()
        pls['is_training']=tf.placeholder(tf.bool,name='is_training')
        pls['pmiu']=tf.placeholder(tf.float32,name='pmiu')
        batch_num_per_epoch=3.0/ BLOCK_SIZE * 3.0 / BLOCK_SIZE * 2827 / (FLAGS.batch_size * FLAGS.num_gpus)
        ops=train_ops(pls['points'],pls['labels'],pls['nidxs'],pls['nidxs'],pls['nidxs'],pmiu.shape[0],pls['is_training'],
                      FLAGS.num_classes,batch_num_per_epoch,pls['pmiu'])

        feed_dict = {}
        feed_dict[pls['pmiu']]=pmiu
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
    from provider import ProviderV3
    from s3dis.draw_util import output_points
    from s3dis.data_util import get_class_colors
    train_list,test_list=get_block_train_test_split()
    import random
    random.shuffle(train_list)

    train_list=['data/S3DIS/room_block_10_10/'+fn for fn in train_list[:20]]
    test_list=['data/S3DIS/room_block_10_10/'+fn for fn in test_list]
    train_provider = ProviderV3(train_list,'train',4,read_fn)
    try:
        max_label=0
        begin=time.time()
        i=0
        colors=get_class_colors()
        count=0
        for data in train_provider:
            data2=unpack_feats_labels(data,4)
            # i+=1
            print data2[0][0].shape
            print data2[1][0].shape
            print data2[2][0].shape
            print data2[3][0].shape
            print data2[4][0].shape
            # print '/////////////'
            # count+=1
            # print data[0][0].shape,data[1][0].shape
            points=data[0][0]
            labels=data[1][0]
            # print np.min(points,axis=0)
            # print np.max(points,axis=0)
            # max_label=max(np.max(labels),max_label)
            nidxs=data2[2][0]
            nidxs_lens=data2[3][0]
            nidxs_bgs=data2[4][0]
            output_points('test_result/class{}.txt'.format(i),points,colors[labels[:,0],:])
            for npti in xrange(len(nidxs_lens)):
                output_points('test_result/nidxs{}.txt'.format(npti),points[nidxs[nidxs_bgs[npti]:nidxs_bgs[npti]+nidxs_lens[npti]]])
            # points[:,3:6]*=128
            # points[:,3:6]+=128
            # points[:,3:6][points[:,3:6]>255]=255
            # points[:,3:6][points[:,3:6]<0]=0
            #
            # output_points('test_result/color{}.txt'.format(i),points)
            # output_points('test_result/room{}.txt'.format(i),points[:,6:],points[:,3:6])
            if i>=1:
                break
            i+=1
            # pass

        print 'cost {} s'.format(time.time()-begin)

        print count
    finally:
        train_provider.close()


if __name__=="__main__":
    # if FLAGS.eval:  eval()
    # else: train()
    # test_data_iter()
    train()