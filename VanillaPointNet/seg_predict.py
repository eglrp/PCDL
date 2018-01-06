from seg_network import SegmentationNetwork,ContextSegmentationNetwork
from preprocess import H5ReaderAll
from s3dis_util import util
import h5py
import numpy as np
import tensorflow as tf
from s3dis_util.util import *
import random

cls_colors=np.asarray(
        [[0, 255, 0],
         [0, 0, 255],
         [93, 201, 235],
         [255, 255, 0],
         [255, 140, 0],
         [0, 0, 128],
         [255, 69, 0],
         [255, 127, 80],
         [255, 0, 0],
         [255, 250, 240],
         [255, 0, 255],
         [255, 255, 255],
         [105, 105, 105],
         [205, 92, 92]],dtype=np.int
    )

def room2block(data,label,stride=1.0,block_size=1.0):

    limit = np.amax(data, 0)[0:3]

    # Get the corner location for our sampling blocks
    xbeg_list = []
    ybeg_list = []
    num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
    num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
    for i in range(-1,num_block_x):
        for j in range(-1,num_block_y):
            xbeg_list.append(i * stride)
            ybeg_list.append(j * stride)

    block_data_list = []
    block_label_list = []
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        xcond = (data[:,0]<=xbeg+block_size) & (data[:,0]>=xbeg)
        ycond = (data[:,1]<=ybeg+block_size) & (data[:,1]>=ybeg)
        cond = xcond & ycond
        if np.sum(cond)==0:
            continue
        block_data_list.append(data[cond,:])
        block_label_list.append(label[cond])

    return block_data_list,block_label_list

def pointnet_normalize(data_batch,data,block_size=1.0):
    max_room_x = max(data[:,0])
    max_room_y = max(data[:,1])
    max_room_z = max(data[:,2])
    new_data_batch=[]
    for b in range(len(data_batch)):
        new_single_batch=np.empty([data_batch[b].shape[0],9])
        new_single_batch[:, 6] = data_batch[b][ :, 0]/max_room_x
        new_single_batch[:, 7] = data_batch[b][ :, 1]/max_room_y
        new_single_batch[:, 8] = data_batch[b][ :, 2]/max_room_z
        minx = min(data_batch[b][ :, 0])
        miny = min(data_batch[b][ :, 1])
        new_single_batch[:, 0:6] = np.copy(data_batch[b])
        new_single_batch[:, 0] -= (minx+block_size/2)
        new_single_batch[:, 1] -= (miny+block_size/2)
        new_single_batch[:, 3:6]/=255.0
        new_data_batch.append(new_single_batch)

    return new_data_batch

def declare_network(model_path):
    input_dims=9
    num_classes=13

    net=SegmentationNetwork(input_dims,num_classes,True,1024)
    input=tf.placeholder(tf.float32, [1, None, input_dims],name='inputs')
    is_training=tf.placeholder(tf.bool,name='is_training')
    net.inference(input,'cpu',is_training)
    tf.get_variable_scope().reuse_variables()

    score_layer=net.ops['cpu_mlp8']

    config=tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess=tf.Session(config=config)
    saver=tf.train.Saver()
    saver.restore(sess,model_path)

    return sess,score_layer,input,is_training


def declare_context_network(model_path):
    input_dim=6
    num_classes=13
    local_feat_dim=33

    inputs={}
    net=ContextSegmentationNetwork(input_dim,num_classes,False,local_feat_dim,512)
    inputs['global_pts']=(tf.placeholder(tf.float32, [None, input_dim]))
    inputs['global_indices']=(tf.placeholder(tf.int64,[None,None]))
    inputs['context_pts']=(tf.placeholder(tf.float32, [None, input_dim]))
    inputs['context_batch_indices']=(tf.placeholder(tf.int64,[None]))
    inputs['context_block_indices']=(tf.placeholder(tf.int64,[None,None]))
    inputs['local_feats']=(tf.placeholder(tf.float32,[None,None,local_feat_dim]))
    inputs['is_training']=tf.placeholder(tf.bool,name='is_training')


    net.inference(inputs['global_pts'],inputs['global_indices'],
                  inputs['local_feats'],inputs['context_pts'],
                  inputs['context_batch_indices'],
                  inputs['context_block_indices'],
                  'cpu_',inputs['is_training'])

    score_layer=net.ops['inference_logits']

    config=tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess=tf.Session(config=config)
    saver=tf.train.Saver()
    saver.restore(sess,model_path)

    return sess,score_layer,inputs

def test_pointnet(room_h5_file,sess,score_layer,input_pl,is_training_pl):
    f=h5py.File(room_h5_file)
    data=f['data']
    label=f['label'][:,0]
    data-=np.min(data,axis=0,keepdims=True)
    data_batch,label_batch=room2block(data,label)
    normalized_data_batch=pointnet_normalize(data_batch,data)

    print 'preprocess done block num {}'.format(len(normalized_data_batch))

    preds=[]
    idx=0
    for data,label in zip(normalized_data_batch,label_batch):
        scores=sess.run(score_layer,feed_dict={input_pl:np.expand_dims(data,axis=0),is_training_pl:False})
        # print scores.shape
        pred=np.argmax(np.reshape(scores,[scores.shape[1],scores.shape[3]]),axis=1)
        # print pred.shape
        preds.append(pred)
        print '{} done'.format(idx)
        idx+=1

    print 'predict done'
    return np.concatenate(data_batch,axis=0),np.concatenate(label_batch,axis=0),np.concatenate(preds,axis=0)

def test_context_pointnet(filename,sess,score_layer,inputs):
    block_list,room_sample_data=util.read_pkl(filename)

    feed_dict={}
    all_preds=[]
    all_labels=[]
    all_data=[]
    for block in block_list:
        feed_dict[inputs['global_pts']]=room_sample_data
        feed_dict[inputs['context_pts']]=block['cont']
        feed_dict[inputs['context_batch_indices']]=np.asarray([0],dtype=np.int64)
        feed_dict[inputs['context_block_indices']]=np.expand_dims(block['cont_index'],axis=0)
        feed_dict[inputs['global_indices']]=np.expand_dims(block['room_index'],axis=0)
        feed_dict[inputs['local_feats']]=np.expand_dims(block['feat'],axis=0)
        feed_dict[inputs['is_training']]=False

        scores=sess.run(score_layer,feed_dict)
        pred=np.argmax(scores,axis=2)
        all_preds.append(np.squeeze(pred))
        all_labels.append(np.squeeze(block['label']))
        all_data.append(np.squeeze(block['data']))

    all_preds=np.concatenate(all_preds,axis=0)
    all_labels=np.concatenate(all_labels,axis=0)
    all_data=np.concatenate(all_data,axis=0)

    return all_data,all_labels,all_preds

def output_classified_points(filename,points,labels,colors):
    with open(filename,'w') as f:
        for pt,l in zip(points,labels):
            f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],
                               colors[int(l),0],colors[int(l),1],colors[int(l),2]))

def compute_iou(label,pred):
    fp = np.zeros(13, dtype=np.int)
    tp = np.zeros(13, dtype=np.int)
    fn = np.zeros(13, dtype=np.int)
    for l, p in zip(label, pred):
        if l == p:
            tp[l] += 1
        else:
            fp[p] += 1
            fn[l] += 1

    iou = tp / (fp + fn + tp + 1e-6).astype(np.float)
    miou=np.mean(iou)
    oiou=np.sum(tp) / float(np.sum(tp + fn + fp))
    acc = tp / (tp + fn + 1e-6)
    macc = np.mean(acc)
    oacc = np.sum(tp) / float(np.sum(tp+fn))

    return iou, miou, oiou, acc, macc, oacc

def output_iou(label,pred,names=None):
    iou, miou, oiou, acc, macc, oacc = compute_iou(label,pred)

    print 'overall iou: {}'.format(oiou)
    print 'mean iou: {}'.format(miou)
    print 'overall precision: {}'.format(oacc)
    print 'mean precision: {}'.format(macc)

    if names is not None:
        for val,name in zip(iou,names):
            print 'iou {} : {}'.format(name,val)

def draw_points(file_name,data,label,pred):
    colors=np.asarray(
        [[0, 255, 0],
         [0, 0, 255],
         [93, 201, 235],
         [255, 255, 0],
         [255, 140, 0],
         [0, 0, 128],
         [255, 69, 0],
         [255, 127, 80],
         [255, 0, 0],
         [255, 250, 240],
         [255, 0, 255],
         [255, 255, 255],
         [105, 105, 105],
         [205, 92, 92]],dtype=np.int
    )
    output_classified_points(file_name+'pred.txt',data,pred,colors)
    output_classified_points(file_name+'true.txt',data,label,colors)


def test_room():
    sess,score_layer,input,is_training=declare_network('model/1024_lrelu_seg/epoch35.ckpt')
    data,label,pred=test_pointnet('data/S3DIS/room/260_Area_6_office_35.h5',sess,score_layer,input,is_training)
    draw_points('a6035',data,label,pred)
    output_iou(label,pred)


def test_context_room():
    sess,score_layer,inputs=declare_context_network('model/context_overfit/epoch33_0.402.ckpt')

    data,label,pred=test_context_pointnet('data/S3DIS/tmp/177_Area_5_office_12.pkl',sess,score_layer,inputs)
    draw_points('a5O12',data,label,pred)
    output_iou(label,pred)


import glob
import os
def test_area6_iou(model_path='model/epoch33_0.402.ckpt'):
    a6_room_files=[]
    for fn in glob.glob(os.path.join('data/S3DIS/room','*.h5')):
        if os.path.basename(fn).split('_')[2]=='6':
            a6_room_files.append(fn)

    sess,score_layer,input,is_training=declare_network(model_path)

    all_label,all_pred=[],[]
    for fn in a6_room_files:
        data,label,pred=test_pointnet(fn,sess,score_layer,input,is_training)
        all_label.append(label)
        all_pred.append(pred)
        # draw_points(os.path.basename(fn)[:-4],data,label,pred)
        output_iou(label,pred)

    all_label=np.concatenate(all_label,axis=0)
    all_pred=np.concatenate(all_pred,axis=0)
    output_iou(all_label, all_pred, get_class_names())


def test_area5_iou_context(model_path='model/epoch280.ckpt',test_num=None,draw_result=False):
    train_stems,test_stems=get_train_test_split()
    file_lists=['data/S3DIS/train_v2_nostairs/'+fs+'.pkl' for fs in train_stems]

    random.shuffle(file_lists)
    if test_num is not None:
        file_lists=file_lists[:test_num]

    sess, score_layer, inputs = declare_context_network(model_path)

    all_label,all_pred=[],[]
    for fn in file_lists[:10]:
        data,label,pred=test_context_pointnet(fn,sess,score_layer,inputs)
        all_label.append(label)
        all_pred.append(pred)
        if draw_result:
            draw_points(os.path.basename(fn)[:-4],data,label,pred)
        output_iou(label,pred)

    all_label=np.concatenate(all_label,axis=0)
    all_pred=np.concatenate(all_pred,axis=0)
    output_iou(all_label,all_pred,get_class_names())

def plot_color():
    colors=np.asarray(
        [[0, 255, 0],
         [0, 0, 255],
         [93, 201, 235],
         [255, 255, 0],
         [255, 140, 0],
         [0, 0, 128],
         [255, 69, 0],
         [255, 127, 80],
         [255, 0, 0],
         [255, 250, 240],
         [255, 0, 255],
         [255, 255, 255],
         [105, 105, 105],
         [205, 92, 92]],dtype=np.int
    )
    with open('s3dis_util/class_names.txt','r') as f:
        names=[line.strip('\n') for line in f.readlines()]
    import matplotlib.pyplot as plt
    import matplotlib.colors as c
    import matplotlib.patches as mpatches
    patches=[]
    for i in range(13):
        patches.append(mpatches.Patch(label=names[i],color=c.to_rgb(tuple(colors[i].astype(np.float)/255.0))))
    plt.legend(handles=patches)
    plt.show()


if __name__=="__main__":
    test_area5_iou_context('model/epoch33_0.402.ckpt',test_num=10,draw_result=True)

