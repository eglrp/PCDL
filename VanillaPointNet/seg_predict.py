from seg_network import SegmentationNetwork
from preprocess import H5ReaderAll
import h5py
import numpy as np
import tensorflow as tf

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

def output_classified_points(filename,points,labels,colors):
    with open(filename,'w') as f:
        for pt,l in zip(points,labels):
            f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],
                               colors[int(l),0],colors[int(l),1],colors[int(l),2]))


def output_iou(label,pred,names=None):
    fp = np.zeros(14, dtype=np.int)
    tp = np.zeros(14, dtype=np.int)
    fn = np.zeros(14, dtype=np.int)
    for l, p in zip(label, pred):
        if l == p:
            tp[l] += 1
        else:
            fp[p] += 1
            fn[l] += 1

    iou = tp / (fp + fn + tp + 1e-6).astype(np.float)

    print 'overall iou: {}'.format(np.sum(tp) / float(np.sum(tp + fn + tp)))
    print 'mean iou: {}'.format(np.mean(iou))
    print 'precision: {}'.format(np.sum(label == pred) / float(label.shape[0]))

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


import glob
import os
def test_area6_iou(model_path='model/1024_lrelu_seg/epoch35.ckpt'):
    with open('s3dis_util/class_names.txt','r') as f:
        names=[line.strip('\n') for line in f.readlines()]

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
        draw_points(os.path.basename(fn)[:-4],data,label,pred)
        output_iou(label,pred)

    all_label=np.concatenate(all_label,axis=0)
    all_pred=np.concatenate(all_pred,axis=0)
    output_iou(all_label,all_pred,names)


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
    test_area6_iou('model/epoch220.ckpt')

