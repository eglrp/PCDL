import tensorflow as tf
from cls_network import Network,leaky_relu
from preprocess import ModelBatchReader,add_noise,normalize,rotate,exchange_dims_zy
import PointSample
import numpy as np
import matplotlib.pyplot as plt

import itertools

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def assign_points_to_keypoints(pts,kpts):
    '''
    :param pts: [t,3]
    :param kpts: [k,3]
    :return:
    '''
    indices=np.empty(pts.shape[0])
    for i,pt in enumerate(pts):
        diff=kpts-pt[None,:]
        diff[:,0]**=2
        diff[:,1]**=2
        diff[:,2]**=2
        dist=np.sum(diff,axis=1)
        indices[i]=np.argmin(dist)

    return indices


def output_keypoints_patch():
    model_path='model/epoch46.ckpt'
    net=Network(3,40,True,1024)
    input=tf.placeholder(dtype=tf.float32,shape=[None,None,3,1],name='point_cloud')
    is_training=tf.placeholder(dtype=tf.bool,shape=[],name='is_training')

    net.inference(input,'cpu',is_training)
    feature_layer=net.ops['cpu_mlp5']

    config=tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess=tf.Session(config=config)
    saver=tf.train.Saver()
    saver.restore(sess,model_path)

    batch_files=['data/ModelNet40/train0.batch',
                 'data/ModelNet40/train1.batch',
                 'data/ModelNet40/train2.batch',
                 'data/ModelNet40/train3.batch']
    batch_size=5
    reader=ModelBatchReader(batch_files,batch_size,4,2048,3,'test',PointSample.getPointCloud,normalize)
    data,label=reader.next()
    features=sess.run(feature_layer,feed_dict={input:data,is_training:False})
    print features.shape
    features=np.reshape(features,[features.shape[0],features.shape[1],features.shape[3]])
    max_indices=np.argmax(features,axis=1)                  # [n,k]

    data=np.reshape(data,[data.shape[0],data.shape[1],data.shape[2]])

    kpts=[]
    for i in xrange(batch_size):
        kpt=[]
        added=[]
        for k in max_indices[i]:
            if k not in added:
                added.append(k)
                kpt.append(data[i,k,:])
        kpts.append(np.asarray(kpt))

    belong_indices=np.empty([data.shape[0],data.shape[1]],dtype=np.int)        # [n,t]
    for i in xrange(batch_size):
        belong_indices[i,:]=assign_points_to_keypoints(data[i],kpts[i])

    colors=np.asarray(np.random.uniform(0.0,1.0,[1024,3],)*255,dtype=int)

    # write the key points
    for i in xrange(batch_size):
        with open('{}_original.txt'.format(i),'w') as f:
            for pt in data[i]:
                f.write('{} {} {}\n'.format(pt[0],pt[1],pt[2]))

        with open('{}_keypoint.txt'.format(i),'w') as f:
            for j,pt in enumerate(kpts[i]):
                f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],colors[j,0],colors[j,1],colors[j,2]))

        with open('{}_assigned.txt'.format(i),'w') as f:
            for j in xrange(data.shape[1]):
                f.write('{} {} {} {} {} {}\n'.format(data[i,j,0],data[i,j,1],data[i,j,2],
                                                     colors[belong_indices[i,j],0],
                                                     colors[belong_indices[i,j],1],
                                                     colors[belong_indices[i,j],2]))


def maximize_activation():
    model_path='model/epoch404.ckpt'
    net=Network(3,40,True,1024)
    input=tf.placeholder(dtype=tf.float32,shape=[1,None,3,1],name='point_cloud')
    is_training=tf.placeholder(dtype=tf.bool,shape=[],name='is_training')

    net.inference(input,'cpu',is_training)
    # feature_layer=net.ops['cpu_pool']
    mlp_layer=net.ops['cpu_mlp5']

    feature_grad=tf.placeholder(dtype=tf.float32,shape=[1,None,1,1024],name='feature_grad')
    pts_grad=tf.gradients(mlp_layer,input,grad_ys=feature_grad)

    config=tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess=tf.Session(config=config)
    saver=tf.train.Saver()
    saver.restore(sess,model_path)

    lr=1e-2
    # dim=0
    iter_num=50
    pt_num=4096

    for dim in xrange(1024):
        pts=np.random.uniform(-1,1,[1,pt_num,3,1])
        feature_active_grad=np.zeros([1,pt_num,1,1024],np.float32)
        feature_active_grad[0,:,:,dim]=1.0

        for _ in range(iter_num):
            grad,feature=sess.run([pts_grad,mlp_layer],feed_dict={input:pts,is_training:False,feature_grad:feature_active_grad})
            pts+=lr*grad[0]
            # print 'relu max{}'.format(np.max(feature[0,:,:,dim]))
            print 'feature val {}'.format(np.mean(feature[0,:,:,dim]))

        # plt.hist(feature[0,:,:,dim],100)
        # plt.show()

        pts=pts[:,:,:,0]
        import random
        color=[int(random.random()*255) for _ in range(3)]
        with open('active_{}.txt'.format(dim),'w') as f:
            for index,pt in enumerate(pts[0]):
                if feature[0,index,:,dim]>1e-3:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],color[0],color[1],color[2]))

        print '{} done'.format(dim)

def show_activation():
    model_path='model/epoch404.ckpt'
    net=Network(3,40,True,1024)
    input=tf.placeholder(dtype=tf.float32,shape=[1,None,3,1],name='point_cloud')
    is_training=tf.placeholder(dtype=tf.bool,shape=[],name='is_training')

    net.inference(input,'cpu',is_training)
    # feature_layer=net.ops['cpu_pool']
    mlp_layer=net.ops['cpu_mlp5']

    config=tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess=tf.Session(config=config)
    saver=tf.train.Saver()
    saver.restore(sess,model_path)

    lr=1e-2
    # dim=0
    pt_num=4096

    pts = np.random.uniform(-1, 1, [ pt_num, 3])
    pts /= np.reshape(np.sqrt(pts[:,0]**2+pts[:,1]**2+pts[:,2]**2),[pt_num,1])+1e-7
    pts = np.random.uniform(0,1,[pt_num,1])*pts
    pts = np.reshape(pts,[1,pt_num,3,1])
    feature = sess.run(mlp_layer,feed_dict={input: pts, is_training: False})
    feature=feature[0,:,0,:]
    print feature.shape
    pts = pts[0, :, :, 0]

    dead_unit_num=0
    dead_unit_list=[]
    active_num_array=[]
    for dim in xrange(1024):
        print 'feature val {}'.format(np.max(feature[:,dim]))
        if np.max(feature[:,dim])<1e-8:
            dead_unit_num+=1
            dead_unit_list.append(dim)
            continue

        active_num=0
        indices=np.argsort(-feature[:,dim])
        max_feature_val=np.max(feature[:,dim])
        color=np.random.uniform(0,1,[3])
        color=255.0/np.max(color)*color
        with open('active_{}.txt'.format(dim),'w') as f:
            for i in xrange(pt_num):
                if feature[indices[i],dim]>1e-8:
                    this_color=feature[indices[i],dim]/max_feature_val*color
                    this_color=np.asarray(this_color,np.int)
                    f.write('{} {} {} {} {} {}\n'.format(
                        pts[indices[i],0],pts[indices[i],1],pts[indices[i],2],
                        this_color[0],this_color[1],this_color[2]))
                    active_num+=1

        active_num_array.append(active_num)

        print '{} done'.format(dim)

    plt.hist(active_num_array,20)
    plt.show()

    print 'dead node num {}'.format(dead_unit_num)

def read_category_file(file_name,num_class=40):
    maps={}
    names=[None for _ in xrange(num_class)]
    with open(file_name,'r') as f:
        for line in f.readlines():
            maps[line.split(' ')[0]]=int(line.split(' ')[1])

    for key,val in maps.items():
        names[val]=key

    return maps,names

def test_model():
    _,names=read_category_file('data/ModelNet40/CategoryIDs')
    model_path='model/1024_leaky_relu/epoch499.ckpt'
    net=Network(3,40,True,1024)
    input=tf.placeholder(dtype=tf.float32,shape=[None,None,3,1],name='point_cloud')
    is_training=tf.placeholder(dtype=tf.bool,shape=[],name='is_training')

    net.inference(input,'cpu',is_training,leaky_relu)
    score_layer=net.ops['cpu_fc3']

    config=tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess=tf.Session(config=config)
    saver=tf.train.Saver()
    saver.restore(sess,model_path)

    batch_files=['data/ModelNet40/test0.batch']
    batch_size=30
    reader=ModelBatchReader(batch_files,batch_size,4,2048,3,'test',PointSample.getPointCloud,normalize)
    correct_num=0
    error_num=0
    all_labels=[]
    all_preds=[]
    # reader.cur_pos=1000
    for data,label in reader:
        scores=sess.run(score_layer,feed_dict={input:data,is_training:False})
        preds=np.argmax(scores,axis=1)
        all_labels.append(label)
        all_preds.append(preds)
        correct_num+=np.sum(preds==label)
        # for i in xrange(data.shape[0]):
        #     if preds[i]==label[i]:
        #         continue
        #
        #     with open('misclassified/{}_{}_{}_{:.3}_{:.3}.txt'.format(
        #             names[label[i]],names[preds[i]],
        #             error_num,
        #             scores[i,preds[i]],scores[i,label[i]]),'w') as f:
        #         for pt in data[i,:,:,0]:
        #             f.write('{} {} {}\n'.format(pt[0],pt[1],pt[2]))
        #
        #     error_num+=1

    print 'accuracy {}'.format(correct_num/float(reader.total_size))

    cnf_matrix = confusion_matrix(np.concatenate(all_labels,axis=0), np.concatenate(all_preds,axis=0),labels=range(40))
    plt.figure()
    plot_confusion_matrix(cnf_matrix,names)
    plt.show()


def read_pointnet_sample_category_file(file_name):
    names=[]
    with open(file_name,'r') as f:
        for line in f.readlines():
            if line=="":
                break
            names.append(line[:-1])

    print names
    return names

def map_provided_label_to_batch_label(batch_names,provided_names):
    batch_map={name:index for index,name in enumerate(batch_names)}
    provided_map={index:name for index,name in enumerate(provided_names)}

    index_map={key:batch_map[val] for key,val in provided_map.items()}

    return index_map

import h5py
import math
def test_model_pointnet_sample_version():
    data=[]
    label=[]
    for i in range(2):
        f=h5py.File('/home/pal/data/ModelNet40PointNetSampleVersion/modelnet40_ply_hdf5_2048/ply_data_test{}.h5'.format(i))
        data.append(f['data'][:])
        label.append(f['label'][:])
        print data[i].shape
        print label[i].shape

    data=np.concatenate(data,axis=0)
    label=np.concatenate(label,axis=0)
    label=label[:,0]
    print data.shape,label.shape

    _,batch_names=read_category_file('data/ModelNet40/CategoryIDs')
    provided_names=read_pointnet_sample_category_file('/home/pal/data/ModelNet40PointNetSampleVersion/modelnet40_ply_hdf5_2048/shape_names.txt')
    index_map=map_provided_label_to_batch_label(batch_names,provided_names)

    rectify_label=np.empty_like(label)
    for index,l in enumerate(label):
        rectify_label[index]=index_map[l]
    label=rectify_label

    model_path='/home/pal/model/1024_leaky_relu/epoch499.ckpt'
    net=Network(3,40,True,1024)
    input=tf.placeholder(dtype=tf.float32,shape=[None,None,3,1],name='point_cloud')
    is_training=tf.placeholder(dtype=tf.bool,shape=[],name='is_training')

    net.inference(input,'cpu',is_training,leaky_relu)
    score_layer=net.ops['cpu_fc3']

    config=tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess=tf.Session(config=config)
    saver=tf.train.Saver()
    saver.restore(sess,model_path)

    batch_size=30
    iter_num=int(math.ceil(data.shape[0]/float(batch_size)))

    correct_num=0
    all_labels=[]
    all_preds=[]
    for batch_index in xrange(iter_num):
        begin_index=batch_size*batch_index
        end_index=min((batch_index+1)*batch_size,data.shape[0])
        batch_label=label[begin_index:end_index]
        batch_data=data[begin_index:end_index]
        batch_data=normalize(batch_data)
        batch_data=exchange_dims_zy(batch_data)
        batch_data=np.expand_dims(batch_data,axis=3)

        scores=sess.run(score_layer,feed_dict={input:batch_data,is_training:False})
        preds=np.argmax(scores,axis=1)
        all_labels.append(batch_label)
        all_preds.append(preds)
        correct_num+=np.sum(preds==batch_label)
        # for i in xrange(data.shape[0]):
        #     if preds[i]==label[i]:
        #         continue
        #
        #     with open('misclassified/{}_{}_{}_{:.3}_{:.3}.txt'.format(
        #             names[label[i]],names[preds[i]],
        #             error_num,
        #             scores[i,preds[i]],scores[i,label[i]]),'w') as f:
        #         for pt in data[i,:,:,0]:
        #             f.write('{} {} {}\n'.format(pt[0],pt[1],pt[2]))
        #
        #     error_num+=1
        # print batch_names[batch_label[0]]
        # with open('test.txt','w') as f:
        #     for pt in batch_data[0,:,:,0]:
        #         f.write('{} {} {}\n'.format(pt[0],pt[1],pt[2]))

    print 'accuracy {}'.format(correct_num/float(data.shape[0]))

    cnf_matrix = confusion_matrix(np.concatenate(all_labels,axis=0), np.concatenate(all_preds,axis=0),labels=range(40))
    plt.figure()
    plot_confusion_matrix(cnf_matrix,batch_names)
    plt.show()



def draw_pointnet_sample_version():
    # f=h5py.File('/home/pal/data/ModelNet40PointNetSampleVersion/modelnet40_ply_hdf5_2048/ply_data_test0.h5')
    data=[]
    label=[]
    for i in range(5):
        f=h5py.File('/home/pal/data/ModelNet40PointNetSampleVersion/modelnet40_ply_hdf5_2048/ply_data_train{}.h5'.format(i))
        data.append(f['data'][:])
        label.append(f['label'][:])

    data=np.concatenate(data,axis=0)
    print data.shape

    index=np.random.randint(0,2048,1)

    with open('test.txt','w') as f:
        for pt in data[int(index),:,:]:
            f.write('{} {} {}\n'.format(pt[0],pt[1],pt[2]))


    # batch_files=['data/ModelNet40/train0.batch',
    #              'data/ModelNet40/train1.batch',
    #              'data/ModelNet40/train2.batch',
    #              'data/ModelNet40/train3.batch']
    # batch_size=5
    # reader=ModelBatchReader(batch_files,batch_size,4,2048,3,'test',PointSample.getPointCloud,normalize)
    #
    # print reader.total_size



if __name__=="__main__":
    test_model_pointnet_sample_version()