import pyflann
from s3dis_util.util import read_room_h5,points_downsample
import time
from cls_network import Network
import tensorflow as tf
import numpy as np

from sklearn.cluster import KMeans

def declare_cls_network(model_path):
    input_dims=3
    num_classes=40

    net=Network(input_dims,num_classes,True,1024)
    inputs=tf.placeholder(tf.float32, [1, None, input_dims],name='inputs')
    is_training=tf.placeholder(tf.bool,name='is_training')
    net.inference(inputs,'cpu',is_training)
    feature_layer=net.ops['cpu_pool']

    config=tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess=tf.Session(config=config)
    saver=tf.train.Saver()
    saver.restore(sess,model_path)

    return sess,feature_layer,inputs,is_training

def compute_feature(pts,sess,feature_layer,inputs,is_training):
    pts=np.expand_dims(pts,axis=0)
    feature=sess.run(feature_layer,feed_dict={inputs:pts,is_training:False})
    return feature

def compute_room_feature(room_h5_file='data/S3DIS/room/260_Area_6_office_35.h5',
                         model_path='model/1024_leaky_relu/epoch499.ckpt',
                         downsample_radius=0.05,
                         feature_filename="features_0.05_a6o35.npy"):

    data,label=read_room_h5(room_h5_file)
    print data.shape
    coords=data[:,:3]
    pyflann.set_distance_type('euclidean')
    flann=pyflann.FLANN(algorithm='kdtree_simple',leaf_max_size=15)
    flann.build_index(coords)

    sess, feature_layer, inputs, is_training=declare_cls_network(model_path)

    # print coords.shape
    down_pts=points_downsample(coords, downsample_radius)
    # print down_pts.shape
    # with open('down.txt','w') as f:
    #     for npt in down_pts:
    #         f.write('{} {} {}\n'.format(npt[0],npt[1],npt[2]))

    begin=time.time()
    features=[]
    radius=0.2
    for pt_i,pt in enumerate(down_pts):
        indices,dists=flann.nn_radius(pt,radius*radius)
        neighbor_pts=np.copy(coords[indices,:])
        neighbor_pts-=np.expand_dims(pt,axis=0)
        neighbor_pts/=radius
        feature=compute_feature(neighbor_pts,sess, feature_layer, inputs, is_training)
        features.append(feature)
        if pt_i%1000==0:
            print '{} cost {} s'.format(pt_i,time.time()-begin)

    print 'cost {} s'.format(time.time()-begin)

    features=np.concatenate(features,axis=0)
    print features.shape
    np.save(feature_filename,features)

if __name__=="__main__":
    data,label=read_room_h5('data/S3DIS/room/260_Area_6_office_35.h5')
    coords=data[:,:3]
    down_pts=points_downsample(coords, 0.05)

    features=np.load('features_0.05_a6o35.npy')

    cluster_num=10

    pred=KMeans(n_clusters=cluster_num,n_jobs=-1).fit_predict(features)

    print pred.shape

    colors=np.random.randint(0,255,[cluster_num,3],dtype=np.int)
    for i in range(cluster_num):
        with open("{}.txt".format(i),'w') as f:
            for pt in down_pts[pred==i]:
                f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],colors[i,0],colors[i,1],colors[i,2]))