import tensorflow as tf
from network_class import Network
from preprocess import ModelBatchReader,add_noise,normalize,rotate
import PointSample
import numpy as np
import matplotlib.pyplot as plt


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

if __name__=="__main__":
    show_activation()