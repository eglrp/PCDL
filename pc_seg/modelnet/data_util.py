import h5py
import pyflann
import numpy as np
import math
import os


def read_points(points_h5):
    f=h5py.File(points_h5, 'r')
    data,label = f['data'][:],f['label'][:]
    f.close()

    return data, label


def save_modelnet_v2(filename,points,nidxs,labels):
    h5_fout = h5py.File(filename,'w')
    h5_fout.create_dataset(
            'point', data=points,
            compression='gzip', compression_opts=4,
            dtype='float32')
    h5_fout.create_dataset(
            'label', data=labels,
            compression='gzip', compression_opts=1,
            dtype='uint8')
    h5_fout.create_dataset(
            'nidx', data=nidxs,
            compression='gzip', compression_opts=4,
            dtype='uint32')
    h5_fout.close()
    return


def read_modelnet_v2(filename):
    f=h5py.File(filename, 'r')
    data,label,nidxs = f['point'][:],f['label'][:],f['nidx'][:]
    f.close()
    return data,nidxs,label


def compute_covariance(points):
    '''
    :param points: n,3
    :return: covariance matrix 1,9
    '''
    center=np.mean(points,axis=0,keepdims=True)
    centered_points=points - center
    return (centered_points.transpose().dot(centered_points)).flatten()


def points2covars(points,nn_size):
    '''

    :param points: [n,k,3]
    :return:
    '''
    n,k,_=points.shape
    covars=np.empty([n,k,9])
    for pts_i,pts in enumerate(points):
        flann=pyflann.FLANN()
        flann.build_index(pts,algorithm='kdtree_simple',leaf_max_size=15)
        for pt_i,pt in enumerate(pts):
            idxs,dists=flann.nn_index(pt,nn_size+1)
            covar=compute_covariance(pts[idxs[0,1:]])
            covars[pts_i,pt_i]=covar.flatten()

    return covars


def compute_acc(labels, preds, num_classes):
    correct=np.zeros(num_classes,np.float32)
    incorrect=np.zeros(num_classes,np.float32)
    correct_mask=labels==preds
    incorrect_mask=labels!=preds
    for i in xrange(num_classes):
        label_mask=labels==i

        correct[i]=np.sum(correct_mask&label_mask)
        incorrect[i]=np.sum(incorrect_mask&label_mask)

    acc=correct/(correct+incorrect)
    macc=np.mean(acc)
    oacc=np.sum(correct)/np.sum(correct+incorrect)

    return acc,macc,oacc


def compute_nidxs(points,nn_size=16):
    '''
    :param points: [n,k,3]
    :return:
    '''
    n,k,_=points.shape
    nidxs=np.empty([n,k,nn_size],dtype=np.int)
    for pts_i,pts in enumerate(points):
        flann=pyflann.FLANN()
        flann.build_index(pts,algorithm='kdtree_simple',leaf_max_size=15)
        for pt_i,pt in enumerate(pts):
            idxs,_=flann.nn_index(pt,nn_size+1)
            nidxs[pts_i,pt_i]=idxs[0,1:]

    return nidxs


def prepare_nidxs():
    stems=['ply_data_test{}.h5'.format(i) for i in range(2)]
    for stem in stems:
        fn='../data/ModelNet40PointNetSampleVersion/modelnet40_ply_hdf5_2048/'+stem
        points,labels=read_points(fn)
        nidixs=compute_nidxs(points,16)
        save_modelnet_v2(stem,points,nidixs,labels)
        print 'done'


def compute_covar(points,nidxs):
    n,k,_=points.shape
    covars=np.empty([n,k,9])
    for pts_i,pts in enumerate(points):
        for pt_i,pt in enumerate(pts):
            all_npts=pts[nidxs[pts_i,pt_i]]
            all_npts=np.concatenate([all_npts,pt[None,:]],axis=0)
            covar=compute_covariance(all_npts)
            covars[pts_i,pt_i]=covar

    return covars


def exchange_dims_zy(points):
    #pcs [n,k,3]
    exchanged_data = np.empty(points.shape, dtype=np.float32)

    exchanged_data[:,:,0]= points[:, :, 0]
    exchanged_data[:,:,1]= points[:, :, 2]
    exchanged_data[:,:,2]= points[:, :, 1]
    return exchanged_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def rotate(points,angle=None):
    '''
    :param points: [n,k,3]
    :return:
    '''
    rotated_data = np.empty(points.shape, dtype=np.float32)
    for k in range(points.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi if angle is None else angle
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval,  0],
                                    [      0,      0, 1]],dtype=np.float32)
        shape_pc = points[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc, rotation_matrix)
    return rotated_data


def get_classes_name():
    path = os.path.split(os.path.realpath(__file__))[0]
    f = open(path + '/shape_names.txt', 'r')
    names = [line.strip('\n') for line in f.readlines()]
    f.close()
    return names


def test_nidxs():
    import sys
    sys.path.append('..')
    from s3dis.draw_util import output_points
    points, nidxs, label=read_modelnet_v2('ply_data_train0.h5')
    k=np.random.randint(0,points.shape[0])
    t=0
    for pt,idx in zip(points[k],nidxs[k]):
        colors=np.random.randint(0,255,3)
        colors=np.repeat(colors[None,:],[17])
        pts=np.concatenate([pt[None,:],points[k][idx]],axis=0)
        output_points('test{}.txt'.format(t),pts,colors)
        t+=1


def test_compute_covars():
    import time
    import Points2Voxel
    import sys
    sys.path.append('..')
    from s3dis.draw_util import output_points

    points, nidxs, labels=read_modelnet_v2('../data/ModelNetTrain/nidxs/ply_data_train2.h5')
    begin=time.time()
    # points=exchange_dims_zy(points)
    # points=rotate(points)

    nidxs=np.ascontiguousarray(nidxs,dtype=np.int32)
    points=np.ascontiguousarray(points,dtype=np.float32)

    # covars_np=compute_covar(points,nidxs)
    # covars_np/=np.sqrt(np.sum(covars_np**2,axis=2,keepdims=True))
    # print np.sum(covars_np)
    # print covars_np[0,0,:]

    covars=Points2Voxel.ComputeCovars(points,nidxs,16,0)
    # print np.sum(covars)
    # print covars[0,0,:]
    print 'cost {} s'.format(time.time()-begin,)#np.mean(np.abs(covars-covars_np)))
    # print np.sqrt(np.sum(covars_np**2,axis=2))
    # print np.sqrt(np.sum(covars**2,axis=2))

    from sklearn.cluster import KMeans
    kmeans=KMeans(5)
    preds=kmeans.fit_predict(covars[1])
    colors=np.random.randint(0,255,[5,3])
    output_points('cluster.txt',points[1],colors[preds,:])

if __name__=="__main__":
    left_num=2
    file_list=['../data/ModelNetTrain/nidxs/ply_data_test{}.h5'.format(i) for i in xrange(2)]
    points,nidxs,labels=[],[],[]
    for f in file_list:
        point,nidx,label=read_modelnet_v2(f)
        points.append(point)
        nidxs.append(nidx)
        labels.append(label)

    points=np.concatenate(points,axis=0)
    nidxs=np.concatenate(nidxs,axis=0)
    labels=np.concatenate(labels,axis=0)[:,0]

    left_points,left_nidxs,left_labels=[],[],[]
    retain_points,retain_nidxs,retain_labels=[],[],[]
    for i in xrange(40):
        mask=labels==i
        print np.sum(mask)
        left_points.append(points[mask][:left_num])
        left_nidxs.append(nidxs[mask][:left_num])
        left_labels.append(labels[mask][:left_num])

        retain_points.append(points[mask][left_num:])
        retain_nidxs.append(nidxs[mask][left_num:])
        retain_labels.append(labels[mask][left_num:])

    left_points=np.concatenate(left_points,axis=0)
    left_nidxs=np.concatenate(left_nidxs,axis=0)
    left_labels=np.concatenate(left_labels,axis=0)

    retain_points=np.concatenate(retain_points,axis=0)
    retain_nidxs=np.concatenate(retain_nidxs,axis=0)
    retain_labels=np.concatenate(retain_labels,axis=0)

    save_modelnet_v2('ply_data_test_voxel0.txt',retain_points,retain_nidxs,retain_labels)
    save_modelnet_v2('ply_data_test_voxel1.txt',left_points,left_nidxs,left_labels)
