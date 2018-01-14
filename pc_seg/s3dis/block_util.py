import pyflann
from data_util import *
import numpy as np
from concurrent.futures import ProcessPoolExecutor,wait


def save_block(filename, points, nidxs, covars, rpoints, labels):
    '''

    :param filename:
    :param points: n,k,3
    :param nidxs:  n,k,8
    :param covars: n,k,9
    :param rpoints: n,k,3 normalized room coordinates
    :param labels: n,k
    :return:
    '''
    h5_fout = h5py.File(filename,'w')
    h5_fout.create_dataset(
            'point', data=points,
            compression='gzip', compression_opts=4,
            dtype='float32')
    h5_fout.create_dataset(
            'covar', data=covars,
            compression='gzip', compression_opts=4,
            dtype='float32')
    h5_fout.create_dataset(
            'label', data=labels,
            compression='gzip', compression_opts=1,
            dtype='uint8')
    h5_fout.create_dataset(
            'rpoint', data=rpoints,
            compression='gzip', compression_opts=4,
            dtype='float32')
    h5_fout.create_dataset(
            'nidx', data=nidxs,
            compression='gzip', compression_opts=4,
            dtype='uint32')
    h5_fout.close()


def read_block(filename):
    h5_fin = h5py.File(filename,'r')
    points, nidxs, covars, rpoints, labels= \
        h5_fin['point'][:],h5_fin['nidx'][:],h5_fin['covar'][:],h5_fin['rpoint'][:],h5_fin['label'][:]
    return points, nidxs, covars, rpoints, labels


def save_block_v2(filename, points, covars, rpoints, labels):
    '''

    :param filename:
    :param points: n,k,3
    :param covars: n,k,9
    :param rpoints: n,k,3 normalized room coordinates
    :param labels: n,k
    :return:
    '''
    h5_fout = h5py.File(filename,'w')
    h5_fout.create_dataset(
            'point', data=points,
            compression='gzip', compression_opts=4,
            dtype='float32')
    h5_fout.create_dataset(
            'covar', data=covars,
            compression='gzip', compression_opts=4,
            dtype='float32')
    h5_fout.create_dataset(
            'label', data=labels,
            compression='gzip', compression_opts=1,
            dtype='uint8')
    h5_fout.create_dataset(
            'rpoint', data=rpoints,
            compression='gzip', compression_opts=4,
            dtype='float32')
    h5_fout.close()


def read_block_v2(filename):
    h5_fin = h5py.File(filename,'r')
    points, covars, rpoints, labels= \
        h5_fin['point'][:],h5_fin['covar'][:],h5_fin['rpoint'][:],h5_fin['label'][:]
    return points, covars, rpoints, labels


def compute_covariance(points):
    '''
    :param points: n,3
    :return: covariance matrix 1,9
    '''
    center=np.mean(points,axis=0,keepdims=True)
    centered_points=points - center
    return (centered_points.transpose().dot(centered_points)).flatten()


def compute_nidxs_covars(points, covar_size=16, neighbor_size=8):
    '''
    :param points: n,3
    :param covar_size:
    :param neighbor_size:
    :return:
        nidxs:  n,neighbor_size
        covars: n,9
    '''

    assert neighbor_size<=covar_size

    points_num=points.shape[0]

    flann = pyflann.FLANN()
    flann.build_index(points, algorithm='kdtree_simple', leaf_max_size=15)
    nidxs = np.empty([points_num, neighbor_size])
    covars = np.empty([points_num,9])
    for pt_i, pt in enumerate(points):
        cur_indices, _ = flann.nn_index(pt, covar_size)                     # 1,covar_size
        cur_indices = np.asarray(cur_indices, dtype=np.int).transpose()     # covar_size,1
        # nidxs
        nidxs[pt_i,:] = cur_indices[neighbor_size, 0]                       # neighbor_size
        # covars
        covars[pt_i,:]=compute_covariance(points[cur_indices[:,0],:])

    return nidxs,covars


def compute_radius_covars(points, radius=0.1):

    points_num=points.shape[0]

    flann = pyflann.FLANN()
    flann.build_index(points, algorithm='kdtree_simple', leaf_max_size=15)
    covars = np.empty([points_num,9])
    for pt_i, pt in enumerate(points):
        cur_indices, _ = flann.nn_radius(pt, radius)                     # 1,covar_size
        cur_indices = np.asarray(cur_indices, dtype=np.int).transpose()     # covar_size,1
        # covars
        covars[pt_i,:]=compute_covariance(points[cur_indices,:])

    return covars


def compute_rpoints_list(points_list,room_points):
    '''
    :param points_list: n,k,3
    :param room_points: m,3   already offset to original (min==0)
    :return:
    '''
    rmax=np.max(room_points,axis=0)
    rpoints_list=points_list/rmax[None,None,:]
    return rpoints_list


def normalize_covars_list(covars_list):
    '''
    normalize to norm=1
    :param covars_list: n,k,9
    :return:
    '''
    covars_list/=np.sqrt(np.sum(covars_list ** 2, axis=2, keepdims=True)+1e-6)
    return covars_list


def normalize_points_list(points_list, block_size=1.0):
    '''

    :param points_list: n,k,6 xyzrgb
    :return:
    '''
    # normalize coordinates
    points_list[:, :, :3]-=np.min(points_list[:, :, :3], axis=1, keepdims=True)
    points_list[:, :, 0]-= block_size / 2.0
    points_list[:, :, 1]-= block_size / 2.0
    points_list[:, :, 2]/=np.max(points_list[:, :, 2],axis=1,keepdims=True)

    # normalize colors
    points_list[:, :, 3:]-=128
    points_list[:, :, 3:]/=128

    return points_list


def replace_stairs_labels(labels_list):
    '''
    :param labels_list: n,k
    :return:
    '''
    labels_list[labels_list==13]=12
    return labels_list


def prepare_room_v2(room_fn,save_fn):
    room_points,room_labels=read_room_h5(room_fn)
    room_points-=np.min(room_points,axis=0)

    # points_list n,k,3 labels_list n,k   k=4096
    points_list,labels_list,begs_list=room2blocks(room_points,room_labels,stride=0.5)

    rpoints_list=compute_rpoints_list(points_list[:,:,:3],room_points[:,:3])

    covars_list=[]
    for points in points_list:
        covar=compute_radius_covars(points[:,:3],0.1)
        covars_list.append(np.expand_dims(covar,axis=0))

    covars_list=np.concatenate(covars_list,axis=0)    # n,k,9

    points_list=normalize_points_list(points_list)
    covars_list=normalize_covars_list(covars_list)

    labels_list=np.squeeze(labels_list,axis=2)

    save_block_v2(save_fn,points_list,covars_list,rpoints_list,labels_list)


def prepare_room(room_fn,save_fn):
    room_points,room_labels=read_room_h5(room_fn)
    room_points-=np.min(room_points,axis=0)

    # points_list n,k,3 labels_list n,k   k=4096
    points_list,labels_list,begs_list=room2blocks(room_points,room_labels,stride=0.5)

    rpoints_list=compute_rpoints_list(points_list[:,:,:3],room_points[:,:3])

    nidxs_list,covars_list=[],[]
    for points in points_list:
        nidxs,covar=compute_nidxs_covars(points[:,:3])
        nidxs_list.append(np.expand_dims(nidxs,axis=0))
        covars_list.append(np.expand_dims(covar,axis=0))

    nidxs_list=np.concatenate(nidxs_list,axis=0)    # n,k,3
    covars_list=np.concatenate(covars_list,axis=0)    # n,k,9

    points_list=normalize_points_list(points_list)
    covars_list=normalize_covars_list(covars_list)

    labels_list=np.squeeze(labels_list,axis=2)

    save_block(save_fn,points_list,nidxs_list,covars_list,rpoints_list,labels_list)


def prepare_dataset():
    train_fs,test_fs=get_train_test_split()
    train_fs+=test_fs
    room_path='../data/S3DIS/room/'
    save_path='../data/S3DIS/folding/block_v2/'
    executor=ProcessPoolExecutor(8)

    futures=[]
    for fs in train_fs[180:]:
        room_fn=room_path+fs+'.h5'
        save_fn=save_path+fs+'.h5'
        futures.append(executor.submit(prepare_room_v2,room_fn,save_fn))

    wait(futures)


def merge_data(batch_data):
    batch_num=len(batch_data)
    data_num=len(batch_data[0])

    data_collect=[[] for _ in xrange(data_num)]
    for i in xrange(batch_num):
        for j in xrange(data_num):
            data_collect[j].append(batch_data[i][j])

    for j in xrange(data_num):
        data_collect[j]=np.concatenate(data_collect[j],axis=0)

    return data_collect


def merge_set(all_fs,name):
    cur_idx,cur_size,batch_data=0,0,[]
    for fs in all_fs:
        data=read_block('../data/S3DIS/folding/block/'+fs+'.h5')
        batch_data.append(data)
        cur_size+=data[0].shape[0]

        if cur_size>1000:
            merged_data=merge_data(batch_data)

            for item in merged_data:
                print item.shape

            save_block('../data/S3DIS/folding/block_{}{}.h5'.format(name,cur_idx),*merged_data)
            cur_idx+=1
            cur_size=0
            batch_data=[]

    if cur_size>0:
        save_block('../data/S3DIS/folding/block_{}{}.h5'.format(name,cur_idx), *merge_data(batch_data))


def merge_train_test():
    train_fs,test_fs=get_train_test_split()

    merge_set(train_fs,'train')
    merge_set(test_fs,'test')


def replace_stairs_all():
    train_fs,test_fs=get_train_test_split()
    train_fs+=test_fs
    save_path='../data/S3DIS/folding/block_v2/'
    for fn in train_fs:
        h5_fn=save_path+fn+'.h5'
        points, covars, rpoints, labels = read_block_v2(h5_fn)
        labels=replace_stairs_labels(labels)
        save_block_v2(h5_fn, points, covars, rpoints, labels)
        print '{} done'.format(h5_fn)


def test_prepare_dataset():
    points, nidxs, covars, rpoints, labels = read_block('../data/S3DIS/folding/block/0_Area_1_conferenceRoom_1.h5')

    print 'points:'
    print np.min(points,axis=(0,1))
    print np.max(points,axis=(0,1))

    print 'nidixs'
    print np.min(nidxs,axis=(0,1))
    print np.max(nidxs,axis=(0,1))

    print 'rpoints'
    print np.min(rpoints,axis=(0,1))
    print np.max(rpoints,axis=(0,1))

    print 'covars'
    print np.min(np.sum(covars**2,axis=2)),np.max(np.sum(covars**2,axis=2))

    print 'labels'
    print np.min(labels),np.max(labels)

    from draw_util import output_points

    ccolors=get_class_colors()
    for block_i,block_rpoitns in enumerate(rpoints):
        colors=np.asarray(points[block_i,:,3:]*128+128,np.int)
        output_points('colors{}.txt'.format(block_i),block_rpoitns,colors)
        output_points('labels{}.txt'.format(block_i),block_rpoitns,ccolors[labels[block_i],:])


def test_radius_covar():
    from data_util import downsample_random
    from draw_util import output_points
    points,labels=read_room_h5('../data/S3DIS/room/0_Area_1_conferenceRoom_1.h5')
    print points.shape
    points,labels,_=downsample_random(points,labels,0.02)
    print points.shape
    output_points('test.txt',points)
    covars=compute_radius_covars(points[:, :3], 0.1)
    covars/=np.sqrt(np.sum(covars**2,axis=1,keepdims=True))

    from sklearn.cluster import KMeans
    kmeans=KMeans(8,n_jobs=-1)
    pred=kmeans.fit_predict(covars)
    print pred.shape
    colors=np.random.randint(0,255,[8,3])
    output_points('cluster.txt',points,colors[pred,:])


def test_prepare_v2():
    points, covars, rpoints, labels = read_block_v2('../data/S3DIS/folding/block_v2/5_Area_1_hallway_3.h5')

    print 'points:'
    print np.min(points,axis=(0,1))
    print np.max(points,axis=(0,1))

    print 'rpoints'
    print np.min(rpoints,axis=(0,1))
    print np.max(rpoints,axis=(0,1))

    print 'covars'
    print np.min(np.sum(covars**2,axis=2)),np.max(np.sum(covars**2,axis=2))

    print 'labels'
    print np.min(labels),np.max(labels)

    from draw_util import output_points

    ccolors=get_class_colors()
    for block_i,block_rpoitns in enumerate(rpoints):
        colors=np.asarray(points[block_i,:,3:]*128+128,np.int)
        output_points('colors{}.txt'.format(block_i),block_rpoitns,colors)
        output_points('labels{}.txt'.format(block_i),block_rpoitns,ccolors[labels[block_i],:])

    from sklearn.cluster import KMeans
    kmeans = KMeans(8, n_jobs=-1)
    covars=np.reshape(covars,[-1,9])
    pred=kmeans.fit_predict(covars)
    rpoints=np.reshape(rpoints,[-1,3])
    colors=np.random.randint(0,255,[8,3])
    output_points('cluster.txt',rpoints,colors[pred,:])


if __name__=="__main__":
    replace_stairs_all()

