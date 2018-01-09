import os
import h5py
import numpy as np
import math


def get_class_names():
    names=[]
    path=os.path.split(os.path.realpath(__file__))[0]
    with open(path+'/class_names.txt','r') as f:
        for line in f.readlines():
            names.append(line.strip('\n'))

    return names


def get_room_dirs():
    dirs=[]
    path=os.path.split(os.path.realpath(__file__))[0]
    with open(path+'/room_dirs.txt','r') as f:
        for line in f.readlines():
            dirs.append(line.strip('\n'))

    return dirs


def get_train_test_split(test_area=5):
    '''
    :param test_area: default use area 5 as testset
    :return:
    '''
    path = os.path.split(os.path.realpath(__file__))[0]
    f = open(path + '/room_stems.txt', 'r')
    file_stems = [line.strip('\n') for line in f.readlines()]
    f.close()

    train, test = [], []
    for fs in file_stems:
        if fs.split('_')[2] == str(test_area):
            test.append(fs)
        else:
            train.append(fs)

    return train, test


def read_room_h5(room_h5_file):
    f=h5py.File(room_h5_file,'r')
    data,label = f['data'][:],f['label'][:]
    f.close()

    return data, label


def get_class_colors():
    colors=np.asarray(
            [[0,255,0],
            [0,0,255],
            [93,201,235],
            [255,255,0],
            [255,140,0],
            [0,0,128],
            [255,69,0],
            [255,127,80],
            [255,0,0],
            [255,250,240],
            [255,0,255],
            [255,255,255],
            [105,105,105],
            [205,92,92]],dtype=np.int)
    return colors


def downsample_average(points, sample_stride=0.1):
    '''
    :param points: [n,f] f>=3, x y z ...
    :param sample_stride:
    :return:
        downsample_points:
        indices: same size as points [n,],
                 indicating which downsampled point the original points contribute to.
    '''
    min_coor=np.min(points[:, :3], axis=0, keepdims=True)
    points[:, :3]-=min_coor

    loc2pt={}
    for pt_index,pt in enumerate(points[:, :3]):
        x_index=int(math.ceil(pt[0]/sample_stride))
        y_index=int(math.ceil(pt[1]/sample_stride))
        z_index=int(math.ceil(pt[2]/sample_stride))
        loc=(x_index,y_index,z_index)
        if loc in loc2pt:
            loc2pt[loc].append(pt_index)
        else:
            loc2pt[loc]=[pt_index]

    downsample_points=[]
    indices=np.empty(points.shape[0], dtype=np.int32)
    for k,v in loc2pt.items():
        downsample_points.append(np.mean(points[v, :], axis=0, keepdims=True))
        indices[v]=len(downsample_points)-1

    points[:, :3]+=min_coor
    downsample_points=np.concatenate(downsample_points,axis=0)
    downsample_points[:,:3]+=min_coor

    return downsample_points, indices


def downsample_random(points,labels,sample_stride=0.1):
    '''
    :param points:  [n,f]
    :param labels:  [n,]
    :param sample_stride:
    :return:
        downsample_points: [n_downsample,f]
        downsample_labels: [n_downsample,]
        downsample_indices: [n_downsample,] indicating the index of downsampled point in the original point cloud
    '''
    min_coor=np.min(points[:, :3], axis=0, keepdims=True)
    points[:, :3]-=min_coor

    loc2pt={}
    for pt_index,pt in enumerate(points[:, :3]):
        x_index=int(math.ceil(pt[0]/sample_stride))
        y_index=int(math.ceil(pt[1]/sample_stride))
        z_index=int(math.ceil(pt[2]/sample_stride))
        loc=(x_index,y_index,z_index)
        if loc in loc2pt:
            loc2pt[loc].append(pt_index)
        else:
            loc2pt[loc]=[pt_index]

    downsample_points=[]
    downsample_indices=[]
    downsample_labels=[]
    for k,v in loc2pt.items():
        # grid_points=points[v,:]
        grid_index=int(np.random.randint(0,len(v),1))
        downsample_points.append(np.expand_dims(points[v[grid_index],:],axis=0))
        downsample_labels.append(labels[v[grid_index]])
        downsample_indices.append(v[grid_index])

    downsample_labels=np.concatenate(downsample_labels,axis=0)
    downsample_indices=np.stack(downsample_indices,axis=0)
    downsample_points=np.concatenate(downsample_points,axis=0)
    downsample_points[:,:3]+=min_coor
    points[:, :3]+=min_coor
    return downsample_points,downsample_labels,downsample_indices


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