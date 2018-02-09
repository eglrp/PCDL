import math
import os
import h5py
import numpy as np
import cPickle


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


def get_block_train_test_split(test_area=5):
    '''
    :param test_area: default use area 5 as testset
    :return:
    '''
    path = os.path.split(os.path.realpath(__file__))[0]
    f = open(path + '/room_block_stems.txt', 'r')
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


def save_room_pkl(filename,points,labels):
    with open(filename,'wb') as f:
        cPickle.dump((points,labels),f,protocol=2)


def read_room_pkl(filename):
    with open(filename,'rb') as f:
        points,labels=cPickle.load(f)
    return points,labels


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


def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample-N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), range(N)+list(sample)


def sample_data_label(data, label, num_sample):
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]
    return new_data, new_label


def room2blocks(data, label, num_point=4096, block_size=1.0, stride=1.0,
                random_sample=False, sample_num=None, sample_aug=1, without_sample=False):
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters
        stride: float, stride for block sweeping
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area]
        sample_aug: if random sample, how much aug
        without_sample: use original points
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    limit = np.amax(data, 0)[0:3]

    # Get the corner location for our sampling blocks
    xbeg_list = []
    ybeg_list = []
    if not random_sample:
        num_block_x = int(np.ceil((limit[0] - stride) / stride)) + 1
        num_block_y = int(np.ceil((limit[1] - stride) / stride)) + 1
        for i in range(-1,num_block_x):
            for j in range(-1,num_block_y):
                xbeg_list.append(i * stride)
                ybeg_list.append(j * stride)
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0])
            ybeg = np.random.uniform(-block_size, limit[1])
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)

    # Collect blocks
    block_data_list = []
    block_label_list = []
    block_beg_list=[]
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        xcond = (data[:, 0] <= xbeg + block_size) & (data[:, 0] >= xbeg)
        ycond = (data[:, 1] <= ybeg + block_size) & (data[:, 1] >= ybeg)
        cond = xcond & ycond
        if np.sum(cond) < 100:  # discard block if there are less than 100 pts.
            continue

        block_data = data[cond, :]
        block_label = label[cond]

        block_beg_list.append((xbeg,ybeg))
        if not without_sample:
            # randomly subsample data
            block_data_sampled, block_label_sampled = \
                sample_data_label(block_data, block_label, num_point)
            block_data_list.append(np.expand_dims(block_data_sampled, 0))
            block_label_list.append(np.expand_dims(block_label_sampled, 0))
        else:
            block_data_list.append(np.copy(block_data))
            block_label_list.append(np.copy(block_label))

    if not without_sample:
        block_data_list,block_label_list=np.concatenate(block_data_list, 0),np.concatenate(block_label_list, 0)

    return block_data_list,block_label_list,block_beg_list


def compute_iou(label,pred):
    fp = np.zeros(13, dtype=np.int)
    tp = np.zeros(13, dtype=np.int)
    fn = np.zeros(13, dtype=np.int)
    # for l, p in zip(label, pred):
    #     if l == p:
    #         tp[l] += 1
    #     else:
    #         fp[p] += 1
    #         fn[l] += 1

    correct_mask=label==pred
    incorrect_mask=label!=pred
    for i in range(13):
        label_mask=label==i
        pred_mask=pred==i

        tp[i]=np.sum(correct_mask&label_mask)
        fn[i]=np.sum(incorrect_mask&label_mask)
        fp[i]=np.sum(incorrect_mask&pred_mask)

    iou = tp / (fp + fn + tp + 1e-6).astype(np.float)
    miou=np.mean(iou)
    oiou=np.sum(tp) / float(np.sum(tp + fn + fp))
    acc = tp / (tp + fn + 1e-6)
    macc = np.mean(acc)
    oacc = np.sum(tp) / float(np.sum(tp+fn))

    return iou, miou, oiou, acc, macc, oacc


if __name__=="__main__":
    pass