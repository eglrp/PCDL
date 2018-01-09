import glob
import numpy as np
import os
import h5py
import math
import cPickle
import struct
import FPFHExtractor
from concurrent.futures import ProcessPoolExecutor,wait,ThreadPoolExecutor

def get_class_names():
    names=[]
    path=os.path.split(os.path.realpath(__file__))[0]
    with open(path+'/class_names.txt','r') as f:
        for line in f.readlines():
            names.append(line.strip('\n'))

    return names

def read_room_dirs():
    dirs=[]
    with open('room_dirs.txt','r') as f:
        for line in f.readlines():
            dirs.append(line.strip('\n'))

    return dirs

def read_room(room_dir,class_names):
    pcs=[]
    labels=[]
    for f in glob.glob(os.path.join(room_dir,'*.txt')):
        class_str=os.path.basename(f).split('_')[0]
        if class_str not in class_names:
            class_names.append(class_str)
        class_index=class_names.index(class_str)
        pc=np.loadtxt(f,dtype=np.float32)
        label=np.ones([pc.shape[0],1],dtype=np.uint8)*class_index
        pcs.append(pc)
        labels.append(label)

    pcs=np.concatenate(pcs,axis=0)
    labels=np.concatenate(labels,axis=0)

    return pcs,labels

def read_room_h5(room_h5_file):
    f=h5py.File(room_h5_file,'r')
    return f['data'][:],f['label'][:]

def read_pkl(filename, model='rb'):
    with open(filename, model) as f:
        obj=cPickle.load(f)
    return obj

def save_pkl(obj,filename,model='wb',protocol=2):
    with open(filename,model) as f:
        cPickle.dump(obj,f,protocol)

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
    beg_list=[]
    idx = 0
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

        beg_list.append((xbeg,ybeg))
        # randomly subsample data
        if not without_sample:
            block_data_sampled, block_label_sampled = \
                sample_data_label(block_data, block_label, num_point)
            block_data_list.append(np.expand_dims(block_data_sampled, 0))
            block_label_list.append(np.expand_dims(block_label_sampled, 0))
        else:
            block_data_list.append(np.copy(block_data))
            block_label_list.append(np.copy(block_label))

    if not without_sample:
        block_data_list,block_label_list=np.concatenate(block_data_list, 0),np.concatenate(block_label_list, 0)

    return block_data_list,block_label_list,beg_list

def room2blocks_with_indices(data, label, indices, block_size=1.0, stride=1.0):
    limit = np.amax(data, 0)[0:3]

    # Get the corner location for our sampling blocks
    xbeg_list = []
    ybeg_list = []
    num_block_x = int(np.ceil(limit[0] / stride)) + 1
    num_block_y = int(np.ceil(limit[1] / stride)) + 1
    for i in range(num_block_x):
        for j in range(num_block_y):
            xbeg_list.append(i * stride-(block_size-stride)/2.0)
            ybeg_list.append(j * stride-(block_size-stride)/2.0)

    # Collect blocks
    block_data_list = []
    block_label_list = []
    block_indices_list = []
    beg_list=[]
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
        block_indices = indices[cond]

        beg_list.append((xbeg,ybeg))
        block_data_list.append(np.copy(block_data))
        block_label_list.append(np.copy(block_label))
        block_indices_list.append(np.copy(block_indices))

    return block_data_list,block_label_list,block_indices_list,beg_list

def room2blocks_cond(data,block_size,stride,center_size):
    limit = np.amax(data, 0)[0:3]

    # Get the corner location for our sampling blocks
    xbeg_list = []
    ybeg_list = []
    num_block_x = int(np.ceil(limit[0] / stride)) + 1
    num_block_y = int(np.ceil(limit[1] / stride)) + 1
    for i in range(num_block_x):
        for j in range(num_block_y):
            xbeg_list.append(i * stride-(block_size-center_size)/2.0)
            ybeg_list.append(j * stride-(block_size-center_size)/2.0)

    # Collect blocks
    block_conds=[]
    block_beg_list=[]
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        xcond = (data[:, 0] <= xbeg + block_size) & (data[:, 0] >= xbeg)
        ycond = (data[:, 1] <= ybeg + block_size) & (data[:, 1] >= ybeg)
        cond = xcond & ycond
        if np.sum(cond) < 100:  # discard block if there are less than 100 pts.
            continue

        block_beg_list.append((xbeg,ybeg))
        block_conds.append(cond)

    return block_conds,block_beg_list

def points_downsample(data, sample_stride=0.1):
    min_coor=np.min(data[:,:3],axis=0,keepdims=True)
    data[:, :3]-=min_coor

    loc2pt={}
    for pt_index,pt in enumerate(data[:,:3]):
        x_index=int(math.ceil(pt[0]/sample_stride))
        y_index=int(math.ceil(pt[1]/sample_stride))
        z_index=int(math.ceil(pt[2]/sample_stride))
        loc=(x_index,y_index,z_index)
        if loc in loc2pt:
            loc2pt[loc].append(pt_index)
        else:
            loc2pt[loc]=[pt_index]

    downsample_points=[]
    indices=np.empty(data.shape[0],dtype=np.int32)
    for k,v in loc2pt.items():
        downsample_points.append(np.mean(data[v,:],axis=0,keepdims=True))
        indices[v]=len(downsample_points)-1

    data[:, :3]+=min_coor
    downsample_points=np.concatenate(downsample_points,axis=0)
    downsample_points[:,:3]+=min_coor
    return downsample_points,indices

def save_h5(h5_filename, data, label, data_dtype='float32', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

def save_all_room_h5():
    root_dir='../data/Stanford3dDataset_v1.2_Aligned_Version/'
    class_names=get_class_names()
    room_dirs=read_room_dirs()

    for dir_index,dir in enumerate(room_dirs):
        pcs,labels=read_room(root_dir+dir,class_names)
        save_h5("{}_{}.h5".format(dir_index,str(dir[:-12]).replace('/','_')),pcs,labels)
        print str(dir[:-12]).replace('/','_')

def get_train_test_split():
    import os
    path = os.path.split(os.path.realpath(__file__))[0]
    f = open(path + '/room_stems.txt', 'r')
    file_stems = [line.strip('\n') for line in f.readlines()]
    f.close()

    f = open(path + '/room_block_nums.txt', 'r')
    block_nums = [int(line.strip('\n')) for line in f.readlines()]
    f.close()

    # use area 5 as test
    train, test = [], []
    # train_nums, test_nums = [], []
    for fs, bn in zip(file_stems, block_nums):
        if fs.split('_')[2] == '5':
            test.append(fs)
            # test_nums.append(bn)
        else:
            train.append(fs)
            # train_nums.append(bn)

    return train, test #,train_nums, test_nums

def get_center_cond(context_raw_pts, context_beg, context_size, center_size):
    xcond= (context_raw_pts[:, 0] >= context_beg[0] + (context_size - center_size) / 2.0) & \
           (context_raw_pts[:, 0] <= context_beg[0] + (context_size - center_size) / 2.0 + center_size)
    ycond= (context_raw_pts[:, 1] >= context_beg[1] + (context_size - center_size) / 2.0) & \
           (context_raw_pts[:, 1] <= context_beg[1] + (context_size - center_size) / 2.0 + center_size)
    cond=xcond&ycond

    return cond


# room raw data to room context data
def room2context(filename='../data/S3DIS/room/260_Area_6_office_35.h5',
                 room_sample_interval=0.2,
                 context_sample_interval=0.1,
                 context_size=2.0,
                 center_size=1.0,
                 stride=1.0,
                 center_sample=True,
                 center_sample_point_num=4096,
                 has_fpfh=False,
                 fpfh_normal_radius=0.05,
                 fpfh_feature_radius=0.05,
                 skip_small_block=True,
                 skip_thresh=100
                 ):
    if has_fpfh:
        data,label,fpfhs=read_pkl(filename)
    else:
        data,label=read_room_h5(filename)

    data[:,:3]-=np.min(data[:,:3],axis=0,keepdims=True)
    room_downsample,room_indices=points_downsample(data, room_sample_interval)
    context_conds, context_beg_list=room2blocks_cond(data, block_size=context_size, stride=stride, center_size=center_size)

    block_list=[]
    for i in xrange(len(context_conds)):
        context_raw_data=data[context_conds[i]]
        cond=get_center_cond(context_raw_data,context_beg_list[i],context_size,center_size)

        # less than 100 points not considered
        if np.sum(cond)==0 or (skip_small_block and np.sum(cond)<skip_thresh):
            continue
        context_sample_data, context_sample_indices=points_downsample(context_raw_data, context_sample_interval)
        center_raw_data=context_raw_data[cond,:]

        block_dict={}
        if center_sample:
            center_sample_data, center_sample_indices=sample_data(center_raw_data, center_sample_point_num)
            block_dict['data']= center_sample_data
            block_dict['label']= label[context_conds[i]][cond][center_sample_indices]
            block_dict['cont_index']=context_sample_indices[cond][center_sample_indices]
            block_dict['room_index']= room_indices[context_conds[i]][cond][center_sample_indices]
            block_dict['cont']=context_sample_data
            if has_fpfh:
                block_dict['feat']=fpfhs[context_conds[i]][cond][center_sample_indices]
            else:
                context_raw_indices=np.arange(len(context_raw_data))
                center_raw_indices=context_raw_indices[cond]
                center_fpfh_indices=center_raw_indices[center_sample_indices]
                center_fpfh_indices=np.asarray(center_fpfh_indices,dtype=np.int64)
                # print np.max(center_fpfh_indices)
                # print len(context_raw_data)
                # print 'begin'
                center_fpfh=FPFHExtractor.extractFPFHIndices\
                    (context_raw_data[:,:3],center_fpfh_indices,fpfh_normal_radius,fpfh_feature_radius)
                # print 'here'
                block_dict['feat']=center_fpfh
                # print center_fpfh.shape
                # print center_sample_data.shape
        else:
            block_dict['data']= center_raw_data
            block_dict['label']= label[context_conds[i]][cond]
            block_dict['cont_index']=context_sample_indices[cond]
            block_dict['room_index']= room_indices[context_conds[i]][cond]
            block_dict['cont']=context_sample_data
            if has_fpfh:
                block_dict['feat']=fpfhs[context_conds[i]][cond]

        block_list.append(block_dict)

    return block_list,room_downsample

def one_file(fn_i,fn,f_stem):
    block_list,room_downsample=room2context(fn,stride=0.5)
    with open('../data/S3DIS/tmp/'+f_stem+'.pkl','w') as f:
        cPickle.dump([block_list,room_downsample],f)
    print '{} {} done'.format(fn_i,f_stem)

############for training set###############
def generate_trainset():
    room_files=[fn for fn in glob.glob(os.path.join('../data/S3DIS/room','*.h5'))]

    executor=ProcessPoolExecutor(max_workers=7)
    futures=[]
    for rf_i,rf in enumerate(room_files):
        rf_stem=os.path.basename(rf[:-3])
        futures.append(executor.submit(one_file,rf_i,rf,rf_stem))

    wait(futures)

def normalize_trainset():
    train_fs,test_fs,_,_=get_train_test_split()
    for rf_i,rf in enumerate(test_fs):
        block_list,room_sample_data=read_pkl('../data/S3DIS/tmp/' + rf + '.pkl')
        block_list, room_sample_data=normalize_v2(block_list,room_sample_data)
        save_pkl([block_list,room_sample_data],'../data/S3DIS/train_v2_more/'+rf+'.pkl')



# compute local feature (fpfh) for each block
def output_bnr_block_points(room_file_name,room_dir,room_context_dir,output_dir,):
    room_h5_file=room_dir+room_file_name+'.h5'
    room_context_ply_file=room_context_dir+room_file_name+'.pkl'

    with open(room_context_ply_file,'r') as f:
        block_list,room_downsample=cPickle.load(f)

    room_data,_=read_room_h5(room_h5_file)

    room_data=np.ascontiguousarray(room_data[:,:3],dtype=np.float32)

    print room_data.shape

    with open(output_dir+room_file_name+'.bnr','w') as f:
        f.write(struct.pack('I',room_data.shape[0]))                # room points number
        f.write(room_data.data)                                     # room points
        f.write(struct.pack('I',len(block_list)))                   # block number
        f.write(struct.pack('I',block_list[0]['data'].shape[0]))    # block points number
        for block in block_list:
            block_data=np.ascontiguousarray(block['data'][:,:3],dtype=np.float32) # block points
            f.write(block_data.data)

def compute_feats_all():
    room_files=[fn for fn in glob.glob(os.path.join('../data/S3DIS/room','*.h5'))]

    for rf_i,rf in enumerate(room_files):
        rf_stem = os.path.basename(rf[:-3])
        output_bnr_block_points(rf_stem,'../data/S3DIS/room/','../data/S3DIS/room_context/','./')
        os.system("/home/pal/project/PCDL/experiment/local_feature/cmake-build-release/local_feature {} {}".format(
            "./"+rf_stem+".bnr","./"+rf_stem+".feats"
        ))

def read_feats(feats_file):
    with open(feats_file,'r') as f:
        fpfh_size=struct.unpack('I',f.read(4))[0]
        fpfh_dims=struct.unpack('I',f.read(4))[0]
        feats=f.read(4*fpfh_size*fpfh_dims)

    feats=np.frombuffer(feats,dtype=np.float32,count=fpfh_size*fpfh_dims)
    feats=np.reshape(feats,[-1,fpfh_dims])

    return feats

def merge_local_feats_context(file_stem,output_dir,feats_dir,context_dir):
    feats=read_feats(feats_dir+file_stem+'.feats')
    block_list,room_downsample=read_pkl(context_dir + file_stem + '.pkl')

    feats=np.reshape(feats,[len(block_list),-1,feats.shape[1]])

    for feat,block in zip(feats,block_list):
        block['feat']=feat

    save_pkl([block_list,room_downsample],output_dir+file_stem+".pkl")
###########################################

def test_fpfh_kmeans():
    # feats=read_feats('268_Area_6_office_8.feats')

    block_list, room_downsample=read_pkl('../data/S3DIS/room_context_fpfh/268_Area_6_office_8.pkl')
    block_points=[block['data'] for block in block_list]
    feats=[block['feat'] for block in block_list]
    feats=np.concatenate(feats,axis=0)
    feats/=100
    block_points=np.concatenate(block_points,axis=0)

    from sklearn.cluster import KMeans

    cluster_num=10

    pred=KMeans(n_clusters=cluster_num,n_jobs=-1).fit_predict(feats)

    print pred.shape

    colors=np.random.randint(0,255,[cluster_num,3],dtype=np.int)
    for i in range(cluster_num):
        with open("{}.txt".format(i),'w') as f:
            for pt in block_points[pred==i]:
                f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],colors[i,0],colors[i,1],colors[i,2]))

def test_room2context(block_list,room_downsample):
    room_colors=np.random.randint(0,256,[room_downsample.shape[0],3],dtype=np.int)
    cls_colors = np.asarray(
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
         [205, 92, 92]], dtype=np.int
    )

    for i,block_dict in enumerate(block_list):
        print '/////////////////////'
        print block_dict['cont'].shape
        print block_dict['data'].shape

        with open('{}_data.txt'.format(i), 'w') as f:
            for pt in block_dict['data']:
                f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],pt[3],pt[4],pt[5]))

        with open('{}_cont.txt'.format(i), 'w') as f:
            for pt in block_dict['cont']:
                f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],pt[3],pt[4],pt[5]))

        cont_colors=np.random.randint(0,256,[block_dict['cont'].shape[0],3],dtype=np.int)
        with open('{}_cont_index.txt'.format(i), 'w') as f:
            for pt_i,pt in enumerate(block_dict['data']):
                f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],
                                                     cont_colors[block_dict['cont_index'][pt_i],0],
                                                     cont_colors[block_dict['cont_index'][pt_i],1],
                                                     cont_colors[block_dict['cont_index'][pt_i],2]))

        with open('{}_room_index.txt'.format(i), 'w') as f:
            for pt_i,pt in enumerate(block_dict['data']):
                f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],
                                                     room_colors[block_dict['room_index'][pt_i],0],
                                                     room_colors[block_dict['room_index'][pt_i],1],
                                                     room_colors[block_dict['room_index'][pt_i],2]))

        with open('{}_label.txt'.format(i), 'w') as f:
            for pt_i,pt in enumerate(block_dict['data']):
                f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],
                                                     cls_colors[block_dict['label'][pt_i,0],0],
                                                     cls_colors[block_dict['label'][pt_i,0],1],
                                                     cls_colors[block_dict['label'][pt_i,0],2]))

def test_downsample():
    room_h5_file='../data/S3DIS/room/260_Area_6_office_35.h5'
    f=h5py.File(room_h5_file,mode='r')
    data=f['data'][:]

    downsample_data,indices=points_downsample(data, 0.2)

    print downsample_data.shape

    colors=np.random.randint(0,256,[downsample_data.shape[0],3],dtype=np.int)

    with open('original.txt','w') as f:
        for pt_i,pt in enumerate(data):
            f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],
                                                 colors[indices[pt_i],0],
                                                 colors[indices[pt_i],1],
                                                 colors[indices[pt_i],2]))

    with open('downsample.txt','w') as f:
        for pt in downsample_data:
            f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],pt[3],pt[4],pt[5]))

#######normalize training set############
def normalize(pts):
    pts-=(np.max(pts,axis=0,keepdims=True)+np.min(pts,axis=0,keepdims=True))/2.0
    dist=pts[:,0]**2+pts[:,1]**2+pts[:,2]**2
    max_dist=np.sqrt(np.max(dist,axis=0,keepdims=True))
    pts/=max_dist
    return pts

def normalize_room(pts):
    pts-=np.min(pts,axis=0,keepdims=True)
    pts/=np.max(pts,axis=0,keepdims=True)

    return pts

def normalize_context(pts,block_size):
    pts-=np.min(pts,axis=0,keepdims=True)
    pts[:, 0] -= block_size / 2.0
    pts[:, 1] -= block_size / 2.0

    return pts


def normalize_v2(block_list, global_pts):
    for block in block_list:
        block['cont'][:, :3]=normalize_context(block['cont'][:,:3],2.0)
        block['cont'][:, 3:]/=255.0
        block['feat']/=100

    global_pts[:, :3]=normalize_room(global_pts[:,:3])
    global_pts[:, 3:]/=255.0

    return block_list,global_pts


def normalize_all():
    import time
    f=open('room_stems.txt','r')
    file_stems=[line.strip('\n') for line in f.readlines()]
    f.close()

    begin=time.time()
    for fsi,fs in enumerate(file_stems):
        block_list,global_pts=read_pkl('../data/S3DIS/room_context_fpfh/' + fs + '.pkl', 'rb')
        normalize_v2(block_list,global_pts)
        save_pkl((block_list,global_pts),'../data/S3DIS/train_v2/'+fs+'.pkl')
        print fsi


    print 'cost {} s'.format(time.time()-begin)

def test_normalize():
    import matplotlib.pyplot as plt
    block_list,global_pts=read_pkl('../data/S3DIS/train_v2/107_Area_4_conferenceRoom_1.pkl')

    print 'global: min {}\n max {}\n mean {}'.format(np.min(global_pts,axis=0),np.max(global_pts,axis=0),np.mean(global_pts,axis=0))
    context_pts=[block['cont'] for block in block_list]
    context_pts=np.concatenate(context_pts,axis=0)
    print context_pts.shape
    print 'context: min {}\n max {}\n mean {}'.format(np.min(context_pts,axis=0),np.max(context_pts,axis=0),np.mean(context_pts,axis=0))
    feats=[block['feat'] for block in block_list]
    feats=np.concatenate(feats,axis=0)
    print 'feats: min {}\n max {}\n mean {}'.format(np.min(feats,axis=0),np.max(feats,axis=0),np.mean(feats,axis=0))
    labels=[block['label'] for block in block_list]
    labels=np.concatenate(labels,axis=0)
    print 'labels: min {}\n max {}\n mean {}'.format(np.min(labels,axis=0),np.max(labels,axis=0),np.mean(labels,axis=0))

    plt.hist(labels)
    plt.show()
##########################################

###########testset#######################
def extract_fpfh_testset():

    import time
    train_fs,test_fs,_,_=get_train_test_split()
    print len(test_fs)
    begin=time.time()
    for rf_i,rf in enumerate(test_fs):
        data,label=read_room_h5('../data/S3DIS/room/'+rf+'.h5')
        fpfhs=FPFHExtractor.extractFPFH(data[:,:3],0.05,0.05)
        save_pkl([data,label,fpfhs],'../data/S3DIS/tmp/'+rf+'.pkl')
        print '{} cost {} s'.format(rf_i,time.time()-begin)
        begin=time.time()

def generate_testset():
    train_fs,test_fs,_,_=get_train_test_split()
    for rf_i,rf in enumerate(test_fs):
        block_list,room_sample_data=room2context('../data/S3DIS/testset_raw/' + rf + '.pkl',
                                                 has_fpfh=True,center_sample=False,skip_small_block=False)
        save_pkl([block_list,room_sample_data],'../data/S3DIS/testset_context/' + rf + '.pkl')

def normalize_testset():
    train_fs,test_fs,_,_=get_train_test_split()
    for rf_i,rf in enumerate(test_fs):
        block_list,room_sample_data=read_pkl('../data/S3DIS/testset_context/' + rf + '.pkl')
        block_list, room_sample_data=normalize_v2(block_list,room_sample_data)
        save_pkl([block_list,room_sample_data],'../data/S3DIS/test_v2/'+rf+'.pkl')

def test_feats_testset():
    from sklearn.cluster import KMeans
    train_fs,test_fs,_,_=get_train_test_split()
    for rf_i,rf in enumerate(test_fs):
        data,label,fpfhs=read_pkl('../data/S3DIS/testset_raw/' + rf + '.pkl')
        indices=np.random.randint(0,len(data),[102400])

        block_points=data[indices]
        feats=fpfhs[indices]
        cluster_num=5

        pred=KMeans(n_clusters=cluster_num,n_jobs=-1).fit_predict(feats)

        print pred.shape

        colors=np.random.randint(0,255,[cluster_num,3],dtype=np.int)
        for i in range(cluster_num):
            with open("{}.txt".format(i),'w') as f:
                for pt in block_points[pred==i]:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],colors[i,0],colors[i,1],colors[i,2]))
        break

if __name__=="__main__":
    # normalize_all()
    # test_normalize()
    # test_fpfh_kmeans()
    # generate_testset()
    # generate_testset()
    # normalize_testset()
    #
    # block_list,room_data=room2context(context_size=2.0,stride=0.5,center_size=1.0)
    # test_room2context(block_list,room_data)
    # normalize_trainset()


    # train_fs,test_fs=get_train_test_split()
    # train_fs+=test_fs
    # counts=np.zeros(13)
    # for rf_i,rf in enumerate(train_fs):
    #     block_list,room_sample_data=read_pkl('../data/S3DIS/train_v2_nostairs/' + rf + '.pkl')
    #     for block in block_list:
    #         for l in range(13):
    #             counts[l]+=np.sum(block['label']==l)
    #
    # for c in counts:
    #     print str(int(c))+','


    import matplotlib.pyplot as plt

    counts=[26216625,23867455,26435178,2153315,1671711,1443172,6487113,3774739,4461522,465457,5305832,1053401,11393440]
    names=get_class_names()
    plt.bar(names[:13],counts)
    plt.show()