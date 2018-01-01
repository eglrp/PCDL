import glob
import numpy as np
import os
import h5py
import math
import cPickle
import struct

def read_class_names():
    names=[]
    with open('class_names.txt','r') as f:
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

def read_room_context(room_context_file,model='rb'):
    with open(room_context_file,model) as f:
        block_list,room_downsample=cPickle.load(f)
    return block_list,room_downsample

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
    class_names=read_class_names()
    room_dirs=read_room_dirs()

    for dir_index,dir in enumerate(room_dirs):
        pcs,labels=read_room(root_dir+dir,class_names)
        save_h5("{}_{}.h5".format(dir_index,str(dir[:-12]).replace('/','_')),pcs,labels)
        print str(dir[:-12]).replace('/','_')

# room raw data to room context data
def room2context(room_f5_file='../data/S3DIS/room/260_Area_6_office_35.h5',
                 room_downsample_interval=0.2,
                 block_context_downsample_interval=0.1,
                 block_center_point_num=4096,
                 block_size=2.0,
                 stride=1.0):
    data,label=read_room_h5(room_f5_file)

    data[:,:3]-=np.min(data[:,:3],axis=0,keepdims=True)
    room_downsample,room_indices=points_downsample(data,room_downsample_interval)
    block_data_list, block_label_list, block_indices_list, block_beg_list=\
        room2blocks_with_indices(data, label, room_indices, block_size=block_size, stride=stride)

    block_list=[]
    for i in xrange(len(block_data_list)):
        xcond= (block_data_list[i][:, 0] >= block_beg_list[i][0] + (block_size-stride)/2.0) & \
               (block_data_list[i][:, 0] <= block_beg_list[i][0] + (block_size-stride)/2.0+1.0)
        ycond= (block_data_list[i][:, 1] >= block_beg_list[i][1] + (block_size-stride)/2.0) & \
               (block_data_list[i][:, 1] <= block_beg_list[i][1] + (block_size-stride)/2.0+1.0)
        cond=xcond&ycond

        # less than 100 points not considered
        if np.sum(cond)<100:
            continue
        block_context_downsample, block_context_indices=points_downsample(block_data_list[i], block_context_downsample_interval)
        block_sample_data,block_sample_indices=sample_data(block_data_list[i][cond, :],block_center_point_num)

        block_dict={}
        block_dict['data']= block_sample_data
        block_dict['label']= block_label_list[i][cond, :][block_sample_indices]
        block_dict['cont_index']=block_context_indices[cond][block_sample_indices]
        block_dict['room_index']= block_indices_list[i][cond][block_sample_indices]
        block_dict['cont']=block_context_downsample

        block_list.append(block_dict)

    return block_list,room_downsample


def room2context_without_sample(room_f5_file='../data/S3DIS/room/260_Area_6_office_35.h5',
                                 room_downsample_interval=0.2,
                                 block_context_downsample_interval=0.1,
                                 block_size=2.0,
                                 stride=1.0):
    data,label=read_room_h5(room_f5_file)

    data[:,:3]-=np.min(data[:,:3],axis=0,keepdims=True)
    room_downsample,room_indices=points_downsample(data,room_downsample_interval)
    block_data_list, block_label_list, block_indices_list, block_beg_list=\
        room2blocks_with_indices(data, label, room_indices, block_size=block_size, stride=stride)

    block_list=[]
    for i in xrange(len(block_data_list)):
        xcond= (block_data_list[i][:, 0] >= block_beg_list[i][0] + (block_size-stride)/2.0) & \
               (block_data_list[i][:, 0] <= block_beg_list[i][0] + (block_size-stride)/2.0+1.0)
        ycond= (block_data_list[i][:, 1] >= block_beg_list[i][1] + (block_size-stride)/2.0) & \
               (block_data_list[i][:, 1] <= block_beg_list[i][1] + (block_size-stride)/2.0+1.0)
        cond=xcond&ycond

        # less than 100 points not considered
        if np.sum(cond)<100:
            continue
        block_context_downsample, block_context_indices=points_downsample(block_data_list[i], block_context_downsample_interval)
        block_sample_data,block_sample_indices=sample_data(block_data_list[i][cond, :],block_center_point_num)

        block_dict={}
        block_dict['data']= block_sample_data
        block_dict['label']= block_label_list[i][cond, :][block_sample_indices]
        block_dict['cont_index']=block_context_indices[cond][block_sample_indices]
        block_dict['room_index']= block_indices_list[i][cond][block_sample_indices]
        block_dict['cont']=block_context_downsample

        block_list.append(block_dict)

    return block_list,room_downsample

def room2context_all():
    room_files=[fn for fn in glob.glob(os.path.join('../data/S3DIS/room','*.h5'))]
    for rf_i,rf in enumerate(room_files):
        rf_stem=os.path.basename(rf[:-3])
        block_list,room_downsample=room2context(rf)
        with open(rf_stem+'.pkl','w') as f:
            cPickle.dump([block_list,room_downsample],f)
        print '{} {} done'.format(rf_i,rf_stem)

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
    block_list,room_downsample=read_room_context(context_dir+file_stem+'.pkl')

    feats=np.reshape(feats,[len(block_list),-1,feats.shape[1]])

    for feat,block in zip(feats,block_list):
        block['feat']=feat

    save_pkl([block_list,room_downsample],output_dir+file_stem+".pkl")

def test_fpfh_kmeans():
    feats=read_feats('268_Area_6_office_8.feats')

    block_list, room_downsample=read_room_context('../data/S3DIS/room_context/268_Area_6_office_8.pkl')
    block_points=[block['data'] for block in block_list]
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

def normalize(pts):
    pts-=(np.max(pts,axis=0,keepdims=True)+np.min(pts,axis=0,keepdims=True))/2.0
    dist=pts[:,0]**2+pts[:,1]**2+pts[:,2]**2
    max_dist=np.sqrt(np.max(dist,axis=0,keepdims=True))
    pts/=max_dist
    return pts

def normalize_all():
    import time
    f=open('room_stems.txt','r')
    file_stems=[line.strip('\n') for line in f.readlines()]
    f.close()

    begin=time.time()
    for fs in file_stems:
        block_list,global_pts=read_room_context('../data/S3DIS/room_context_fpfh/'+fs+'.pkl','rb')
        for block in block_list:
            block['cont'][:, :3]=normalize(block['cont'][:,:3])
            block['cont'][:, 3:]-=128.0
            block['cont'][:, 3:]/=128.0
            block['feat']-=50
            block['feat']/=50
            # print np.max(block['feat'],axis=0),np.min(block['feat'],axis=0)

        global_pts[:, :3]=normalize(global_pts[:,:3])
        global_pts[:, 3:]-=128.0
        global_pts[:, 3:]/=128.0
        save_pkl((block_list,global_pts),'../data/S3DIS/train/'+fs+'.pkl')


    print 'cost {} s'.format(time.time()-begin)

def get_train_test_split():
    import os
    path=os.path.split(os.path.realpath(__file__))[0]
    f=open(path+'/room_stems.txt','r')
    file_stems=[line.strip('\n') for line in f.readlines()]
    f.close()

    f=open(path+'/room_block_nums.txt','r')
    block_nums=[int(line.strip('\n')) for line in f.readlines()]
    f.close()

    # use area 5 as test
    train,test=[],[]
    train_nums,test_nums=[],[]
    for fs,bn in zip(file_stems,block_nums):
        if fs.split('_')[2]=='5':
            test.append(fs)
            test_nums.append(bn)
        else:
            train.append(fs)
            train_nums.append(bn)

    return train,test,train_nums,test_nums



if __name__=="__main__":
    print len(get_train_test_split()[0])
    print len(get_train_test_split()[1])