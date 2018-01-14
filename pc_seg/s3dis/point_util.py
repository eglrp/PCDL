from data_util import *
import FPFHExtractor
from concurrent.futures import ProcessPoolExecutor,wait
import pyflann


def extract_FPFH(points,indices=None,normal_radius=0.05,feature_radius=0.1):
    assert points.shape[1]>=3
    if indices is None:
        return FPFHExtractor.extractFPFH(points[:,:3],normal_radius,feature_radius)
    else:
        indices=np.asarray(indices,dtype=np.int64)
        return FPFHExtractor.extractFPFHIndices(points[:,:3],indices,normal_radius,feature_radius)


def save_points_feats(filename,feats,labels):
    h5_fout = h5py.File(filename,'w')
    h5_fout.create_dataset(
            'feat', data=feats,
            compression='gzip', compression_opts=4,
            dtype='float32')
    h5_fout.create_dataset(
            'label', data=labels,
            compression='gzip', compression_opts=1,
            dtype='uint8')
    h5_fout.close()


def read_points_feats(filename):
    h5_fin = h5py.File(filename,'r')
    feats,labels= h5_fin['feat'][:],h5_fin['label'][:]
    h5_fin.close()
    return feats,labels


def compute_normal(points,k=5):
    assert k>=5
    flann=pyflann.FLANN()
    flann.build_index(points,algorithm='kdtree_simple',leaf_max_size=3)
    normals=[]
    eigvals=[]
    indices=[]
    mdists=[]
    for pt in points:
        nidxs,ndists=flann.nn_index(pt,k+1)
        npts=points[nidxs[0,:],:]
        # compute normal
        mean=np.mean(npts,axis=0,keepdims=True)
        var=(npts-mean).transpose().dot(npts-mean)
        eigval,eigvec=np.linalg.eigh(var)
        normals.append(np.expand_dims(eigvec[0],axis=0))
        eigvals.append(np.expand_dims(eigval,axis=0))
        # only use 5 points
        indices.append(nidxs[:,1:6])
        mdists.append(np.mean(np.sqrt(ndists[0,1:6]),axis=0))

    normals=np.concatenate(normals,axis=0)
    eigvals=np.concatenate(eigvals,axis=0)
    indices=np.concatenate(indices,axis=0)
    mdists=np.stack(mdists,axis=0)
    # flip normals
    masks=np.sum(normals*points,axis=1)<0
    normals[masks]=-normals[masks]

    return normals,eigvals,indices,mdists


def prepare_point_dataset_impl(room_path_prefix,save_path_prefix,fn,
                               downsample_stride,normal_radius,feature_radius):
        # read room data
        points,label=read_room_h5(room_path_prefix+fn+'.h5')
        # print points.shape
        # downsample
        ds_points,ds_labels,ds_idxs=downsample_random(points,label,downsample_stride)
        # print ds_points.shape,ds_idxs.shape,ds_labels.shape

        # room loc
        room_min=np.min(points[:,:3],axis=0,keepdims=True)
        points[:,:3]=points[:,:3]-room_min
        ds_points[:,:3]=ds_points[:,:3]-room_min
        room_max = np.max(points[:, :3], axis=0, keepdims=True)
        ds_room_loc=ds_points[:,:3]/room_max

        # normalize color
        ds_color=ds_points[:,3:]/255.0

        # feature extract
        ds_fpfh=extract_FPFH(points,ds_idxs,normal_radius,feature_radius)
        # print ds_fpfh.shape
        ds_fpfh/=np.max(ds_fpfh,axis=1,keepdims=True)+1e-5

        # write
        ds_feats=np.concatenate([ds_room_loc,ds_color,ds_fpfh],axis=1)
        save_points_feats(save_path_prefix+fn+'.h5',ds_feats,ds_labels)

        print '{] done'.format(fn)


def prepare_point_dataset(downsample_stride=0.05,normal_radius=0.05,feature_radius=0.1):
    trainset,testset=get_train_test_split()
    trainset+=testset
    room_path_prefix='../data/S3DIS/room/'
    save_path_prefix='../data/S3DIS/point/fpfh/'
    executor=ProcessPoolExecutor(8)
    futures=[]
    for fn in trainset:
        futures.append(executor.submit(prepare_point_dataset_impl,room_path_prefix,save_path_prefix,fn,
                                       downsample_stride,normal_radius,feature_radius))
    wait(futures)


def stairs2clutter():
    path='../data/S3DIS/point/fpfh/'
    train_fs,test_fs=get_train_test_split()
    train_fs+=test_fs
    for fs in train_fs:
        feats,labels=read_points_feats(path+fs+'.h5')
        labels[labels==13]=12
        save_points_feats(path+fs+'.h5',feats,labels)


def prepare_input_list(path,batch_size):
    train_fs,test_fs=get_train_test_split()
    train_input_list=[]
    for fs in train_fs:
        _,labels=read_points_feats(path+fs+'.h5')
        num=labels.shape[0]
        batch_num=int(math.ceil(num/float(batch_size)))
        train_input_list+=[(path+fs+'.h5',i) for i in xrange(batch_num)]
    test_input_list=[]

    for fs in test_fs:
        _,labels=read_points_feats(path+fs+'.h5')
        num=labels.shape[0]
        batch_num=int(math.ceil(num/float(batch_size)))
        test_input_list+=[(path+fs+'.h5',i) for i in xrange(batch_num)]

    return train_input_list,test_input_list


def fetch_data(model,filename,index,batch_size):
    feats,labels=read_points_feats(filename)
    beg_idx=index*batch_size
    end_idx=min((1+index) * batch_size,feats.shape[0])
    cur_size=end_idx-beg_idx
    data=feats[beg_idx:end_idx]
    label=labels[beg_idx:end_idx]
    if model=='train' and cur_size<batch_size:
        ridx=np.random.randint(0,feats.shape[0],batch_size-cur_size)
        data=np.concatenate([data,feats[ridx,:]],axis=0)
        label=np.concatenate([label,labels[ridx]],axis=0)

    return data,label


def fetch_batch(model,batch):
    return np.asarray(batch[0][0],dtype=np.float32),np.asarray(batch[0][1],dtype=np.int64)


def test_fpfh():
    import draw_util
    from sklearn.cluster import KMeans
    feats,labels=read_points_feats('/home/pal/project/PCDL/pc_seg/data/S3DIS/point/fpfh/0_Area_1_conferenceRoom_1.h5')
    print np.max(feats,axis=0)
    print np.min(feats,axis=0)
    print np.mean(feats,axis=0)
    kmeans=KMeans(n_clusters=5,n_jobs=-1)
    preds=kmeans.fit_predict(feats[:,6:])
    for i in range(5):
        mask=preds==i
        color=np.random.randint(0,255,3)
        draw_util.output_points("kmeans{}.txt".format(i),feats[mask,0:3],color)

    colors=get_class_colors()
    draw_util.output_points("labels.txt",feats[:,:3],colors[labels,:])


def test_point_provider():
    import sys
    sys.path.append('../')
    import provider
    import functools
    import time
    train_list, test_list = prepare_input_list('../data/S3DIS/point/fpfh/', 256)
    fetch_data_with_batch = functools.partial(fetch_data, batch_size=256)
    train_provider = provider.Provider(train_list, 1, fetch_data_with_batch, 'train', 4, fetch_batch, max_worker_num=1)

    print train_provider.batch_num

    begin = time.time()
    for feats, labels in train_provider:
        print feats.shape, labels.shape
        print time.time() - begin
        begin = time.time()


if __name__=="__main__":
    # test_fpfh()
    # stairs2clutter()
    print get_train_test_split()[0]
    print get_train_test_split()[1]