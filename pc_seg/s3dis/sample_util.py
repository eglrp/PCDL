import numpy as np
import PointsUtil

def get_list(maxx,block_size,stride,resample_ratio=0.03):
    x_list=[]
    spacex=maxx-block_size
    if spacex<0: x_list.append(0)
    else:
        x_list+=list(np.arange(0,spacex,stride))
        if (spacex-int(spacex/stride)*stride)/block_size>resample_ratio:
            x_list+=list(np.arange(spacex,0,-stride))

    return x_list


def get_list_without_back_sample(maxx,block_size,stride):
    x_list=[]
    spacex=maxx-block_size
    if spacex<0: x_list.append(0)
    else:
        x_list+=list(np.arange(0,spacex,stride))
        x_list.append(spacex)

    return x_list


def uniform_sample_block(points,labels,block_size=4.0,stride=1.0,normalized=False):
    assert stride<=block_size

    if not normalized:
        points[:,:3] -= np.min(points[:,:3],axis=0,keepdims=True)

    # uniform sample
    max_xyz=np.max(points[:,:3],axis=0,keepdims=True)
    maxx,maxy=max_xyz[0,0],max_xyz[0,1]
    beg_list=[]
    x_list=get_list_without_back_sample(maxx,block_size,stride)
    y_list=get_list_without_back_sample(maxy,block_size,stride)
    for x in x_list:
        for y in y_list:
            beg_list.append((x,y))

    block_points_list,block_labels_list=[],[]
    for beg in beg_list:
        x_cond=(points[:,0]>=beg[0])&(points[:,0]<beg[0]+block_size)
        y_cond=(points[:,1]>=beg[1])&(points[:,1]<beg[1]+block_size)
        cond=x_cond&y_cond
        if(np.sum(cond)<500):
            continue

        block_points_list.append(points[cond])
        block_labels_list.append(labels[cond])

    return block_points_list,block_labels_list


def flip(points,axis=0):
    result_points=points[:]
    result_points[:,axis]=-result_points[:,axis]
    return result_points


def swap_xy(points):
    result_points = np.empty_like(points, dtype=np.float32)
    result_points[:,0]=points[:,1]
    result_points[:,1]=points[:,0]
    result_points[:,2:]=points[:,2:]

    return result_points


def random_rotate_sample_block(points,labels,block_size=4.0,stride=1.0,retain_ratio=0.8,split_num=10,normalized=False,gpu_index=0):
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    points = np.ascontiguousarray(points, dtype=np.float32)

    if not normalized:
        points[:, :3] -= np.min(points[:, :3], axis=0, keepdims=True)

    max_coor = np.max(points[:, :2], axis=0)
    maxx, maxy = max_coor[0], max_coor[1]

    block_points_list, block_labels_list = PointsUtil.UniformSampleBlock\
        (points, labels, stride, block_size, retain_ratio, split_num, maxx, maxy)

    return block_points_list,block_labels_list


def downsample_random_gpu(points,labels,sample_stride=0.01):
    points[:,:3]-=np.min(points[:,:3],axis=0,keepdims=True)
    points = np.ascontiguousarray(points, dtype=np.float32)
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    ds_points,ds_labels=PointsUtil.GridDownSample(points,labels,sample_stride)
    return ds_points,ds_labels


def split_dataset():
    from data_util import read_room_h5,get_train_test_split,save_room_pkl
    from draw_util import output_points
    import time

    train_list, test_list = get_train_test_split()
    train_list += test_list

    f=open('room_block_stems.txt','w')

    idx=0
    for fn in train_list:
        begin=time.time()
        points,labels=read_room_h5('../data/S3DIS/room/'+fn+'.h5')

        labels[labels==13]=12
        block_points_list, block_labels_list = uniform_sample_block(points, labels, block_size=10, stride=8)

        # print
        block_name=str.join('_', fn.split('_')[1:])
        for pts,lbs in zip(block_points_list,block_labels_list):
            # output_points('test_result/'+str(idx)+'_'+block_name+'.txt',pts)
            output_name='{}_{}.pkl'.format(idx,block_name)
            f.write(output_name+'\n')
            save_room_pkl('test_result/'+output_name,pts,lbs)
            idx+=1

        print 'cost {} s'.format(time.time()-begin)

    f.close()


def test_uniform_sample():
    from data_util import read_room_pkl,get_block_train_test_split,save_room_pkl
    from draw_util import output_points
    import time

    train_list, test_list = get_block_train_test_split()
    train_list += test_list

    for fn in train_list:
        begin=time.time()
        points,labels=read_room_pkl('../data/S3DIS/room_block_10_10/'+fn)

        labels[labels==13]=12
        block_points_list, block_labels_list = uniform_sample_block(points, labels, block_size=3.0, stride=1.5)

        # print
        # block_name=str.join('_', fn.split('_')[1:])
        # for pts,lbs in zip(block_points_list,block_labels_list):
        #     # output_points('test_result/'+str(idx)+'_'+block_name+'.txt',pts)
        #     output_name='{}_{}.pkl'.format(idx,block_name)
        #     save_room_pkl('test_result/'+output_name,pts,lbs)
        #     idx+=1

        print 'cost {} s'.format(time.time()-begin)


def test_random_rotate_sample():
    from data_util import read_room_pkl,get_block_train_test_split,save_room_pkl
    from draw_util import output_points
    import time

    train_list, test_list = get_block_train_test_split()
    train_list += test_list

    idx=0
    for fn in train_list[:1]:
        begin=time.time()
        points,labels=read_room_pkl('../data/S3DIS/room_block_10_10/'+'153_Area_4_hallway_3.pkl')
        # output_points('test.txt',points)

        labels[labels==13]=12
        block_points_list, block_labels_list = random_rotate_sample_block(points, labels, block_size=3.0, stride=1.5)
        for block_points in block_points_list:
            # print block_points.shape
            print fn
            assert block_points.shape[0]!=0


        fs=fn.split('.')[0]
        for pts,lbs in zip(block_points_list,block_labels_list):
            output_points('test_result/'+fs+'_'+str(idx)+'.txt',pts)
            idx+=1

        print 'cost {} s'.format(time.time() - begin)




if __name__=="__main__":
    # from data_util import read_room_pkl
    # points,labels=read_room_pkl('/home/pal/project/PCDL/pc_seg/data/S3DIS/room_block_10_10/34_Area_1_office_24.pkl')
    # print points.shape
    # print labels.shape
    test_random_rotate_sample()

