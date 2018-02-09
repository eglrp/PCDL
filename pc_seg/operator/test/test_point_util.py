import PointsUtil
import sys
sys.path.append('..')
from s3dis.data_util import read_room_pkl,get_block_train_test_split,get_class_colors
from s3dis.draw_util import output_points
import time
import numpy as np

def test_random_rotate_sample_block():
    train_list,test_list=get_block_train_test_split()
    import random
    random.shuffle(train_list)
    for fn in train_list[:1]:
        points,labels=read_room_pkl('../data/S3DIS/room_block_10_10/{}'.format(fn))

        labels=np.asarray(labels,dtype=np.int32)
        points=np.ascontiguousarray(points,dtype=np.float32)
        points[:,:2]-=np.min(points[:,:2],axis=0,keepdims=True)
        max_coor=np.max(points[:,:2],axis=0)
        maxx,maxy=max_coor[0],max_coor[1]
        begin = time.time()
        block_points_list,block_labels_list=PointsUtil.UniformSampleBlock(points,labels,1.0,5.0,0.8,10,maxx,maxy)

        print 'cost {} s'.format(time.time()-begin)
        for i,pts in enumerate(block_points_list):
            output_points('test/{}.txt'.format(i),pts)

if __name__=="__main__":
    train_list,test_list=get_block_train_test_split()
    import random
    random.shuffle(train_list)
    for fn in train_list[:1]:
        points,labels=read_room_pkl('../data/S3DIS/room_block_10_10/'+fn)
        points=np.ascontiguousarray(points,dtype=np.float32)
        labels=np.ascontiguousarray(labels,dtype=np.int32)
        points[:,:3]-=np.min(points[:,:3],axis=0,keepdims=True)
        output_points('original.txt', points)
        begin = time.time()
        points,labels=PointsUtil.GridDownSample(points,labels,0.05)
        print 'cost {} s'.format(time.time()-begin)

        colors=get_class_colors()

        output_points('downsample.txt', points)





