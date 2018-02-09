import tensorflow as tf
from s3dis.data_util import room2blocks
from s3dis.voxel_util import points2covars_gpu
import numpy as np

# 2. compute covariances
# 3. predict each voxel
# 4. concat back

def predict(room_points,room_labels):
    room_points-=np.min(room_points,axis=0,keepdims=True)
    # 1. room2block
    block_points_list, block_labels_list, _ = room2blocks(room_points,room_labels,without_sample=True)
    # nearest 0.1 meter using flann
    block_nidxs_list=[]
    block_covars_list=[]
    for i in range(len(block_points_list)):
        block_covars_list.append(points2covars_gpu(block_nidxs_list[i],block_nidxs_list[i],16))
    # normalize
    original_points_list=block_points_list[:]
    for k in xrange(len(block_points_list)):
        block_points_list[k][:,:3]/=


