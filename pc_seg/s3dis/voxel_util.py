import math
import PointsUtil
import numpy as np


def neighbor_idxs(x,y,z,order=2,):
    xs=np.asarray([x-i for i in xrange(-order,order+1)],dtype=np.int32)
    ys=np.asarray([y-i for i in xrange(-order,order+1)],dtype=np.int32)
    zs=np.asarray([z-i for i in xrange(-order,order+1)],dtype=np.int32)
    X,Y,Z=np.meshgrid(xs,ys,zs)
    coords=np.concatenate([X[:,:,:,None],Y[:,:,:,None],Z[:,:,:,None]],axis=3)
    coords=np.reshape(coords,[-1,3])

    return coords


def validate_idx(x,y,z,max_num):
    if 0<=x<max_num and 0<=y<max_num and 0<=z<max_num:
        return True
    else:
        return False


def compute_dist(pt, idx, stride):
    return ((pt[0] - (idx[0]+0.5)*stride) ** 2 +
             (pt[1] - (idx[1]+0.5)*stride) ** 2 +
             (pt[2] - (idx[2]+0.5)*stride) ** 2)


def point2voxel(points,split_num):
    '''
    :param points:      k,3 already normalize to (0,1)^3 cubic
    :param split_num:
    :return:
    '''
    stride=1.0/split_num

    # flann=pyflann.FLANN()
    # flann.build_index(points,algorithm='kdtree_simple',leaf_max_size=3)
    voxel_state=np.zeros([split_num+1,split_num+1,split_num+1],dtype=np.float32)
    split_num+=1

    for pt in points:

        x_index=int(math.floor(pt[0] / stride))
        y_index=int(math.floor(pt[1] / stride))
        z_index=int(math.floor(pt[2] / stride))

        x_ratio=1-abs(x_index*stride-pt[0])/stride
        y_ratio=1-abs(y_index*stride-pt[1])/stride
        z_ratio=1-abs(z_index*stride-pt[2])/stride

        if validate_idx(x_index,y_index,z_index,split_num):
            voxel_state[x_index,y_index,z_index]+=x_ratio*y_ratio*z_ratio

        if validate_idx(x_index+1,y_index,z_index,split_num):
            voxel_state[x_index+1,y_index,z_index]+=(1-x_ratio)*y_ratio*z_ratio

        if validate_idx(x_index,y_index+1,z_index,split_num):
            voxel_state[x_index,y_index+1,z_index]+=x_ratio*(1-y_ratio)*z_ratio

        if validate_idx(x_index,y_index,z_index+1,split_num):
            voxel_state[x_index,y_index,z_index+1]+=x_ratio*y_ratio*(1-z_ratio)

        if validate_idx(x_index+1,y_index+1,z_index,split_num):
            voxel_state[x_index+1,y_index+1,z_index]+=(1-x_ratio)*(1-y_ratio)*z_ratio

        if validate_idx(x_index,y_index+1,z_index+1,split_num):
            voxel_state[x_index,y_index+1,z_index+1]+=x_ratio*(1-y_ratio)*(1-z_ratio)

        if validate_idx(x_index+1,y_index,z_index+1,split_num):
            voxel_state[x_index+1,y_index,z_index+1]+=(1-x_ratio)*y_ratio*(1-z_ratio)

        if validate_idx(x_index+1,y_index+1,z_index+1,split_num):
            voxel_state[x_index+1,y_index+1,z_index+1]+=(1-x_ratio)*(1-y_ratio)*(1-z_ratio)

    voxel_state[voxel_state>1.0]=1.0

    voxel_state=np.asarray(voxel_state.flatten())
    return voxel_state


def points2voxel_gpu_s3dis(points, split_num, gpu_index=0):
    '''
    :param points: n,k,3
    :param split_num:
    :return:
    '''
    pts = np.ascontiguousarray(np.copy(points[:, :, :3]))
    pts[:, :, :2] += 0.5
    voxels = PointsUtil.Points2VoxelBatchGPU(pts, split_num, gpu_index)
    voxels[voxels > 1.0] = 1.0

    return voxels

def points2voxel_gpu_modelnet(points,split_num,gpu_index=0):
    '''
    :param points: n,k,3
    :param split_num:
    :return:
    '''
    pts = np.ascontiguousarray(np.copy(points[:, :, :3]))
    pts+=1.0
    pts/=2.0
    voxels = PointsUtil.Points2VoxelBatchGPU(pts, split_num, gpu_index)
    voxels[voxels > 1.0] = 1.0

    return voxels

def points2voxel_color_gpu(points,split_num,gpu_index=0):
    '''
    :param points:      n,k,6 xyz:(-0.5,0.5)(-0.5,0.5)(0.0,1.0) rgb:(-1.0,1.0)*3
    :param split_num:
    :param gpu_index:
    :return:
    '''
    pts = np.ascontiguousarray(np.copy(points[:, :, :6]))
    pts[:, :, :2] += 0.5
    pts[:, :, 3:] += 1.0
    pts[:, :, 3:] /= 2.0
    voxels = PointsUtil.Points2VoxeColorlBatchGPU(pts, split_num, gpu_index)
    voxel_state,voxel_color=voxels[:,:,0],voxels[:,:,1:]
    voxel_color/=voxel_state[:,:,None]+1e-6
    voxel_state[voxel_state>1.0]=1.0
    voxel_color[voxel_state>1.0]=1.0

    return voxel_state,voxel_color


def voxel2points(voxel):
    '''
    :param voxel:
    :return:
    '''
    if len(voxel.shape)==1:
        side_num=int(round(pow(voxel.shape[0], 1 / 3.0)))
        cubic_state=np.reshape(voxel, [side_num, side_num, side_num])
        pts=[]
        for i in xrange(side_num):
            for j in xrange(side_num):
                for k in xrange(side_num):
                    color_val=int(255 * cubic_state[i, j, k])
                    coord_val=np.asarray([i,j,k,color_val,color_val,color_val],dtype=np.float32)
                    pts.append(coord_val)

    else:
        side_num=int(round(pow(voxel.shape[0], 1 / 3.0)))
        cubic_state=np.reshape(voxel, [side_num, side_num, side_num, 3])
        pts=[]
        for i in xrange(side_num):
            for j in xrange(side_num):
                for k in xrange(side_num):
                    color_val=np.asarray(255 * cubic_state[i, j, k, :], np.int32)
                    color_val[color_val>255]=255
                    color_val[color_val<0]=0

                    coord_val=np.asarray([i,j,k,color_val[0],color_val[1],color_val[2]],dtype=np.float32)
                    pts.append(coord_val)

    pts=np.asarray(pts)
    pts[:,:3]/=np.max(pts[:,:3],axis=0,keepdims=True)
    return pts


def points2covars_gpu(points,nidxs,nn_size,gpu_index=0):
    '''

    :param points: n,k,3
    :param ndixs:  n,k,nn_size
    :param nn_size:
    :param gpu_index:
    :return:
        covars: n,k,9
    '''

    nidxs=np.ascontiguousarray(nidxs,dtype=np.int32)
    points=np.ascontiguousarray(points,dtype=np.float32)
    covars=PointsUtil.ComputeCovars(points,nidxs,nn_size,gpu_index)

    return covars


if __name__=="__main__":
    from draw_util import output_points
    import time
    points=np.loadtxt('/home/liuyuan/tmp/0_8_true.txt',dtype=np.float32)
    points[:,:2]+=0.5
    print np.min(points,axis=0)
    print np.max(points,axis=0)
    output_points('points.txt',points)

    bg=time.time()
    points=np.repeat(points[None,:,:],2,axis=0)
    # print points.shape

    voxels=PointsUtil.Points2VoxelBatchGPU(points,30)
    print 'cost {} s'.format(time.time()-bg)
    voxels=voxels[0]
    voxel_points= voxel2points(voxels)
    voxel_points=voxel_points.astype(np.float32)
    voxel_points[:,:3]/=np.max(voxel_points[:,:3],axis=0,keepdims=True)
    output_points('voxels.txt',voxel_points)