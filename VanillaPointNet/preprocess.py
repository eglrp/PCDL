import numpy as np
import math
import time
import random
from concurrent.futures import ThreadPoolExecutor
import math

import PointSample


def add_noise(pcs,stddev=1e-2):
    '''
    :param pcs:
    :param stddev:
    :return:
    '''
    pcs+=np.random.normal(0,stddev,pcs.shape)
    return pcs


def normalize(pcs):
    '''
    :param pcs: [n,k,3]
    :return:
    '''
    eps=1e-8
    pcs-=(np.max(pcs,axis=1,keepdims=True)+np.min(pcs,axis=1,keepdims=True))/2.0  #centralize
    dist=pcs[:,:,0]**2+pcs[:,:,1]**2+pcs[:,:,2]**2
    max_dist=np.sqrt(np.max(dist,axis=1,keepdims=True))
    pcs/=(np.expand_dims(max_dist,axis=2)+eps)
    return pcs


def exchange_dims_zy(pcs):
    #pcs [n,k,3]
    exchanged_data = np.empty(pcs.shape, dtype=np.float32)

    exchanged_data[:,:,0]=pcs[:,:,0]
    exchanged_data[:,:,1]=pcs[:,:,2]
    exchanged_data[:,:,2]=pcs[:,:,1]
    return exchanged_data


def rotate(pcs):
    '''
    :param pcs: [n,k,3]
    :return:
    '''
    rotated_data = np.empty(pcs.shape, dtype=np.float32)
    for k in range(pcs.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval,  0],
                                    [      0,      0, 1]],dtype=np.float32)
        shape_pc = pcs[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc, rotation_matrix)
    return rotated_data


def compute_group(pcs):
    def func(coord):
        if coord[0]>0:
            if coord[1]>0:
                if coord[2]>0:
                    return 0
                else:
                    return 1
            else:
                if coord[2]>0:
                    return 2
                else:
                    return 3
        else:
            if coord[1]>0:
                if coord[2]>0:
                    return 4
                else:
                    return 5
            else:
                if coord[2]>0:
                    return 6
                else:
                    return 7

    indices=np.empty([pcs.shape[0],pcs.shape[1]],dtype=np.int64)
    for i in range(pcs.shape[0]):
        indices[i,:]=np.apply_along_axis(func,1,pcs[i,:])
    return indices


def renormalize(pcs,indices,patch_num):
    '''
    :param pcs: n k 3
    :param indices:  n k
    :return:
    '''
    rearranged_indices=np.empty_like(indices)
    renormalized_pts=np.empty_like(pcs)
    for b in range(pcs.shape[0]):
        filled_index=0
        for i in range(patch_num):
            mask=indices[b,:]==i
            if(np.sum(mask)==0):
                continue
            patch_pts=np.copy(pcs[b][mask])
            patch_pts-=(np.min(patch_pts,axis=0,keepdims=True)+np.max(patch_pts,axis=0,keepdims=True))/2.0
            dist=patch_pts[:,0]**2+patch_pts[:,1]**2+patch_pts[:,2]**2
            max_dist=np.sqrt(np.max(dist))
            patch_pts/=max_dist+1e-6
            rearranged_indices[b,filled_index:filled_index+patch_pts.shape[0]]=i
            renormalized_pts[b,filled_index:filled_index+patch_pts.shape[0]]=patch_pts
            filled_index+=patch_pts.shape[0]

    return rearranged_indices,renormalized_pts


class ModelBatchReader:
    def __init__(self,batch_files,batch_size,thread_num,pt_num,input_dims,model='train',
                 read_func=PointSample.getPointCloudRelativePolarForm,aug_func=None):
        self.example_list = []
        for f in batch_files:
            model_num=PointSample.getModelNum(f)
            for i in xrange(model_num):
                self.example_list.append((f,i))

        self.model=model
        if model=='train':
            random.shuffle(self.example_list)

        self.cur_pos=0
        self.executor=ThreadPoolExecutor(max_workers=thread_num)
        self.batch_size=batch_size
        self.total_size=len(self.example_list)
        self.pt_num=pt_num
        self.input_dims=input_dims
        self.aug_func=aug_func
        self.read_func=read_func

    def __iter__(self):
        return self

    def next(self):

        if self.cur_pos>self.total_size:
            self.cur_pos=0
            if self.model=='train':
                random.shuffle(self.example_list)
            raise StopIteration

        cur_read_size=min(self.total_size-self.cur_pos,self.batch_size)
        cur_batch_list=[]
        cur_batch_list+=self.example_list[self.cur_pos:self.cur_pos+cur_read_size]

        if self.model=='train':
            cur_sample_size=self.batch_size-cur_read_size
            if cur_sample_size>0:
                for _ in xrange(cur_sample_size):
                    sample_index=random.randint(0,len(self.example_list)-1)
                    cur_batch_list.append(self.example_list[sample_index])

        file_names=[t[0] for t in cur_batch_list]
        model_indices=[t[1] for t in cur_batch_list]
        pt_nums=[self.pt_num for _ in xrange(len(cur_batch_list))]

        input_total_dim=1
        if self.input_dims is list:
            for dim in self.input_dims:
                input_total_dim*=dim

            input_shapes=[self.pt_num*self.input_dims[0],self.input_dims[1]]       # [pt_num*(pt_num-1),5]
        else:
            input_total_dim=self.input_dims
            input_shapes=[self.pt_num,self.input_dims]                        # [pt_num,3]

        results=self.executor.map(self.read_func, file_names, model_indices, pt_nums)
        inputs=[]
        labels=[]
        for input,label in results:
            data=np.frombuffer(input,dtype=np.float64,count=self.pt_num*input_total_dim)
            inputs.append(np.reshape(data,input_shapes).astype(np.float32))
            labels.append(label)

        if self.aug_func is not None:
            inputs=self.aug_func(np.asarray(inputs))

        self.cur_pos+=self.batch_size

        return np.asarray(inputs),np.asarray(labels)

import h5py
class H5Reader:
    def __init__(self,file_list,):
        data=[]
        label=[]
        for f in file_list:
            h5f=h5py.File(f)
            data.append(h5f['data'][:])
            label.append(h5f['label'][:])

        self.data=np.concatenate(data,axis=0)
        self.data = exchange_dims_zy(self.data)
        self.label=np.concatenate(label,axis=0)
        self.total_size=self.label.shape[0]
        self.cur_pos=0

        




    def __iter__(self):
        return self

    def next(self):
        pass


def test_reader():
    batch_files=['data/ModelNet40/test0.batch']
    batch_size=30
    thread_num=2
    pt_num=2048
    point_stddev=1e-2
    batch_num=1#int(math.ceil(reader.total_size/float(batch_size)))

    begin=time.time()

    def aug_func(pcs):
        pcs=normalize(pcs)
        pcs=add_noise(pcs,point_stddev)
        pcs=rotate(pcs)
        pcs=normalize(pcs)
        return pcs

    reader=ModelBatchReader(batch_files,batch_size,thread_num,pt_num,3,
                            model='test',read_func=PointSample.getPointCloud,
                            aug_func=aug_func)

    i=0
    for data,label in reader:
        t1=time.time()
        indices=compute_group(data)
        print 'half {} s'.format(time.time()-t1)
        renormalized_indices,renormalized_data=renormalize(data,indices,8)
        print 'done {} s'.format(time.time()-t1)
        # if random.random()<1.0:
        #     for l in xrange(len(label)):
        #         for p in xrange(8):
        #             with open('test_part_{0}_{1}_{2}.txt'.format(i,l,p),'w') as f:
        #                 for pt in renormalized_data[l][renormalized_indices[l]==p]:
        #                     f.write('{} {} {}\n'.format(pt[0],pt[1],pt[2]))
        #
        #         with open('test_part_{0}_{1}.txt'.format(i, l), 'w') as f:
        #             for pt in data[l]:
        #                 f.write('{} {} {}\n'.format(pt[0], pt[1], pt[2]))
        # break


    print '{} examples per second'.format(reader.total_size/float(time.time()-begin))


def test_group():
    # pts=np.array([[1,1,1],[0,0,0],[0,0,1]],dtype=np.float32)
    # pts=np.reshape(pts,[1,3,3,1])
    pts=np.random.uniform(-1000,1000,[30,500,3])

    indices=compute_group(pts)
    renormalize(pts,indices,8)

    color=np.random.uniform(0,1,[8,3])*255
    color=color.astype(np.int)

    pt_index=np.random.randint(0,30,1)
    coords=pts[pt_index,:,:]
    print indices.shape
    print coords[0].shape

    with open('test.txt','w') as f:
        for index,pt in enumerate(coords[0]):
            f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],
                                                 color[int(indices[pt_index,index]), 0],
                                                 color[int(indices[pt_index,index]), 1],
                                                 color[int(indices[pt_index,index]), 2]))


if __name__=="__main__":
    test_reader()


# deprecated below due to efficiency
def polar_transform(pcs):
    '''
    :param pcs: [n,k,3] n point clouds, each has k points (x,y,z)
    :return: [n,k,k-1,5]
    '''
    # subtract gravity center
    pcs-=np.expand_dims(np.mean(pcs,axis=1),axis=1)

    t1=time.time()
    # compute polar coordinate
    polar_pcs=np.empty_like(pcs)
    polar_pcs[:,:,0]=np.arctan2(pcs[:,:,1],pcs[:,:,0])      # theta
    polar_pcs[:,:,1]=np.sqrt(pcs[:,:,1]**2+pcs[:,:,0]**2)   # range
    polar_pcs[:,:,2]=pcs[:,:,2]                             # height

    print time.time()-t1
    t2=time.time()

    relative_polar_pcs=np.empty([polar_pcs.shape[0],polar_pcs.shape[1],polar_pcs.shape[1]-1,5])   # [n,k,k-1,5]
    # set delta theta
    delta_theta=np.expand_dims(polar_pcs[:,:,0],axis=2)-np.expand_dims(polar_pcs[:,:,0],axis=1)   # [n,k,k]

    masks=np.ones([polar_pcs.shape[1],polar_pcs.shape[1]]) # [k,k]
    np.fill_diagonal(masks,0)
    masks=np.asarray(masks,dtype=np.bool)
    masks=np.repeat(np.expand_dims(masks, axis=0), polar_pcs.shape[0], axis=0)  # [n,k,k]

    delta_theta[delta_theta>math.pi]-=math.pi
    delta_theta[delta_theta<-math.pi]+=math.pi
    delta_theta=delta_theta/math.pi*180

    delta_theta=delta_theta[masks]                     #n*k*(k-1)
    relative_polar_pcs[:,:,:,0]=np.reshape(delta_theta,[polar_pcs.shape[0],polar_pcs.shape[1],polar_pcs.shape[1]-1])

    # set r1 r2 h1 h2
    relative_polar_pcs[:,:,:,1]=np.repeat(np.expand_dims(polar_pcs[:,:,1],axis=2),polar_pcs.shape[1]-1,axis=2) #r1
    relative_polar_pcs[:,:,:,3]=np.repeat(np.expand_dims(polar_pcs[:,:,2],axis=2),polar_pcs.shape[1]-1,axis=2) #h1


    r2=np.repeat(np.expand_dims(polar_pcs[:, :, 1], axis=1), polar_pcs.shape[1],axis=1)[masks]
    relative_polar_pcs[:,:,:,2]=np.reshape(r2,[polar_pcs.shape[0],polar_pcs.shape[1],polar_pcs.shape[1]-1])

    h2=np.repeat(np.expand_dims(polar_pcs[:, :, 2], axis=1), polar_pcs.shape[1],axis=1)[masks]
    relative_polar_pcs[:,:,:,4]=np.reshape(h2,[polar_pcs.shape[0],polar_pcs.shape[1],polar_pcs.shape[1]-1])

    print time.time()-t2

    return relative_polar_pcs


def test_transform():
    pc1=[
        [[2, 0, 1], [-2, 0, -1], [0, 1, 0], [0, -1, 0]]
        ,[[2, 2, 1], [-2, -2, -1], [0, 1, 0], [0, -1, 0]]
    ]
    print polar_transform(np.asarray(pc1,dtype=np.float64))

    begin=time.time()
    for _ in range(5):
        polar_transform(np.random.uniform(-1,1,[20,2048,3]))
    print 'cost {}s'.format(time.time()-begin)


def compute_dist(pcs):
    '''
    :param pcs:
    :return:
    '''
    eps=1e-8
    dists=np.empty([pcs.shape[0],pcs.shape[1],pcs.shape[1],1])
    for i in xrange(pcs.shape[0]):
        pc=pcs[i]
        dist=np.sum((np.repeat(pc[:,None,:],pcs.shape[1],axis=1)-
                    np.repeat(pc[None,:,:],pcs.shape[1],axis=0))**2,axis=2)
        # print dist.shape
        dists[i,:,:,:]=np.sqrt(dist+eps)

    dists=np.reshape(dists,[pcs.shape[0],pcs.shape[1],pcs.shape[1]])

    return dists


def test_diff():
    pts=np.random.uniform(-1000,1000,[30,500,3,1])
    # pts=np.array([[1,1,1],[0,0,0],[0,0,1]],dtype=np.float32)
    pts=np.asarray(pts,dtype=np.float32)
    # pts=np.reshape(pts,[1,3,3,1])
    dist=PointSample.getPointInterval(pts)

    # print dist

    import sys
    print sys.getrefcount(dist)
    print dist.shape

    dist2=compute_dist(pts)

    print np.mean(dist-dist2)