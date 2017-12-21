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

        inputs=[]
        labels=[]
        results=self.executor.map(self.read_func, file_names, model_indices, pt_nums)
        for input,label in results:
            data=np.frombuffer(input,dtype=np.float64,count=self.pt_num*input_total_dim)
            inputs.append(np.reshape(data,input_shapes).astype(np.float32))
            labels.append(label)

        if self.aug_func is not None:
            inputs=self.aug_func(np.asarray(inputs))

        self.cur_pos+=self.batch_size

        return np.expand_dims(np.asarray(inputs),axis=3),np.asarray(labels)


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
                            model='test',read_func=PointSample.getPointCloud,aug_func=aug_func)

    i=0
    for data,label in reader:
        if random.random()<1.0:
            for l in xrange(len(label)):
                with open('test{0}_{1}.txt'.format(i,l),'w') as f:
                    for k in range(pt_num):
                        f.write('{0} {1} {2}\n'.format(data[l][k,0,0],data[l][k,1,0],data[l][k,2,0]))
        i+=1
        break
        pass


    print '{} examples per second'.format(reader.total_size/float(time.time()-begin))



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
