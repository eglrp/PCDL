import PointSample
import random
import numpy as np
import threading
import pyflann
from concurrent.futures import ThreadPoolExecutor

def rotate(pts,rotation_angle=None):
    if rotation_angle is None:
        rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]], dtype=np.float32)
    rotated_pts = np.dot(pts, rotation_matrix)
    return rotated_pts

def normalize(pts):
    eps=1e-7
    pts-=(np.max(pts,axis=0,keepdims=True)+np.min(pts,axis=0,keepdims=True))/2.0  #centralize
    dist=pts[:,0]**2+pts[:,1]**2+pts[:,2]**2
    max_dist=np.sqrt(np.max(dist,keepdims=True))
    pts/=(max_dist+eps)
    return pts

class Reader:
    def __init__(self):
        pass

    def get_example(self, example_file, example_index, model, data):
        pass

    def get_batch(self,data,batch_size,total_size,cur_pos,items,slots,mutex):
        pass

class NormalDiffReader(Reader):
    def __init__(self,sample_num,neighbor_size):
        Reader.__init__(self)
        self.sample_num=sample_num
        self.neighbor_size=neighbor_size

    def get_example(self, example_file, example_index, model, data):
        points, normals, label = PointSample.getPointCloudNormal(example_file, example_index, self.sample_num)

        flann = pyflann.FLANN()
        flann.build_index(points, algorithm='kdtree_simple', leaf_max_size=15)
        indices = np.empty([self.sample_num, self.neighbor_size])
        dists=[]
        for pt_i, pt in enumerate(points):
            cur_indices, cur_dists = flann.nn_index(pt, self.neighbor_size+1) # [1,t+1]
            cur_indices = np.asarray(cur_indices, dtype=np.int).transpose()   # [t+1,1]
            indices[pt_i] = cur_indices[1:,0]
            dists.append(cur_dists[:,1:])

        points -= (np.max(points, axis=0, keepdims=True) + np.min(points, axis=0, keepdims=True)) / 2.0  # centralize
        if model=='train':
            points = rotate(points)

        # flip normals
        mask=np.sum(points*normals,axis=1)<0
        normals[mask,:]=-normals[mask,:]
        # out points
        dists=np.concatenate(dists,axis=0)          # [k,t]
        out_points=points+normals*np.mean(dists,axis=1,keepdims=True)
        points=np.concatenate([points,out_points],axis=0)

        # normalize
        dist = points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2
        max_dist = np.sqrt(np.max(dist, keepdims=True))
        points /= (max_dist + points)

        data.append((points, indices, label))

    def get_batch(self,data,batch_size,total_size,cur_pos,items,slots,mutex):
        all_points,all_indices,all_labels=[],[],[]
        cur_size=0
        while cur_pos<total_size and cur_size<batch_size:
            items.acquire()
            mutex.acquire()
            points,indices,label=data.pop(0)
            mutex.release()
            slots.release()
            all_points.append(np.expand_dims(points,axis=0))
            # indices [n,t]
            indices=np.expand_dims(indices, axis=2)                 #[k,t,1]
            batch_indices=np.full_like(indices,cur_size)            #[k,t,1]
            indices=np.concatenate([batch_indices,indices],axis=2)  #[k,t,2]
            all_indices.append(np.expand_dims(indices,axis=0))      #[1,k,t,2]
            all_labels.append(label)
            cur_pos+=1
            cur_size+=1

        all_points=np.concatenate(all_points,axis=0)        #[n,k,3]
        all_indices=np.concatenate(all_indices,axis=0)      #[n,k,t,2]
        all_labels=np.asarray(all_labels,dtype=np.int64)    #[n]

        return (all_points,all_indices,all_labels.flatten()),cur_pos

class NormalReader(Reader):
    def __init__(self,sample_num,noise_level):
        Reader.__init__(self)
        self.sample_num=sample_num
        self.noise_level=noise_level

    def get_example(self, example_file, example_index, model, data):
        points, normals, label = PointSample.getPointCloudNormal(example_file, example_index, self.sample_num)
        # normalize
        points = normalize(points)
        if model == 'train':
            # rotation
            rotation_angle = np.random.uniform() * 2 * np.pi
            points = rotate(points, rotation_angle)
            normals = rotate(normals, rotation_angle)
            # add noise
            points += np.random.normal(0, self.noise_level, points.shape)
            # normalize
            points = normalize(points)

        data.append((points, normals, label))

    def get_batch(self,data,batch_size,total_size,cur_pos,items,slots,mutex):
        all_points,all_normals,all_labels=[],[],[]
        cur_size=0
        while cur_pos<total_size and cur_size<batch_size:
            items.acquire()
            mutex.acquire()
            points,normals,label=data.pop(0)
            mutex.release()
            slots.release()
            all_points.append(np.expand_dims(points,axis=0))
            all_normals.append(np.expand_dims(normals,axis=0))
            all_labels.append(label)
            cur_pos+=1
            cur_size+=1

        all_points=np.concatenate(all_points,axis=0)
        all_normals=np.concatenate(all_normals,axis=0)
        all_labels=np.asarray(all_labels,dtype=np.int64)

        return (all_points,all_normals,all_labels.flatten()),cur_pos

class PointReader(Reader):
    def __init__(self,sample_num,noise_level):
        Reader.__init__(self)
        self.sample_num=sample_num
        self.noise_level=noise_level

    def get_example(self, example_file, example_index, model, data):
        points, label = PointSample.getPointCloud(example_file, example_index, self.sample_num)
        # normalize
        points = normalize(points)
        if model == 'train':
            # rotation
            rotation_angle = np.random.uniform() * 2 * np.pi
            points = rotate(points, rotation_angle)
            # add noise
            points += np.random.normal(0, self.noise_level, points.shape)
            # normalize
            points = normalize(points)

        data.append((points, label))

    def get_batch(self,data,batch_size,total_size,cur_pos,items,slots,mutex):
        all_points,all_labels=[],[]
        cur_size=0
        while cur_pos<total_size and cur_size<batch_size:
            items.acquire()
            mutex.acquire()
            points,label=data.pop(0)
            mutex.release()
            slots.release()
            all_points.append(np.expand_dims(points,axis=0))
            all_labels.append(label)
            cur_pos+=1
            cur_size+=1

        all_points=np.concatenate(all_points,axis=0)
        all_labels=np.asarray(all_labels,dtype=np.int64)

        return (all_points,all_labels.flatten()),cur_pos

class PointSampler(threading.Thread):
    def __init__(self,example_list,model,slots,items,mutex,data,end,reader):
        threading.Thread.__init__(self)
        self.example_list=example_list

        self.slots=slots
        self.items=items
        self.mutex=mutex
        self.data=data
        self.reset_event=threading.Event()
        self.end=end
        self.model=model
        self.reader=reader

    def run(self):
        while(True):
            for fn,fi in self.example_list:
                self.slots.acquire()
                self.mutex.acquire()
                if self.end.is_set():
                    exit(0)
                # fetch data
                self.reader.get_example(fn,fi,self.model,self.data)
                self.mutex.release()
                self.items.release()

            # wait for reset
            self.reset_event.clear()
            self.reset_event.wait()

    def reset(self):
        if self.model=='train':
            random.shuffle(self.example_list)
        self.reset_event.set()


class PointSampleProvider:
    def __init__(self, batch_files, batch_size, reader, model='train'):
        self.example_list = []
        for f in batch_files:
            model_num=PointSample.getModelNum(f)
            for i in xrange(model_num):
                self.example_list.append((f,i))

        self.model=model
        if model=='train':
            random.shuffle(self.example_list)

        self.reader=reader

        self.slots=threading.Semaphore(batch_size*2)
        self.items=threading.Semaphore(0)
        self.mutex=threading.Lock()
        self.close_thread=threading.Event()
        self.close_thread.clear()
        self.queue=[]
        self.sampler=PointSampler(self.example_list, model, self.slots,
                     self.items, self.mutex, self.queue, self.close_thread, self.reader)
        self.sampler.start()

        self.cur_pos=0
        self.batch_size=batch_size
        self.total_size=len(self.example_list)

    def __iter__(self):
        return self

    def next(self):

        if self.cur_pos>=self.total_size:
            self.cur_pos=0
            self.sampler.reset()
            raise StopIteration

        data,self.cur_pos=self.reader.get_batch(self.queue,self.batch_size,self.total_size,
                                                self.cur_pos,self.items,self.slots,self.mutex)
        return data

    def close(self):
        self.close_thread.set()
        self.slots.release()


class ProviderMultiGPUWrapper:
    def __init__(self,gpu_num,provider):
        '''
        :param gpu_num:
        :param provider: use next() close()
        '''
        self.provider=provider
        self.gpu_num=gpu_num
        self.end_iter=False

    def __iter__(self):
        return self

    def next(self):
        items=[]

        if self.end_iter:
            self.end_iter=False
            raise StopIteration

        for i in range(self.gpu_num):
            try:
                # when end_iter is True, just copy the previous item
                if self.end_iter:
                    item=items[-1]
                else:
                    item=self.provider.next()

            # detect the first time we encounter the StopIteration
            except StopIteration:
                if i==0:
                    raise StopIteration
                else:
                    self.end_iter=True
                    item=items[-1]

            items.append(item)

        return items

    def close(self):
        self.provider.close()

if __name__=="__main__":
    import time
    train_batch_files=['data/ModelNet40/train0.batch',]

    train_provider = PointSampleProvider(train_batch_files, 30, PointReader(2048, 1e-3), 'train')
    try:
        begin=time.time()
        for data,label in train_provider:
            # with open('test.txt','w') as f:
            #     for pt in data[0]:
            #         f.write('{} {} {}\n'.format(pt[0],pt[1],pt[2]))
            # break
            print data.shape,label.shape
            print 'cost {} s'.format(time.time()-begin)
            begin=time.time()
        print 'done'
    finally:
        train_provider.close()