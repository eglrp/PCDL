from s3dis_util.util import read_pkl
import numpy as np
import random
import time
from s3dis_util.util import get_train_test_split

import threading


class BlockProducer(threading.Thread):
    def __init__(self,file_list,file_indices,slots,items,mutex,data,end):
        threading.Thread.__init__(self)
        self.file_list=file_list
        self.file_indices=file_indices

        self.slots=slots
        self.items=items
        self.mutex=mutex
        self.data=data
        self.reset_event=threading.Event()
        self.end=end

    def run(self):
        while(True):
            for fi in self.file_indices:
                self.slots.acquire()
                self.mutex.acquire()
                if self.end.is_set():
                    exit(0)
                # fetch data
                item=read_pkl(self.file_list[fi], "rb")
                self.data.append(item)
                self.mutex.release()
                self.items.release()

            # wait for reset
            self.reset_event.clear()
            self.reset_event.wait()

    def reset(self,indices):
        self.file_indices=indices
        self.reset_event.set()


class BlockProvider:
    def __init__(self, file_list, batch_size, max_queue_list=5, model='train', with_original_pts=False):
        self.batch_size=batch_size
        self.file_list=file_list
        self.cur_file_pos=0
        self.file_len=len(self.file_list)
        self.with_original_pts=with_original_pts

        file_indices=np.arange(0,len(file_list))
        self.model=model
        if model=='train':
            random.shuffle(file_indices)

        self.slots=threading.Semaphore(max_queue_list)
        self.items=threading.Semaphore(0)
        self.mutex=threading.Lock()
        self.close_thread=threading.Event()
        self.close_thread.clear()
        self.queue=[]
        self.producer=BlockProducer(file_list,file_indices,self.slots,self.items,self.mutex,self.queue,self.close_thread)
        self.producer.start()

        self.file_block_nums=[]
        for f in self.file_list:
            block_list,_=read_pkl(f)
            self.file_block_nums.append(len(block_list))

        self.total_size = sum(self.file_block_nums)
        self.cur_block_pos = 0

        self.block_list=None
        self.global_pts=None
        self.block_indices=None

        self._get_block()

    def _get_block(self):
        # print 'next block'

        self.items.acquire()
        self.mutex.acquire()
        self.block_list, self.global_pts=self.queue.pop(0)
        self.mutex.release()
        self.slots.release()

        self.cur_block_pos=0
        self.block_indices=np.arange(0,len(self.block_list))
        self.block_len=len(self.block_list)
        if self.model=='train':
            random.shuffle(self.block_indices)

        # print 'cur file pos {}'.format(self.cur_file_pos)
        self.cur_file_pos+=1

    def _get_batch(self):
        end_index=min(self.cur_block_pos+self.batch_size,self.block_len)
        batch_indices=list(self.block_indices[self.cur_block_pos:end_index])
        if len(batch_indices)<self.batch_size and self.model=='train':
            batch_indices+=list(np.random.randint(0,self.block_len,self.batch_size-len(batch_indices)))

        cont_block_indices=[np.expand_dims(self.block_list[index]['cont_index'],0) for index in batch_indices]
        room_indices=[np.expand_dims(self.block_list[index]['room_index'],0) for index in batch_indices]
        labels=[np.expand_dims(self.block_list[index]['label'],0) for index in batch_indices]
        cont_block_indices=np.concatenate(cont_block_indices,axis=0).astype(np.int64)        # [batch_size,4096]
        room_indices=np.concatenate(room_indices,axis=0).astype(np.int64)
        labels=np.squeeze(np.concatenate(labels,axis=0),axis=2).astype(np.int64)

        cont_points=[self.block_list[index]['cont'] for index in batch_indices]
        cont_points=np.concatenate(cont_points,axis=0).astype(np.float32)

        cont_lens=[len(self.block_list[index]['cont']) for index in batch_indices]
        cont_batch_indices=[0 for _ in xrange(len(batch_indices))]
        for i in xrange(1,len(batch_indices)):
            cont_batch_indices[i]=cont_batch_indices[i-1]+cont_lens[i-1]

        self.cur_block_pos+=self.batch_size

        local_feats=[np.expand_dims(self.block_list[index]['feat'],0) for index in batch_indices]
        local_feats=(np.concatenate(local_feats,axis=0)).astype(np.float32)      # [batch_size,4096,33]

        batch_data={}
        batch_data['global_pts']=self.global_pts
        batch_data['global_indices']=room_indices
        batch_data['context_pts']=cont_points
        batch_data['context_batch_indices']=cont_batch_indices
        batch_data['context_block_indices']=cont_block_indices
        batch_data['local_feats']=local_feats
        batch_data['labels']=labels

        if self.with_original_pts:
            pts = [np.expand_dims(self.block_list[index]['data'], 0) for index in batch_indices]
            pts = np.concatenate(pts,axis=0)
            batch_data['data']=pts

        return batch_data


    def __iter__(self):
        return self

    def next(self):
        if self.cur_file_pos>=self.file_len and self.cur_block_pos>=self.block_len:
            self.cur_file_pos=0
            indices=range(0,self.file_len)
            if self.model=='train':
                random.shuffle(indices)
            self.producer.reset(indices)

            self._get_block()
            raise StopIteration

        if self.cur_block_pos>=self.block_len:
            self._get_block()

        return self._get_batch()

    def close(self):
        self.close_thread.set()
        self.slots.release()


class BlockProviderMultiGPUWrapper:
    def __init__(self,gpu_num,file_list,batch_size,max_queue_list=5,model='train'):
        self.producer=BlockProvider(file_list,batch_size,max_queue_list,model)
        self.gpu_num=gpu_num
        self.end_iter=False
        self.total_size=self.producer.total_size

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
                    item=self.producer.next()

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
        self.producer.close()


if __name__=="__main__":
    train_fs,test_fs,train_nums,test_nums=get_train_test_split()
    train_list=['data/S3DIS/train/'+fs+'.pkl' for fs in train_fs]
    train_reader=BlockProviderMultiGPUWrapper(2,train_list,train_nums,5,model='train')

    for data in train_reader:
        print np.max(data[0]['context_pts'],axis=0),np.min(data[0]['context_pts'],axis=0)
        # print np.max(data[1]['context_pts'],axis=0),np.min(data[1]['context_pts'],axis=0)
        print '///////////////////////'
        print np.max(data[0]['global_pts'],axis=0),np.min(data[0]['global_pts'],axis=0)
        # print np.max(data[1]['global_pts'],axis=0),np.min(data[1]['global_pts'],axis=0)
        print '///////////////////////'
        print np.max(data[0]['local_feats'],axis=(0,1))
        print np.min(data[0]['local_feats'],axis=(0,1))
        # print np.max(data[1]['local_feats'],axis=(0,1)),np.min(data[1]['local_feats'],axis=(0,1))
        break




