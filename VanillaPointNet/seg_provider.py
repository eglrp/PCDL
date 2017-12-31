from s3dis_util.util import read_room_context
import numpy as np
import random
import time

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
                item=read_room_context(self.file_list[fi],"rb")
                self.data.append(item)
                self.mutex.release()
                self.items.release()

            # wait for reset
            self.reset_event.clear()
            self.reset_event.wait()

    def reset(self,indices):
        self.indices=indices
        self.reset_event.set()


class BlockProvider:
    def __init__(self,file_list,block_nums,batch_size,max_queue_list=5,model='train'):
        self.batch_size=batch_size
        self.file_list=file_list
        self.cur_file_pos=0
        self.file_len=len(self.file_list)

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

        self.file_block_nums = block_nums
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
        self.block_list, self.global_pts=self.queue.pop()
        self.mutex.release()
        self.slots.release()

        self.cur_block_pos=0
        self.block_indices=np.arange(0,len(self.block_list))
        self.block_len=len(self.block_list)
        if self.model=='train':
            random.shuffle(self.block_indices)

        print 'cur file pos {}'.format(self.cur_file_pos)
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
        labels=np.squeeze(np.concatenate(labels,axis=0)).astype(np.int64)

        cont_points=[self.block_list[index]['cont'] for index in batch_indices]
        cont_points=np.concatenate(cont_points,axis=0).astype(np.float32)

        cont_batch_indices=[np.ones(len(self.block_list[index]['cont'])) for index in batch_indices]
        cont_batch_indices=np.concatenate(cont_batch_indices,axis=0).astype(np.int64)

        self.cur_block_pos+=self.batch_size

        feats=[np.expand_dims(self.block_list[index]['feat'],0) for index in batch_indices]
        feats=np.squeeze(np.concatenate(feats,axis=0)).astype(np.float32)      # [batch_size,4096,33]

        return self.global_pts.astype(np.float32),room_indices,cont_points,cont_batch_indices,cont_block_indices,feats,labels


    def __iter__(self):
        return self

    def next(self):
        if self.cur_file_pos>=self.file_len and self.cur_block_pos>=self.block_len:
            self.cur_file_pos=0
            if self.model=='train':
                self.producer.reset(random.shuffle(range(0,self.file_len)))
            else:
                self.producer.reset(range(0,self.file_len))

            self._get_block()
            raise StopIteration

        if self.cur_block_pos>=self.block_len:
            self._get_block()

        return self._get_batch()

    def close(self):
        self.close_thread.set()
        self.slots.release()


class BlockProviderMultiGPUWrapper:
    def __init__(self,gpu_num,file_list,block_nums,batch_size,max_queue_list=5,model='train'):
        self.producer=BlockProvider(file_list,block_nums,batch_size,max_queue_list,model)
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
    f=open('s3dis_util/room_stems.txt','r')
    file_stems=[line.strip('\n') for line in f.readlines()]
    f.close()
    print 'fs count: {}'.format(len(file_stems))

    f=open('s3dis_util/room_block_nums.txt','r')
    block_nums=[int(line.strip('\n')) for line in f.readlines()]
    f.close()

    file_list=['data/S3DIS/train/'+fs+'.pkl' for fs in file_stems]

    provider=BlockProviderMultiGPUWrapper(2,file_list,block_nums,batch_size=5,max_queue_list=5,model='test')
    print 'done'

    begin=time.time()
    for i,data in enumerate(provider):
        print i,len(data[0][-3]),len(data[1][-3])
        # with open('room_{}.txt'.format(i),'w') as f:
        #     for pt in data[0][0]:
        #         f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],
        #                                            int(pt[3]*128+128),
        #                                            int(pt[4]*128+128),
        #                                            int(pt[5]*128+128)))
        #
        #
        # with open('block_{}.txt'.format(i),'w') as f:
        #     for pt in data[0][2]:
        #         f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],
        #                                            int(pt[3]*128+128),
        #                                            int(pt[4]*128+128),
        #                                            int(pt[5]*128+128)))
        #
        # if i >5: break

    print 'cost {} s '.format(time.time()-begin)
    provider.close()




