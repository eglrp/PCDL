import threading
import random
from concurrent.futures import ThreadPoolExecutor
import math
import numpy as np


class ProviderV2(threading.Thread):
    def __init__(self,file_list,model,batch_size,batch_fn,read_fn,max_cache=2):
        threading.Thread.__init__(self)

        self.slots=threading.Semaphore(max_cache)
        self.items=threading.Semaphore(0)
        self.mutex=threading.Lock()
        self.epoch_end=threading.Event()
        self.thread_end=threading.Event()
        self.data_cache=[]

        self.file_list=tuple(file_list)
        self.file_len=len(self.file_list)
        self.indices=range(len(file_list))

        self.file_cur=0

        self.model=model
        self.read_fn=read_fn
        self.batch_fn=batch_fn

        self.batch_size=batch_size
        self.done=False

        self.start()

        self.request_data()

    def run(self):
        while True:
            for idx in self.indices:
                self.slots.acquire()
                self.mutex.acquire()
                if self.thread_end.is_set():
                    exit(0)

                self.data_cache.append(self.read_fn(self.file_list[idx]))
                self.mutex.release()
                self.items.release()

            # wait for reset
            self.epoch_end.clear()
            self.epoch_end.wait()

            if self.model=='train':
                random.shuffle(self.indices)

    def request_data(self):
        # print 'request'
        if self.file_cur>=self.file_len:
            self.cur_data=None
            return

        self.items.acquire()
        self.mutex.acquire()
        file_data=self.data_cache.pop(0)
        self.mutex.release()
        self.slots.release()

        self.file_cur+=1

        self.cur_data=file_data
        if self.cur_data is not None:
            self.cur_data_index=0
            self.cur_indices=range(self.cur_data[0].shape[0])
            if self.model=='train':
                random.shuffle(self.cur_indices)

    def reset(self):
        self.file_cur=0
        self.epoch_end.set()

    def close(self):
        self.thread_end.set()
        self.slots.release()
        self.epoch_end.set()

    def __iter__(self):
        return self

    def next(self):
        if self.done:
            self.done=False
            self.reset()
            self.request_data()
            raise StopIteration

        batch_data, actual_size = self.batch_fn(self.cur_data,self.cur_data_index,self.cur_indices,self.batch_size)
        self.cur_data_index+=actual_size

        left_size = self.batch_size - actual_size

        while self.cur_data_index>=self.cur_data[0].shape[0]:
            self.request_data()

            # no data available
            if self.cur_data is None:
                self.done=True
                break

            # data available and we still need to sample
            if left_size>0:
                left_batch_data, actual_size = self.batch_fn(self.cur_data,self.cur_data_index,self.cur_indices, left_size)
                for data_idx in xrange(len(left_batch_data)):
                    batch_data[data_idx]=np.concatenate([batch_data[data_idx],left_batch_data[data_idx]],axis=0)

                left_size -= actual_size
                self.cur_data_index += actual_size
            else: break

        return batch_data


######################deprecated below################
class ThreadReader(threading.Thread):
    def __init__(self,file_list,model,read_fn,max_cache=2):
        threading.Thread.__init__(self)

        self.slots=threading.Semaphore(max_cache)
        self.items=threading.Semaphore(0)
        self.mutex=threading.Lock()
        self.epoch_end=threading.Event()
        self.thread_end=threading.Event()
        self.data_cache=[]

        self.file_list=tuple(file_list)
        self.file_len=len(self.file_list)
        self.indices=range(len(file_list))

        self.file_cur=0

        self.model=model
        self.read_fn=read_fn

        self.start()

    def run(self):
        while True:
            for idx in self.indices:
                self.slots.acquire()
                self.mutex.acquire()
                if self.thread_end.is_set():
                    exit(0)

                self.data_cache.append(self.read_fn(self.file_list[idx]))
                self.mutex.release()
                self.items.release()

            # wait for reset
            self.epoch_end.clear()
            self.epoch_end.wait()

            if self.model=='train':
                random.shuffle(self.indices)

    def request_data(self):
        # print 'request'
        if self.file_cur>=self.file_len:
            return None

        self.items.acquire()
        self.mutex.acquire()
        file_data=self.data_cache.pop(0)
        self.mutex.release()
        self.slots.release()

        self.file_cur+=1

        return file_data

    def reset(self):
        self.file_cur=0
        self.epoch_end.set()

    def close(self):
        self.thread_end.set()
        self.slots.release()
        self.epoch_end.set()


class Provider(threading.Thread):
    def __init__(self, input_list, batch_size, fetch_fn, model, cache_batch_num=2, batch_fn=None, max_worker_num=4):

        threading.Thread.__init__(self)
        self.slots=threading.Semaphore(cache_batch_num)
        self.items=threading.Semaphore(0)
        self.mutex=threading.Lock()
        self.epoch_end = threading.Event()
        self.thread_end = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=max_worker_num)

        self.indices = range(len(input_list))
        self.cur_pos = 0
        self.batch_size = batch_size
        self.batch_index = 0
        self.batch_num = int(math.ceil(float(len(input_list))/batch_size))
        # print self.batch_num

        self.model = model
        self.input_list = input_list
        self.output_queue = []

        self.fetch_fn = fetch_fn
        self.batch_fn = batch_fn

        if self.model == 'train':
            random.shuffle(self.indices)

        self.start()

    def run(self):
        while True:

            if self.cur_pos >= len(self.indices):
                self.epoch_end.clear()
                self.epoch_end.wait()

            # fetch data
            end_index = min(self.cur_pos + self.batch_size, len(self.indices))
            futures = []
            for i in xrange(self.cur_pos, end_index):
                futures.append(self.executor.submit(self.fetch_fn, self.model, *self.input_list[self.indices[i]]))

            batch_data = []
            for f in futures:
                batch_data.append(f.result())
            self.cur_pos = end_index

            # insert data
            self.slots.acquire()
            self.mutex.acquire()
            if self.thread_end.is_set():
                exit(0)

            self.output_queue.append(batch_data)
            self.mutex.release()
            self.items.release()

    def _reset(self):
        if self.model == 'train':
            random.shuffle(self.indices)

        self.cur_pos = 0
        self.epoch_end.set()

    def close(self):
        self.thread_end.set()
        self.slots.release()
        self.epoch_end.set()

    def __iter__(self):
        return self

    def next(self):

        if self.batch_index>=self.batch_num:
            self.batch_index=0
            self._reset()
            raise StopIteration

        self.items.acquire()
        self.mutex.acquire()
        batch_data=self.output_queue.pop(0)
        self.mutex.release()
        self.slots.release()
        self.batch_index+=1

        # post process
        if self.batch_fn is not None:
            batch_data=self.batch_fn(self.model,batch_data)

        return batch_data


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
    import numpy as np
    train_data=np.arange(32)
    def fetch_data(model,index):
        if model=='train':
            return train_data[index] + 1
        return train_data[index]

    input_list=[(i,) for i in range(32)]
    provider=Provider(input_list,3,fetch_data,'train')

    try:
        for i,data in enumerate(provider):
            print i,data
            raise RuntimeError
    finally:
        provider.close()