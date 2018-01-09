import threading
import random
from concurrent.futures import ThreadPoolExecutor
import math


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