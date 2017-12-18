import PointSample
import numpy as np
import random
from time import time
from concurrent.futures import ThreadPoolExecutor

pt_num=2048

test_num=200
model_num=PointSample.getModelNum("/home/pal/data/ModelNet40/train0.batch")
total_time=0
transfer_time=0
io_time=0

batch_size=20

executor=ThreadPoolExecutor(max_workers=2)


for k in range(test_num/batch_size):
    # get batch info
    paths=[]
    indices=[]
    pt_nums=[]
    for i in range(batch_size):
        paths.append("/home/pal/data/ModelNet40/train0.batch")
        indices.append(random.randint(0,model_num-1))
        pt_nums.append(pt_num)

    begin=time()
    results=executor.map(PointSample.getPointCloudRelativePolarForm,paths,indices,pt_nums)
    total_time+=time()-begin

    t=0
    for buf,type in results:
        data=np.frombuffer(buf,dtype=np.float_,count=pt_num*3)
        data.shape=[pt_num,3]

        # with open('results/test{0}_{1}.txt'.format(k,t),'w') as f:
        #     for i in range(pt_num):
        #         f.write('{0} {1} {2}\n'.format(data[i][0],data[i][1],data[i][2]))
        t+=1

print 'time used:{0}s, average{1}s'.format(total_time,total_time/float(test_num))


# no parallel version
print '////////////without parallel////////////'
for k in range(test_num):
    begin=time()
    buf,type=PointSample.getPointCloudRelativePolarForm("/home/pal/data/ModelNet40/train0.batch",random.randint(0,model_num-1),pt_num)

    data=np.frombuffer(buf,dtype=np.float_,count=pt_num*3)
    data.shape=[pt_num,3]
    total_time+=time()-begin

    # with open('results/test{0}.txt'.format(k),'w') as f:
    #     for i in range(pt_num):
    #         f.write('{0} {1} {2}\n'.format(data[i][0],data[i][1],data[i][2]))

print 'time used:{0}s, average{1}s'.format(total_time,total_time/float(test_num))