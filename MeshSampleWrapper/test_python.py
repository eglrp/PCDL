import point_sample
import numpy as np

pt_num=10000

buf,type=point_sample.getPointCloud("/home/pal/data/ModelNet40/train0.batch",23,pt_num)

print len(buf)

data=np.frombuffer(buf,dtype=np.float_,count=pt_num*3)
data.shape=[pt_num,3]

with open('test.txt','w') as f:
    for i in range(pt_num):
        f.write('{0} {1} {2}\n'.format(data[i][0],data[i][1],data[i][2]))