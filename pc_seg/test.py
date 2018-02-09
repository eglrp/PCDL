import numpy as np
import time
import random

t=range(10000)

begin=time.time()
for _ in xrange(100):
    np.random.shuffle(t)
print 'cost {} s'.format(time.time()-begin)

begin=time.time()
for _ in xrange(100):
    random.shuffle(t)
print 'cost {} s'.format(time.time()-begin)

t=np.asarray(t)
begin=time.time()
for _ in xrange(100):
    idx=np.random.choice(1000,1000)
    c=t[idx]
print 'cost {} s'.format(time.time()-begin)