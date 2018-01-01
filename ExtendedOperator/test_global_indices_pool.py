import tensorflow as tf
import numpy as np
import global_indices_pool_grad

def numpy_grad(feats,indices,pool_grad):
    n,f=feats.shape
    b,k=indices.shape

    feats_grad=np.zeros([n,f],dtype=np.float32)
    for i in xrange(b):
        for j in xrange(k):
            feats_grad[indices[i,j]]+=pool_grad[i,j]

    return feats_grad

global_indices_pool_module=tf.load_op_library('./build/libGlobalIndicesPool.so')
def test_global_pool():
    pts=np.random.uniform(-1,1,[5,2]).astype(np.float32)
    indices=np.random.randint(0,5,[2,5],dtype=np.int64)
    print pts
    print indices
    with tf.Session():
        print global_indices_pool_module.global_indices_pool(pts,indices).eval()

def test_global_pool_grad():
    pts=np.random.uniform(-1,1,[5,2]).astype(np.float32)
    indices=np.random.randint(0,5,[2,5],dtype=np.int64)
    pool_grads=np.random.uniform(0,1,[2,5,2]).astype(np.float32)

    print pts
    print '////////////////////'
    print indices
    print '////////////////////'
    print pool_grads
    print '////////////////////'

    feature_pl=tf.placeholder(tf.float32,shape=[None,None])
    indices_pl=tf.placeholder(tf.int64,shape=[None,None])
    pool_grads_pl=tf.placeholder(tf.float32,shape=[None,None,None])

    with tf.Session() as sess:
        pool_feats=global_indices_pool_module.global_indices_pool(feature_pl,indices_pl)
        feats_grads=tf.gradients(pool_feats,feature_pl,pool_grads_pl)

        feats_val,feats_grads_val=sess.run([pool_feats,feats_grads],feed_dict={
            feature_pl:pts,
            indices_pl:indices,
            pool_grads_pl:pool_grads
        })

    print feats_val
    print '////////////////////'
    print feats_grads_val
    print '////////////////////'

    print feats_grads_val-numpy_grad(pts,indices,pool_grads)

if __name__=="__main__":
    test_global_pool_grad()