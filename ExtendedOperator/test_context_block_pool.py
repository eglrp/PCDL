import tensorflow as tf
import numpy as np
import context_block_pool_grad

def numpy_gradients(feat,batch_indices,block_indices,pool_grad):
    n,f=feat.shape
    b,k=block_indices.shape

    feat_grad=np.zeros([n,f])
    for i in xrange(b):
        for j in xrange(k):
            feat_grad[batch_indices[i]+block_indices[i,j],:]+=pool_grad[i,j,:]

    return feat_grad

context_block_pool_module=tf.load_op_library('./build/libContextBlockPool.so')
def test_global_pool():
    cont_len=np.random.randint(1,3,3)

    batch_indices=np.zeros(3)
    for i in range(1,3):
        batch_indices[i]=batch_indices[i-1]+cont_len[i-1]

    cont_feature=[]
    for l in cont_len:
        cont_feature.append(np.random.uniform(-1,1,[l,3]))
    cont_feature=np.concatenate(cont_feature,axis=0)

    block_indices=[]
    for l in cont_len:
        block_indices.append(np.random.randint(0,l,[1,5],dtype=np.int64))
    block_indices=np.concatenate(block_indices,axis=0)


    print cont_feature
    print '//////////////////////////'
    print cont_len
    print '//////////////////////////'
    print block_indices
    print '//////////////////////////'
    with tf.Session():
        print context_block_pool_module.context_block_pool(cont_feature, batch_indices, block_indices).eval()


def test_global_pool_grad():
    cont_len=np.random.randint(1,3,3)

    batch_indices=np.zeros(3,dtype=np.int)
    for i in range(1,3):
        batch_indices[i]=batch_indices[i-1]+cont_len[i-1]

    cont_feature=[]
    for l in cont_len:
        cont_feature.append(np.random.uniform(-1,1,[l,3]))
    cont_feature=np.concatenate(cont_feature,axis=0)

    block_indices=[]
    for l in cont_len:
        block_indices.append(np.random.randint(0,l,[1,5],dtype=np.int64))
    block_indices=np.concatenate(block_indices,axis=0)

    pool_grads=np.random.uniform(-1,1,[3,5,3])

    feature_pl=tf.placeholder(tf.float32,shape=[None,None])
    batch_indices_pl=tf.placeholder(tf.int64,shape=[None,])
    block_indices_pl=tf.placeholder(tf.int64,shape=[None,None])
    pool_grads_pl=tf.placeholder(tf.float32,shape=[None,None,None])


    print '/////////////cont_feature/////////////'
    print cont_feature
    print '/////////////cont_len/////////////'
    print cont_len
    print '//////////////block_indices////////////'
    print block_indices
    print '/////////////pool_grads/////////////'
    print pool_grads
    with tf.Session() as sess:
        pool_feats=context_block_pool_module.context_block_pool(feature_pl,batch_indices_pl,block_indices_pl)
        feats_grads=tf.gradients(pool_feats,feature_pl,pool_grads_pl)

        feats_val,feats_grads_val=sess.run([pool_feats,feats_grads],feed_dict={
            feature_pl:cont_feature,
            batch_indices_pl:batch_indices,
            block_indices_pl:block_indices,
            pool_grads_pl:pool_grads
        })

    print '//////////////feats_val////////////'
    print feats_val
    print '//////////////feats_grads_val////////////'
    print feats_grads_val

    print feats_grads_val[0]-numpy_gradients(cont_feature,batch_indices,block_indices,pool_grads)


if __name__=="__main__":
    test_global_pool_grad()

