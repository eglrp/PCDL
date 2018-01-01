import tensorflow as tf
import numpy as np
import context_batch_pool_grad

def numpy_gradients(feat,batch_indices,pool_grad):
    indices=list(np.copy(batch_indices))
    indices.append(feat.shape[0])
    indices=np.asarray(indices)

    feat_grad=np.zeros_like(feat)
    for bi in xrange(indices.shape[0]-1):
        max_indices=np.argmax(feat[indices[bi]:indices[bi+1]],axis=0)
        max_indices+=indices[bi]
        for fi in xrange(feat.shape[1]):
            feat_grad[max_indices[fi],fi]=pool_grad[bi,fi]

    return feat_grad

context_batch_pool_module=tf.load_op_library('./build/libContextBatchPool.so')
def test_global_pool():
    cont_len=np.random.randint(1,3,3)

    batch_indices=np.zeros(3)
    for i in range(1,3):
        batch_indices[i]=batch_indices[i-1]+cont_len[i-1]

    cont_feature=[]
    for l in cont_len:
        cont_feature.append(np.random.uniform(-1,1,[l,3]))
    cont_feature=np.concatenate(cont_feature,axis=0)

    print cont_feature
    print '//////////////////////////'
    print cont_len
    print '//////////////////////////'
    with tf.Session():
        print context_batch_pool_module.context_batch_pool(cont_feature, batch_indices).eval()

def test_global_pool_grad():
    cont_len=np.random.randint(1,3,3)

    batch_indices=np.zeros(3,dtype=np.int)
    for i in range(1,3):
        batch_indices[i]=batch_indices[i-1]+cont_len[i-1]

    cont_feature=[]
    for l in cont_len:
        cont_feature.append(np.random.uniform(-1,1,[l,3]))
    cont_feature=np.concatenate(cont_feature,axis=0)

    pool_grads=np.random.uniform(-1,1,[3,3])

    feature_pl=tf.placeholder(tf.float32,shape=[None,None])
    batch_indices_pl=tf.placeholder(tf.int64,shape=[None,])
    pool_grads_pl=tf.placeholder(tf.float32,shape=[None,None])


    print '/////////////cont_feature/////////////'
    print cont_feature
    print '/////////////cont_len/////////////'
    print cont_len
    print '/////////////pool_grads/////////////'
    print pool_grads
    with tf.Session() as sess:
        pool_feats=context_batch_pool_module.context_batch_pool(feature_pl,batch_indices_pl)
        feats_grads=tf.gradients(pool_feats,feature_pl,pool_grads_pl)

        feats_val,feats_grads_val=sess.run([pool_feats,feats_grads],feed_dict={
            feature_pl:cont_feature,
            batch_indices_pl:batch_indices,
            pool_grads_pl:pool_grads
        })

    print '//////////////feats_val////////////'
    print feats_val
    print '//////////////feats_grads_val////////////'
    print feats_grads_val
    print '//////////////////////////////'
    print feats_grads_val[0]-numpy_gradients(cont_feature,batch_indices,pool_grads)

if __name__=="__main__":
    test_global_pool_grad()