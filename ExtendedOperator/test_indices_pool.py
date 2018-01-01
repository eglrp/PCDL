import tensorflow as tf
import numpy as np
import indices_pool_grad


indices_pool_module=tf.load_op_library("./build/libIndicesPool.so")
def numpy_indices_pool(feats,indices,patch_num):
    '''
    :param feats: n k f
    :param indices:  n k
    :return:
    '''
    output=np.empty(shape=[feats.shape[0],patch_num,feats.shape[2]])
    for i in range(feats.shape[0]):
        for j in xrange(patch_num):
            # print feats[i,indices[i,:]==j].shape
            output[i,j,:]=np.max(feats[i,indices[i,:]==j],axis=0)

    return output

def test_forward():
    pts=np.random.uniform(-1,1,[30,1024,1024]).astype(np.float32)
    indices=np.random.randint(0,8,[30,1024],dtype=np.int64)

    # pts=np.reshape(pts,[1,3,3])
    # indices=np.reshape(indices,[1,3])

    numpy_result=numpy_indices_pool(pts,indices,patch_num=8)


    with tf.Session():
        print np.mean(indices_pool_module.indices_pool(pts,indices,patch_num=8).eval()-numpy_result)

def test_backward_forwad():
    # pts=np.random.uniform(-1,1,[30,1024,1024]).astype(np.float32)
    # indices=np.random.randint(0,8,[30,1024],dtype=np.int64)

    pts=np.array([[1,2,3],[3,2,1],[1,3,5]],dtype=np.float32)
    indices=np.array([1,1,0])
    pts=np.reshape(pts,[1,3,3])
    indices=np.reshape(indices,[1,3])
    pool_grads=np.random.uniform(0,1,[1,2,3]).astype(np.float32)

    print pool_grads

    feature_pl=tf.placeholder(tf.float32,shape=[None,None,None])
    indices_pl=tf.placeholder(tf.int64,shape=[None,None])
    pool_grads_pl=tf.placeholder(tf.float32,shape=[None,None,None])

    with tf.Session() as sess:
        pool_feats=indices_pool_module.indices_pool(feature_pl,indices_pl,patch_num=2)
        feats_grads=tf.gradients(pool_feats,feature_pl,pool_grads_pl)

        feats_val,feats_grads_val=sess.run([pool_feats,feats_grads],feed_dict={
            feature_pl:pts,
            indices_pl:indices,
            pool_grads_pl:pool_grads
        })

    print feats_val
    print feats_grads_val

def test_backward_forwad_numeric():
    batch_size=30
    pt_num=4096
    feat_dim=1024
    patch_num=8
    pts=np.random.uniform(-1,1,[batch_size,pt_num,feat_dim]).astype(np.float32)
    indices=np.random.randint(0,patch_num,[batch_size,pt_num],dtype=np.int64)

    feature_pl=tf.placeholder(tf.float32,shape=[None,None,None])
    indices_pl=tf.placeholder(tf.int64,shape=[None,None])

    argmax_indices=np.argmax(pts,axis=1)

    with tf.Session() as sess:
        pool_feats=indices_pool_module.indices_pool(feature_pl,indices_pl,patch_num=patch_num)
        sum_tensor=tf.reduce_mean(pool_feats,axis=[0,1,2])
        feats_grads=tf.gradients(sum_tensor,feature_pl,1.0)
        import time
        begin=time.time()
        sum_val,feats_grads_val=sess.run([sum_tensor,feats_grads],feed_dict={
            feature_pl:pts,
            indices_pl:indices,
            # pool_grads_pl:pool_grads
        })
        print 'cost {}s'.format(time.time()-begin)

        # print sum_val
        # # print feats_grads_val
        # # print indices
        # print pts
        #
        # feats_grads_val=feats_grads_val[0]
        #
        # eps=1
        # for _ in range(1000):
        #     x=int(np.random.randint(0,pts.shape[0],1))
        #     y=int(np.random.randint(0,pts.shape[2],1))
        #     pts2=np.copy(pts)
        #     pts2[x,argmax_indices[x,y],y]+=eps
        #     sum_val_added,_=sess.run([sum_tensor,feats_grads],feed_dict={
        #         feature_pl:pts2,
        #         indices_pl:indices,
        #         # pool_grads_pl:pool_grads
        #     })
        #     pts2=np.copy(pts)
        #     pts2[x,argmax_indices[x,y],y]-=eps
        #     sum_val_subed,_=sess.run([sum_tensor,feats_grads],feed_dict={
        #         feature_pl:pts2,
        #         indices_pl:indices,
        #         # pool_grads_pl:pool_grads
        #     })
        #     # print feats_grads_val[x,argmax_indices[x,y],y]
        #     diff=(sum_val_added-sum_val)/eps-feats_grads_val[x,argmax_indices[x,y],y]
        #     print '{} {} {} diff {:.8} val {}'.format(x,argmax_indices[x,y],y,diff,feats_grads_val[x,argmax_indices[x,y],y])
        #     assert abs(diff)<1e-5

def test_backward():
    grad_module=tf.load_op_library("./cmake-build-debug/libIndicesPoolGrad.so")
    pts=np.array([[1,2,3],[3,2,1],[1,3,5]],dtype=np.float32)
    indices=np.array([1,1,0])
    pts=np.reshape(pts,[1,3,3])
    indices=np.reshape(indices,[1,3])
    pool_grads=np.random.uniform(0,1,[1,2,3]).astype(np.float32)

    print pool_grads

    with tf.Session():
        print grad_module.indices_pool_grad(pts,indices,pool_grads,patch_num=2).eval()



if __name__=="__main__":
    test_backward_forwad_numeric()


