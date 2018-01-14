from autoencoder_network import *
import numpy as np
import time
from s3dis.data_util import voxel2points


def test_whole_folding_net():
    batch_size=24
    pt_num=2048
    grid_num=2025

    pts_pl=tf.placeholder(tf.float32,[batch_size,pt_num,6])
    covar_pl=tf.placeholder(tf.float32,[batch_size,pt_num,9])
    nidxs_pl=tf.placeholder(tf.int32,[batch_size,pt_num,8,2])
    grids_pl=tf.placeholder(tf.float32,[grid_num,3])


    codewords=folding_net_encoder(pts_pl, covar_pl, nidxs_pl)
    gen_pts=folding_net_decoder(codewords, grids_pl, 6)
    dist_loss,color_loss=chamfer_color_loss(pts_pl,gen_pts)
    opt=tf.train.AdamOptimizer(1e-3)
    loss=color_loss+dist_loss
    train_op=opt.minimize(loss)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    begin_time=time.time()
    for i in xrange(3200):
        pts=np.random.uniform(-1,1,[batch_size,pt_num,6])
        covar=np.random.uniform(-1,1,[batch_size,pt_num,9])
        nidxs=np.random.randint(0,batch_size,[batch_size,pt_num,8,2])
        grids=np.random.uniform(-1,1,[grid_num,3])

        _,loss_val=sess.run([train_op,loss],{pts_pl:pts,nidxs_pl:nidxs,grids_pl:grids,covar_pl:covar})

        if i%32==0:
            print 'loss {} | {} examples/s'.format(loss_val,(32*batch_size)/(time.time()-begin_time))
            begin_time=time.time()


def numpy_loss(ref_pts,gen_pts):
    n,k1,_=ref_pts.shape
    n,k2,_=gen_pts.shape

    chamfer_dists=[]
    for i in xrange(n):
        ref_dists=[]
        for k in xrange(k1):
            diff=(np.expand_dims(ref_pts[i,k,:],axis=0)-gen_pts[i])**2   # k2,3
            dist=np.min(np.sum(diff,axis=1),axis=0)
            ref_dists.append(np.sqrt(dist))

        ref_dist=np.mean(np.asarray(ref_dists))

        gen_dists=[]
        for k in xrange(k2):
            diff=(np.expand_dims(gen_pts[i,k,:],axis=0)-ref_pts[i])**2   # k2,3
            dist=np.min(np.sum(diff,axis=1),axis=0)
            gen_dists.append(np.sqrt(dist))

        gen_dist=np.mean(np.asarray(gen_dists))

        chamfer_dists.append(max(gen_dist,ref_dist))

    return np.mean(np.asarray(chamfer_dists))


def numpy_color_loss(ref_pts,gen_pts):
    n,k1,_=ref_pts.shape
    n,k2,_=gen_pts.shape

    chamfer_dists=[]
    for i in xrange(n):
        ref_dists=[]
        ref_color_dist=[]
        for k in xrange(k1):
            diff=(np.expand_dims(ref_pts[i,k,:3],axis=0)-gen_pts[i,:,:3])**2   # k2,3
            diff=np.sum(diff, axis=1)
            dist=np.min(diff,axis=0)
            ref_dists.append(np.sqrt(dist))

            color_diff=ref_pts[i,k,3:]-gen_pts[i,np.argmin(diff),3:]
            color_diff=np.sum(color_diff**2)
            ref_color_dist.append(np.sqrt(color_diff))

        ref_dist=np.mean(np.asarray(ref_dists))
        ref_color_dist=np.mean(np.asarray(ref_color_dist))

        gen_dists=[]
        gen_color_dist=[]
        for k in xrange(k2):
            diff=(np.expand_dims(gen_pts[i,k,:3],axis=0)-ref_pts[i,:,:3])**2   # k2,3
            diff=np.sum(diff, axis=1)
            dist=np.min(diff,axis=0)
            gen_dists.append(np.sqrt(dist))

            color_diff=gen_pts[i,k,3:]-ref_pts[i,np.argmin(diff),3:]
            color_diff=np.sum(color_diff**2)
            gen_color_dist.append(np.sqrt(color_diff))

        gen_dist=np.mean(np.asarray(gen_dists))
        gen_color_dist=np.mean(np.asarray(gen_color_dist))

        chamfer_dists.append(max(gen_dist,ref_dist)+max(gen_color_dist,ref_color_dist))

    return np.mean(np.asarray(chamfer_dists))


def test_loss():
    ref_pts_pl=tf.placeholder(tf.float32,[None,None,3])
    gen_pts_pl=tf.placeholder(tf.float32,[None,None,3])

    loss=chamfer_loss(ref_pts_pl,gen_pts_pl)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    for i in xrange(100):
        ref_pts=np.random.uniform(-1,1,[30,2048,3])
        gen_pts=np.random.uniform(-1,1,[30,2045,3])

        loss_val=sess.run(loss,{ref_pts_pl:ref_pts,gen_pts_pl:gen_pts})
        np_loss_val=numpy_loss(ref_pts,gen_pts)

        print loss_val,np_loss_val
        print np.max(loss_val-np_loss_val)


def test_graph_layer():
    nidxs_pl=tf.placeholder(tf.int32,[5,2048,8])
    nidxs=nidxs_pl

    n,k,t=tf.shape(nidxs)[0],tf.shape(nidxs)[1],tf.shape(nidxs)[2]
    nidxs=tf.expand_dims(nidxs,axis=3)                     # n,k,t,1
    batch_idxs=tf.range(n,dtype=tf.int32)
    batch_idxs=tf.expand_dims(batch_idxs,axis=1)           # n,1,1,1
    batch_idxs=tf.expand_dims(batch_idxs,axis=2)           # n,1,1,1
    batch_idxs=tf.expand_dims(batch_idxs,axis=3)           # n,1,1,1
    batch_idxs=tf.tile(batch_idxs,[1,k,t,1])               # n,k,t,1
    nidxs=tf.concat([batch_idxs,nidxs],axis=3)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    fixed_nidxs=np.random.randint(0,2048,[5,2048,8],dtype=np.int32)
    print fixed_nidxs[1]
    print sess.run(nidxs,feed_dict={nidxs_pl:fixed_nidxs})[1]


def test_color_loss():
    ref_pts_pl=tf.placeholder(tf.float32,[None,None,6])
    gen_pts_pl=tf.placeholder(tf.float32,[None,None,6])

    dist_loss,color_loss=chamfer_color_loss(ref_pts_pl,gen_pts_pl)
    loss=dist_loss+color_loss

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    for i in xrange(100):
        ref_pts=np.random.uniform(-1,1,[30,2048,6])
        gen_pts=np.random.uniform(-1,1,[30,2045,6])

        loss_val=sess.run(loss,{ref_pts_pl:ref_pts,gen_pts_pl:gen_pts})
        np_loss_val=numpy_color_loss(ref_pts,gen_pts)

        print loss_val,np_loss_val
        print np.max(loss_val-np_loss_val)


def test_voxel_filling_net():
    import numpy as np
    from s3dis.draw_util import output_points
    voxel_num=64000
    batch_size=32
    points_pl=tf.placeholder(tf.float32,[batch_size,4096,3],'points')
    true_state_pl=tf.placeholder(tf.float32,[batch_size,voxel_num],'voxel_true_state')

    feats=vanilla_pointnet_encoder(points_pl)
    voxel_state=fc_voxel_decoder(feats, voxel_num)
    loss=voxel_filling_loss(voxel_state, true_state_pl)
    opt=tf.train.AdamOptimizer(1e-3)
    minimize_op=opt.minimize(loss)

    points=np.random.uniform(-1,1,[batch_size,4096,3])
    true_state=np.random.uniform(0,1,[batch_size,voxel_num])
    true_state=np.asarray(true_state>0.9,np.float32)
    true_state=np.asarray(true_state,np.float32)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    output_points('true.txt', voxel2points(true_state[0]))
    for i in range(100000):
        _,loss_val,pred_state=sess.run([minimize_op,loss,voxel_state],{points_pl:points,true_state_pl:true_state})

        if i%100==0:
            output_points('pred{}.txt'.format(i), voxel2points(pred_state[0]))
            print 'step {} loss val {}'.format(i,loss_val)


def test_voxel_filling_color_net():
    import numpy as np
    from s3dis.draw_util import output_points
    voxel_num=64000
    batch_size=32
    points_pl=tf.placeholder(tf.float32,[batch_size,4096,15],'points')
    true_state_pl=tf.placeholder(tf.float32,[batch_size,voxel_num],'voxel_true_state')
    true_color_pl=tf.placeholder(tf.float32,[batch_size,voxel_num,3],'voxel_true_color')

    feats=vanilla_pointnet_encoder(points_pl)
    voxel_state,voxel_color=fc_voxel_decoder(feats, voxel_num, True)

    filling_loss=voxel_filling_loss(voxel_state, true_state_pl)
    color_loss=voxel_color_loss(voxel_color, true_color_pl)
    loss=filling_loss+color_loss

    opt=tf.train.AdamOptimizer(1e-3)
    minimize_op=opt.minimize(loss)

    points=np.random.uniform(-1,1,[batch_size,4096,3])
    true_state=np.random.uniform(0,1,[batch_size,voxel_num])
    true_state=np.asarray(true_state>0.9,np.float32)
    true_state=np.asarray(true_state,np.float32)

    true_color=np.random.uniform(-1,1,[batch_size,voxel_num,3])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    output_points('true.txt', voxel2points(true_color[0]))
    begin=time.time()
    for i in range(100000):
        _,loss_val,pred_state,pred_color=sess.run([minimize_op,loss,voxel_state,voxel_color],
                                                  {points_pl:points,
                                                   true_state_pl:true_state,
                                                   true_color_pl:true_color})

        if i%100==0:
            output_points('pred{}.txt'.format(i), voxel2points(pred_color[0]))
            print 'step {} loss val {} | {} examples/s'.format(i,loss_val,100*batch_size/(time.time()-begin))


def test_points2voxel():
    from provider import ProviderV2
    from s3dis.block_util import read_block_v2
    from s3dis.data_util import point2voxel,get_train_test_split

    def read_fn(filename):
        points,covars=read_block_v2(filename)[:2]
        voxels=[]
        for pts in points:
            # voxel=point2voxel(pts,30)
            voxels.append(np.arange(5))

        voxels=np.asarray(voxels)

        return points, covars, voxels

    def batch_fn(file_data, cur_idx, data_indices, require_size):
        points, covars, voxels = file_data
        end_idx = min(cur_idx + require_size, points.shape[0])

        return [points[data_indices[cur_idx:end_idx], :],
                covars[data_indices[cur_idx:end_idx], :],
                voxels[data_indices[cur_idx:end_idx], :]
                ], end_idx - cur_idx

    train_list,_=get_train_test_split()
    train_list=['data/S3DIS/folding/block_v2/{}.h5'.format(fn) for fn in train_list]

    train_provider = ProviderV2(train_list,'train',32, batch_fn, read_fn,2)

    begin=time.time()
    for data in train_provider:
        for item in data:
            print item.shape

        print 'cost {} s'.format(time.time()-begin)
        begin=time.time()

if __name__=="__main__":
    test_points2voxel()