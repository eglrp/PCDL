from cls_network import Network
from cls_provider_v2 import *
import tensorflow as tf
from draw_util import *

if __name__=="__main__":
    input_dim=3
    final_dim=16
    num_classes=40

    points_pl=tf.placeholder(tf.float32,[None,None,3])
    indices_pl=tf.placeholder(tf.int64,[None,None,None,2])
    labels_pl=tf.placeholder(tf.int64,[None,])
    is_training=tf.placeholder(tf.bool)

    net=Network(input_dim,num_classes,True,final_dim)
    net.inference_normal(points_pl,'cpu_',is_training)
    net.normal_constraint_loss(indices_pl,net.ops['points_feats'],10.0)
    opt=tf.train.GradientDescentOptimizer(1e-1)
    loss_op=tf.add_n(tf.get_collection('losses'))
    train_op=opt.minimize(loss_op)

    # train_file_list=['data/ModelNet40/ply_data_train{}.h5'.format(i) for i in range(0,5)]
    test_file_list=['data/ModelNet40/ply_data_test{}.h5'.format(i) for i in range(0,2)]

    train_points,train_normals,train_nidxs,train_mdists,train_labels=read_all_data(test_file_list)
    train_fetch_data=functools.partial(fetch_data,points=train_points,labels=train_labels,
                      normals=train_normals,nidxs=train_nidxs,mdists=train_mdists)
    input_list=[(i,) for i in range(train_points.shape[0])]
    provider=Provider(input_list,16,train_fetch_data,'train',batch_fn=fetch_batch)
    points,labels,indices=provider.next()
    provider.close()
    fixed_points=np.expand_dims(points[0],axis=0)
    fixed_indices=np.expand_dims(indices[0],axis=0)

    output_points(fixed_points[0],"result/model.txt")

    random_points=np.random.uniform(-1, 1, [1, 4096, 3])
    random_points/=np.sqrt(np.sum(random_points ** 2, axis=2, keepdims=True))
    random_points*=np.random.uniform(0, 1, random_points.shape)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        _,loss_val=sess.run([train_op,loss_op],feed_dict={indices_pl:fixed_indices,points_pl:fixed_points,is_training:True})

        if i%10==0:
            print 'step {} loss {:.8}'.format(i,loss_val)

        if i%500==0:
            feats_val=sess.run(net.ops['points_feats'],feed_dict={points_pl:fixed_points,is_training:True})
            for dim in list(range(16)):
                # output_activation(feats_val[0],'result/{}_active_{}.txt'.format(i,dim),dim,fixed_points[0])
                output_activation_distribution(feats_val[0],dim,'result/{}_dist_{}'.format(i,dim))