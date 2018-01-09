from cls_network import Network
from cls_provider import *
import tensorflow as tf
import time

log_dir = 'train.log'
def log_info(message):
    with open(log_dir,'a') as f:
        f.write(message)
        f.write('\n')
    print message

def train():
    gpu_num=2
    lr_init=1e-3
    lr_decay_rate=0.9
    lr_decay_epoch=3
    bn_init=0.95
    bn_decay_rate=0.9
    bn_decay_epoch=3
    normal_ratio=1e-5
    bn_clip=0.95
    batch_size=8

    input_dims=3
    num_classes=40
    pt_num=2048

    log_epoch=1
    model_dir='model'
    train_epoch_num=500

    test_batch_files=['data/ModelNet40/test0.batch']
    train_batch_files=['data/ModelNet40/train0.batch',
                       'data/ModelNet40/train1.batch',
                       'data/ModelNet40/train2.batch',
                       'data/ModelNet40/train3.batch']

    train_provider=PointSampleProvider(train_batch_files,batch_size,pt_num,'train')
    test_provider=PointSampleProvider(test_batch_files,batch_size,pt_num,'test')
    trainset=ProviderMultiGPUWrapper(gpu_num,train_provider)
    testset=ProviderMultiGPUWrapper(gpu_num,test_provider)


    total_size=train_provider.total_size

    inputs=[]
    normals=[]
    labels=[]
    for i in xrange(gpu_num):
        inputs.append(tf.placeholder(tf.float32,[None,pt_num,input_dims]))
        normals.append(tf.placeholder(tf.float32,[None,pt_num,input_dims]))
        labels.append(tf.placeholder(tf.int64,[None,]))
    is_training=tf.placeholder(tf.bool)

    net=Network(input_dims,num_classes,True,final_dim=16)
    net.declare_train_net_normal(inputs,normals,labels,is_training,gpu_num,normal_ratio,
                                 lr_init,lr_decay_rate,lr_decay_epoch,
                                 bn_init,bn_decay_rate,bn_decay_epoch,bn_clip,
                                 batch_size,total_size)

    saver=tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train',sess.graph)

    try:
        for i in xrange(train_epoch_num):
            feed_dict={}
            train_count=0
            begin=time.time()
            for feedin in trainset:
                for k in xrange(gpu_num):
                    feed_dict[inputs[k]],feed_dict[normals[k]],feed_dict[labels[k]]= \
                        feedin[k][0],feedin[k][1],feedin[k][2]
                feed_dict[is_training]=True
                # print 'reader cost {:.5} s'.format(time.time()-test_time)

                train_count+=1

                # test_time=time.time()
                sess.run(net.ops['apply_grad'],feed_dict)
                # print 'run cost {:.5} s'.format(time.time()-test_time)

                if train_count%log_epoch==0:
                    grad_norm,points_grad,acc,loss,summary_str=sess.run([net.ops['points_grad_norm'],net.ops['points_grad'],net.ops['accuracy'],net.ops['loss'],merged],feed_dict=feed_dict)
                    train_writer.add_summary(summary_str,global_step=(i*total_size+train_count))
                    train_writer.flush()
                    print np.min(points_grad),np.max(points_grad)
                    print np.min(grad_norm),np.max(grad_norm)
                    log_info('epoch {} step {}: train accuracy {:.5} loss {:.5} | {:.5} examples per second'.
                             format(i,train_count,acc,loss,batch_size*gpu_num*log_epoch/(time.time()-begin)))
                    begin=time.time()

            test_correct_num=0
            total_num=0
            for feedin in testset:
                all_labels=[]
                for k in xrange(gpu_num):
                    feed_dict[inputs[k]],feed_dict[normals[k]],feed_dict[labels[k]]= \
                        feedin[k][0],feedin[k][1],feedin[k][2]
                    all_labels.append(feedin[k][2])

                all_labels=np.concatenate(all_labels,axis=0)
                feed_dict[is_training]=False

                logits=sess.run(net.ops['logits'],feed_dict=feed_dict)
                # print logits
                preds=np.argmax(logits,axis=1)
                test_correct_num+=np.sum(preds==all_labels)
                total_num+=len(all_labels)

            log_info('epoch {}: test accuracy {}'.format(i,test_correct_num/float(total_num)))
            saver.save(sess,model_dir+'/epoch{}.ckpt'.format(i))
    finally:
        trainset.close()
        testset.close()

def train_pointnet():
    gpu_num=2

    lr_init=1e-1
    lr_decay_rate=0.9
    lr_decay_epoch=3

    bn_init=0.95
    bn_decay_rate=0.9
    bn_decay_epoch=3
    bn_clip=0.95

    batch_size=16
    input_dims=3
    num_classes=40
    pt_num=2048

    log_epoch=20
    model_dir='model'
    train_epoch_num=500

    test_batch_files=['data/ModelNet40/test0.batch']
    train_batch_files=['data/ModelNet40/train0.batch',
                       'data/ModelNet40/train1.batch',
                       'data/ModelNet40/train2.batch',
                       'data/ModelNet40/train3.batch']

    reader=PointReader(pt_num, 5e-3)
    train_provider=PointSampleProvider(train_batch_files,batch_size,reader,'train')
    test_provider=PointSampleProvider(test_batch_files,batch_size,reader,'test')
    trainset=ProviderMultiGPUWrapper(gpu_num,train_provider)
    testset=ProviderMultiGPUWrapper(gpu_num,test_provider)

    total_size=train_provider.total_size

    inputs=[]
    # normals=[]
    labels=[]
    for i in xrange(gpu_num):
        inputs.append(tf.placeholder(tf.float32,[None,pt_num,input_dims]))
        # normals.append(tf.placeholder(tf.float32,[None,pt_num,input_dims]))
        labels.append(tf.placeholder(tf.int64,[None,]))
    is_training=tf.placeholder(tf.bool)

    net=Network(input_dims,num_classes,True,1024,True)
    net.declare_train_net(inputs,labels,is_training,gpu_num,
                         lr_init,lr_decay_rate,lr_decay_epoch,
                         bn_init,bn_decay_rate,bn_decay_epoch,bn_clip,
                         batch_size,total_size)

    saver=tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train',sess.graph)

    try:
        for i in xrange(train_epoch_num):
            feed_dict={}
            train_count=0
            begin=time.time()
            for feedin in trainset:
                for k in xrange(gpu_num):
                    feed_dict[inputs[k]],feed_dict[labels[k]]= \
                        feedin[k][0],feedin[k][1]
                feed_dict[is_training]=True
                # print 'reader cost {:.5} s'.format(time.time()-test_time)

                train_count+=1

                # test_time=time.time()
                sess.run(net.ops['apply_grad'],feed_dict)
                # print 'run cost {:.5} s'.format(time.time()-test_time)

                if train_count%log_epoch==0:
                    logits,global_step,acc,loss,summary_str=sess.run([net.ops['logits'],net.ops['global_step'],net.ops['accuracy'],
                                                               net.ops['loss'],merged],feed_dict=feed_dict)
                    train_writer.add_summary(summary_str,global_step=global_step)
                    train_writer.flush()
                    # print np.argmax(logits,axis=1)
                    log_info('epoch {} step {}: train accuracy {:.5} loss {:.5} | {:.5} examples per second'.
                             format(i,train_count,acc,loss,batch_size*gpu_num*log_epoch/(time.time()-begin)))
                    begin=time.time()

            test_correct_num=0
            total_num=0
            for feedin in testset:
                all_labels=[]
                for k in xrange(gpu_num):
                    feed_dict[inputs[k]],feed_dict[labels[k]]= \
                        feedin[k][0],feedin[k][1]
                    all_labels.append(feedin[k][1])

                all_labels=np.concatenate(all_labels,axis=0)
                feed_dict[is_training]=False

                logits=sess.run(net.ops['logits'],feed_dict=feed_dict)
                # print logits
                preds=np.argmax(logits,axis=1)
                test_correct_num+=np.sum(preds==all_labels)
                total_num+=len(all_labels)

            log_info('epoch {}: test accuracy {}'.format(i,test_correct_num/float(total_num)))
            saver.save(sess,model_dir+'/epoch{}.ckpt'.format(i))
    finally:
        trainset.close()
        testset.close()

from cls_provider_v2 import *
def train_normal_pointnet():
    gpu_num=2

    lr_init=5e-5
    lr_decay_rate=0.5
    lr_decay_epoch=20

    bn_init=0.5
    bn_decay_rate=0.5
    bn_decay_epoch=20
    bn_clip=0.99

    constraint_ratio=0.0

    batch_size=16
    input_dims=3
    num_classes=40
    pt_num=2048

    log_epoch=20
    model_dir='model'
    train_epoch_num=500

    train_file_list=['data/ModelNet40/ply_data_train{}.h5'.format(i) for i in range(0,5)]
    test_file_list=['data/ModelNet40/ply_data_test{}.h5'.format(i) for i in range(0,2)]

    train_points,train_normals,train_nidxs,train_mdists,train_labels=read_all_data(train_file_list)
    test_points,_,_,_,test_labels=read_all_data(test_file_list)
    train_fetch_data=functools.partial(fetch_data,points=train_points,labels=train_labels,
                      normals=train_normals,nidxs=train_nidxs,mdists=train_mdists)
    test_fetch_data=functools.partial(fetch_data,points=test_points,labels=test_labels)

    train_input_list=[(i,) for i in range(train_points.shape[0])]
    test_input_list=[(i,) for i in range(test_points.shape[0])]
    train_provider=Provider(train_input_list,batch_size,train_fetch_data,'train',batch_fn=fetch_batch)
    test_provider=Provider(test_input_list,batch_size,test_fetch_data,'test',batch_fn=fetch_batch)

    trainset=ProviderMultiGPUWrapper(gpu_num,train_provider)
    testset=ProviderMultiGPUWrapper(gpu_num,test_provider)

    total_size=train_provider.batch_num

    points_pl=[]
    nidxs_pl=[]
    labels_pl=[]
    for i in xrange(gpu_num):
        points_pl.append(tf.placeholder(tf.float32,[None,None,input_dims]))
        nidxs_pl.append(tf.placeholder(tf.int32,[None,None,5,2]))
        labels_pl.append(tf.placeholder(tf.int64,[None,]))
    is_training=tf.placeholder(tf.bool)

    net=Network(input_dims,num_classes,True,1024,False)
    net.declare_train_net_normal(points_pl,labels_pl,nidxs_pl,constraint_ratio,
                                 is_training,gpu_num,
                                 lr_init,lr_decay_rate,lr_decay_epoch,
                                 bn_init,bn_decay_rate,bn_decay_epoch,bn_clip,
                                 batch_size,total_size)

    saver=tf.train.Saver(max_to_keep=500)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess,'model/epoch123_0.86.ckpt')

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train',sess.graph)

    try:
        for i in xrange(124,train_epoch_num):
            feed_dict={}
            train_count=0
            begin=time.time()
            # test_time=time.time()
            all_cn,all_bn=0,0
            for feedin in trainset:
                for k in xrange(gpu_num):
                    feed_dict[points_pl[k]],feed_dict[labels_pl[k]],feed_dict[nidxs_pl[k]]= \
                        feedin[k][0],feedin[k][1].flatten(),feedin[k][2]
                feed_dict[is_training]=True
                # print 'reader cost {:.5} s'.format(time.time()-test_time)

                train_count+=1

                # test_time=time.time()
                _,cn,bn=sess.run([net.ops['apply_grad'],net.ops['correct_num'],net.ops['batch_num']],feed_dict)
                # print cn,bn
                all_cn+=cn
                all_bn+=bn
                # print 'run cost {:.5} s'.format(time.time()-test_time)
                # test_time=time.time()

                if train_count%log_epoch==0:
                    logits,global_step,loss,summary_str=sess.run([net.ops['logits'],net.ops['global_step'],
                                                               net.ops['loss'],merged],feed_dict=feed_dict)
                    train_writer.add_summary(summary_str,global_step=global_step)
                    train_writer.flush()
                    # print np.argmax(logits,axis=1)
                    log_info('epoch {} step {}: train accuracy {:.5} loss {:.5} | {:.5} examples per second'.
                             format(i,train_count,float(all_cn)/all_bn,loss,batch_size*gpu_num*log_epoch/(time.time()-begin)))
                    begin=time.time()
                    all_cn=0
                    all_bn=0

            test_correct_num=0
            total_num=0
            for feedin in testset:
                all_labels=[]
                for k in xrange(gpu_num):
                    feed_dict[points_pl[k]],feed_dict[labels_pl[k]]= \
                        feedin[k][0],feedin[k][1].flatten()
                    all_labels.append(feedin[k][1].flatten())

                all_labels=np.concatenate(all_labels,axis=0)
                feed_dict[is_training]=False

                logits=sess.run(net.ops['logits'],feed_dict=feed_dict)
                # print logits
                preds=np.argmax(logits,axis=1)
                test_correct_num+=np.sum(preds==all_labels)
                total_num+=len(all_labels)

            acc=test_correct_num/float(total_num)
            log_info('epoch {}: test accuracy {}'.format(i,acc))
            saver.save(sess,model_dir+'/epoch{}_{:.3}.ckpt'.format(i,acc))
    finally:
        trainset.close()
        testset.close()



if __name__=="__main__":
    train_normal_pointnet()