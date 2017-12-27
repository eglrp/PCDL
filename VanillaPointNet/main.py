from network_class import Network
from preprocess import *
import tensorflow as tf
import PointSample

log_dir = 'train.log'
def log_info(message):
    with open(log_dir,'a') as f:
        f.write(message)
        f.write('\n')
    print message

def train():
    gpu_num=2
    lr_init=1e-2
    lr_decay_rate=0.9
    lr_decay_epoch=3
    bn_init=0.95
    bn_decay_rate=0.9
    bn_decay_epoch=3
    bn_clip=0.95
    batch_size=30
    momentum=0.9

    log_epoch=30
    model_dir='model'
    point_stddev=1e-2

    test_batch_files=['data/ModelNet40/test0.batch']
    train_batch_files=['data/ModelNet40/train0.batch',
                       'data/ModelNet40/train1.batch',
                       'data/ModelNet40/train2.batch',
                       'data/ModelNet40/train3.batch']
    read_thread_num=2
    pt_num=2048

    def train_aug_func(pcs):
        pcs=normalize(pcs)
        pcs=rotate(pcs)
        pcs=add_noise(pcs,point_stddev)
        pcs=normalize(pcs)
        return pcs

    train_epoch_num=500

    train_reader=ModelBatchReader(train_batch_files,batch_size*gpu_num,read_thread_num,pt_num,3,'train',
                                  PointSample.getPointCloud,train_aug_func)
    test_reader=ModelBatchReader(test_batch_files,batch_size*gpu_num,read_thread_num,pt_num,3,'test',
                                 PointSample.getPointCloud,normalize)

    total_size=train_reader.total_size
    inputs=[]
    labels=[]
    for i in xrange(gpu_num):
        inputs.append(tf.placeholder(tf.float32,[None,None,3]))
        labels.append(tf.placeholder(tf.float32,[None,]))
    is_training=tf.placeholder(tf.bool)

    net=Network(3,40,True)
    net.declare_train_net(inputs,labels,is_training,gpu_num,
                          lr_init,lr_decay_rate,lr_decay_epoch,
                          bn_init,bn_decay_rate,bn_decay_epoch,bn_clip,
                          batch_size*gpu_num,total_size,momentum)

    saver=tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'model/epoch53.ckpt')

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train',sess.graph)


    for i in xrange(train_epoch_num):
        feed_dict={}
        train_count=0
        begin=time.time()
        for data,label in train_reader:
            # test_time=time.time()
            for k in xrange(gpu_num):
                feed_dict[inputs[k]],feed_dict[labels[k]]=\
                    data[k*batch_size:(k+1)*batch_size],label[k*batch_size:(k+1)*batch_size]
            feed_dict[is_training]=True
            # print 'reader cost {:.5} s'.format(time.time()-test_time)

            train_count+=1

            # test_time=time.time()
            sess.run(net.ops['apply_grad'],feed_dict)
            # print 'run cost {:.5} s'.format(time.time()-test_time)

            if train_count%log_epoch==0:
                acc,loss,summary_str=sess.run([net.ops['accuracy'],net.ops['loss'],merged],feed_dict=feed_dict)
                train_writer.add_summary(summary_str,global_step=(i*total_size+train_count))
                train_writer.flush()
                log_info('epoch {} step {}: train accuracy {:.5} loss {:.5} | {:.5} examples per second'.
                         format(i,train_count,acc,loss,batch_size*gpu_num*log_epoch/(time.time()-begin)))
                begin=time.time()

        test_correct_num=0
        for data,label in test_reader:
            for k in xrange(gpu_num):
                feed_dict[inputs[k]],feed_dict[labels[k]]=\
                    data[k*batch_size:(k+1)*batch_size],label[k*batch_size:(k+1)*batch_size]
            feed_dict[is_training]=False

            logits=sess.run(net.ops['logits'],feed_dict=feed_dict)
            # print logits
            preds=np.argmax(logits,axis=1)
            test_correct_num+=np.sum(preds==label)

        log_info('epoch {}: test accuracy {}'.format(i,test_correct_num/float(test_reader.total_size)))
        saver.save(sess,model_dir+'/epoch{}.ckpt'.format(i))

def train_split():
    gpu_num=2
    lr_init=1e-2
    lr_decay_rate=0.9
    lr_decay_epoch=3
    bn_init=0.95
    bn_decay_rate=0.9
    bn_decay_epoch=3
    bn_clip=0.95
    batch_size=30
    momentum=0.9
    patch_num=8

    log_epoch=30
    model_dir='model'
    point_stddev=1e-2

    test_batch_files=['data/ModelNet40/test0.batch']
    train_batch_files=['data/ModelNet40/train0.batch',
                       'data/ModelNet40/train1.batch',
                       'data/ModelNet40/train2.batch',
                       'data/ModelNet40/train3.batch']
    read_thread_num=2
    pt_num=2048

    def train_aug_func(pcs):
        pcs=normalize(pcs)
        pcs=rotate(pcs)
        pcs=add_noise(pcs,point_stddev)
        pcs=normalize(pcs)
        return pcs

    train_epoch_num=500

    train_reader=ModelBatchReader(train_batch_files,batch_size*gpu_num,read_thread_num,pt_num,3,'train',
                                  PointSample.getPointCloud,train_aug_func)
    test_reader=ModelBatchReader(test_batch_files,batch_size*gpu_num,read_thread_num,pt_num,3,'test',
                                 PointSample.getPointCloud,normalize)

    total_size=train_reader.total_size
    inputs=[]
    labels=[]
    split_indices=[]
    for i in xrange(gpu_num):
        inputs.append(tf.placeholder(tf.float32,[None,None,3]))
        labels.append(tf.placeholder(tf.float32,[None,]))
        split_indices.append(tf.placeholder(tf.int64,[None,None]))
    is_training=tf.placeholder(tf.bool)

    net=Network(3,40,True,1024,True,patch_num)
    net.declare_train_net_split(inputs,labels,split_indices,is_training,gpu_num,
                                lr_init,lr_decay_rate,lr_decay_epoch,
                                bn_init,bn_decay_rate,bn_decay_epoch,bn_clip,
                                batch_size*gpu_num,total_size)

    saver=tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'model/epoch53.ckpt')

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train',sess.graph)


    for i in xrange(train_epoch_num):
        feed_dict={}
        train_count=0

        begin=time.time()
        for data,label in train_reader:

            # test_time=time.time()
            indices=compute_group(data)
            indices,normalized_part=renormalize(data,indices,patch_num)
            # print normalized_part.shape,data.shape
            data=np.concatenate([data,normalized_part],axis=1)
            # print data.shape
            for k in xrange(gpu_num):
                feed_dict[inputs[k]],feed_dict[labels[k]],feed_dict[split_indices[k]]=\
                    data[k * batch_size:(k + 1) * batch_size],\
                    label[k*batch_size:(k+1)*batch_size],\
                    indices[k*batch_size:(k+1)*batch_size]

            feed_dict[is_training]=True
            # print 'split cost {:.5} s'.format(time.time()-test_time)

            train_count+=1

            # test_time=time.time()
            sess.run(net.ops['apply_grad'],feed_dict)
            # print 'run cost {:.5} s'.format(time.time()-test_time)

            if (train_count+1)%log_epoch==0:
                acc,loss,summary_str=sess.run([net.ops['accuracy'],net.ops['loss'],merged],feed_dict=feed_dict)
                train_writer.add_summary(summary_str,global_step=(i*total_size+train_count))
                train_writer.flush()
                log_info('epoch {} step {}: train accuracy {:.5} loss {:.5} | {:.5} examples per second'.
                         format(i,train_count,acc,loss,batch_size*gpu_num*log_epoch/(time.time()-begin)))
                begin=time.time()

        test_correct_num=0
        for data,label in test_reader:
            indices=compute_group(data)
            indices,normalized_part=renormalize(data,indices,patch_num)
            data=np.concatenate([data,normalized_part],axis=1)
            for k in xrange(gpu_num):
                feed_dict[inputs[k]],feed_dict[labels[k]],feed_dict[split_indices[k]]=\
                    data[k * batch_size:(k + 1) * batch_size],\
                    label[k*batch_size:(k+1)*batch_size],\
                    indices[k*batch_size:(k+1)*batch_size]
                if k*batch_size>=data.shape[0]:
                    feed_dict[inputs[k]],feed_dict[labels[k]],feed_dict[split_indices[k]]=\
                        feed_dict[inputs[k-1]], feed_dict[labels[k-1]], feed_dict[split_indices[k-1]]
                    label=np.concatenate([label,feed_dict[labels[k-1]]],axis=0)
            feed_dict[is_training]=False

            logits=sess.run(net.ops['logits'],feed_dict=feed_dict)
            preds=np.argmax(logits,axis=1)
            test_correct_num+=np.sum(preds==label)

        log_info('epoch {}: test accuracy {}'.format(i,test_correct_num/float(test_reader.total_size)))
        saver.save(sess,model_dir+'/epoch{}.ckpt'.format(i))

if __name__=="__main__":
    train_split()