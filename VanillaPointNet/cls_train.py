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

if __name__=="__main__":
    train()