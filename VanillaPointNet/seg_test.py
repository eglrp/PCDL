from seg_network import ContextSegmentationNetwork
from seg_provider import *
import tensorflow as tf
import matplotlib.pyplot as plt
from s3dis_util.util import read_pkl,get_class_names
from sklearn.metrics import confusion_matrix
from activation import plot_confusion_matrix


def declare_context_network(model_path):
    input_dim=6
    num_classes=14
    local_feat_dim=33

    inputs={}
    net=ContextSegmentationNetwork(input_dim,num_classes,False,local_feat_dim,512)
    inputs['global_pts']=(tf.placeholder(tf.float32, [None, input_dim]))
    inputs['global_indices']=(tf.placeholder(tf.int64,[None,None]))
    inputs['context_pts']=(tf.placeholder(tf.float32, [None, input_dim]))
    inputs['context_batch_indices']=(tf.placeholder(tf.int64,[None]))
    inputs['context_block_indices']=(tf.placeholder(tf.int64,[None,None]))
    inputs['local_feats']=(tf.placeholder(tf.float32,[None,None,local_feat_dim]))
    inputs['is_training']=tf.placeholder(tf.bool,name='is_training')


    net.inference(inputs['global_pts'],inputs['global_indices'],
                  inputs['local_feats'],inputs['context_pts'],
                  inputs['context_batch_indices'],
                  inputs['context_block_indices'],
                  'cpu_',inputs['is_training'])

    config=tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess=tf.Session(config=config)
    saver=tf.train.Saver()
    saver.restore(sess,model_path)

    return sess,net,inputs

def test_feature_gradients():
    sess,net,inputs=declare_context_network('model/context_overfit/epoch280.ckpt')

    logits_grads=tf.placeholder(tf.float32,[None,None,14])

    feats_grads=tf.gradients(net.ops['inference_logits'],net.ops['feats'],logits_grads)

    train_fs,test_fs=get_train_test_split()
    file_idxs=np.random.randint(0,len(test_fs),10)
    feats_grads_vals_list=[]
    for idx in file_idxs:
        block_list,room_sample_data=read_pkl('data/S3DIS/train_v2_more/'+test_fs[idx]+'.pkl')

        feed_dict={}
        block_idx=np.random.randint(0,len(block_list))
        block=block_list[block_idx]
        feed_dict[inputs['global_pts']]=room_sample_data
        feed_dict[inputs['context_pts']]=block['cont']
        feed_dict[inputs['context_batch_indices']]=np.asarray([0],dtype=np.int64)
        feed_dict[inputs['context_block_indices']]=np.expand_dims(block['cont_index'],axis=0)
        feed_dict[inputs['global_indices']]=np.expand_dims(block['room_index'],axis=0)
        feed_dict[inputs['local_feats']]=np.expand_dims(block['feat'],axis=0)
        feed_dict[inputs['is_training']]=False
        logits_grads_val=np.zeros([feed_dict[inputs['local_feats']].shape[0],
                                  feed_dict[inputs['local_feats']].shape[1],14],
                                  dtype=np.float32)
        logits_grads_val[:,:,13]=1.0
        feed_dict[logits_grads]=logits_grads_val

        feats_grads_vals=sess.run(feats_grads,feed_dict)
        feats_grads_vals_list.append(np.abs(feats_grads_vals[0]))
        print feats_grads_vals[0].shape

    feats_grads_vals_list=np.concatenate(feats_grads_vals_list,axis=0)
    feats_grads_means=np.mean(feats_grads_vals_list,axis=(0,1))
    plt.bar(xrange(len(feats_grads_means)),feats_grads_means)
    plt.show()


if __name__=="__main__":
    test_feature_gradients()