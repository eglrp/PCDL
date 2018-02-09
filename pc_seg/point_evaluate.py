from point_train import *
from s3dis.draw_util import output_points

if __name__=="__main__":
    model_path='model/point_wise_mlp_s3dis/model50_0.625733369338.ckpt'
    train_list, test_list = get_train_test_split()

    pls={}
    pls['feats']=tf.placeholder(tf.float32,[None,39],'feats')
    # pls['labels']=tf.placeholder(tf.int64,[None,],'labels')
    pls['is_training']=tf.placeholder(tf.bool,[],'is_training')

    logits=inference(pls['feats'],pls['is_training'],13)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    saver = tf.train.Saver(max_to_keep=500)
    what=saver.restore(sess,model_path)

    path='data/S3DIS/point/fpfh/'
    print 'miou aiou macc oacc'
    colors = get_class_colors()
    all_labels=[]
    all_preds=[]
    for fn in test_list:
        feats,labels=read_points_feats(path+fn+'.h5')
        logit_vals=sess.run(logits,feed_dict={pls['feats']:feats,pls['is_training']:False})
        preds=np.argmax(logit_vals,axis=1)
        output_points(fn+'_true.txt',feats[:,:3],colors[labels,:])
        output_points(fn+'_pred.txt',feats[:,:3],colors[preds,:])

        all_labels.append(labels)
        all_preds.append(preds)

    all_labels=np.concatenate(all_labels,axis=0)
    all_preds=np.concatenate(all_preds,axis=0)

    get_class_names()

    iou,miou,oiou,acc,macc,oacc=compute_iou(all_labels,all_preds)
    print 'miou {} oiou {} macc {} oacc {}'.format(miou,oiou,macc,oacc)
    for i,name in enumerate(get_class_names()):
        print '{} iou: {}'.format(name,iou[i])




