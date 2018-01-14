import tensorflow as tf
from tensorflow.contrib.framework import arg_scope


def inference(feats,is_training,num_classes,reuse=False):
    normalizer_params={'scale':False,'is_training':is_training,'reuse':reuse}
    with arg_scope([tf.contrib.layers.fully_connected],
                   activation_fn=tf.nn.relu,
                   normalizer_fn=tf.contrib.layers.batch_norm,
                   # normalizer_params={'scale':False,'is_training':is_training,'reuse':reuse},
                   reuse=reuse
                   ):

        normalizer_params['scope']='fc1/bn'
        fc1=tf.contrib.layers.fully_connected(feats,64,scope='fc1',normalizer_params=normalizer_params)

        normalizer_params['scope']='fc2/bn'
        fc2=tf.contrib.layers.fully_connected(fc1,128,scope='fc2',normalizer_params=normalizer_params)

        normalizer_params['scope']='fc3/bn'
        fc3=tf.contrib.layers.fully_connected(fc2,128,scope='fc3',normalizer_params=normalizer_params)

        normalizer_params['scope']='fc4/bn'
        fc4=tf.contrib.layers.fully_connected(fc3,256,scope='fc4',normalizer_params=normalizer_params)

        normalizer_params['scope']='fc5/bn'
        fc4=tf.contrib.layers.dropout(fc4,0.7,is_training=is_training)
        logits=tf.contrib.layers.fully_connected(fc4,num_classes,activation_fn=None,normalizer_fn=None,scope='fc5')

        # print tf.trainable_variables()

    return logits


if __name__=="__main__":
    pls={}
    pls['feats'] = tf.placeholder(tf.float32, [None, 39], 'feats')
    # pls['labels'] = tf.placeholder(tf.int64, [None, ], 'labels')
    pls['is_training'] = tf.placeholder(tf.bool, [], 'is_training')
    inference(pls['feats'],pls['is_training'],13)
