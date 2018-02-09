import tensorflow as tf
import tensorflow.contrib.framework as framework


def segmentation_classifier(feats, num_classes, reuse=False):
    '''

    :param feats: n,k,f
    :param num_classes:
    :param reuse:
    :return:
    '''
    feats=tf.expand_dims(feats,axis=2) # n,k,1,f
    with tf.name_scope('segmentation_classifier'):
        with framework.arg_scope([tf.contrib.layers.conv2d],kernel_size=[1,1],stride=1,
                                 padding='VALID',activation_fn=tf.nn.relu,reuse=reuse,
                                 ):
            class_mlp1 = tf.contrib.layers.conv2d(feats, num_outputs=1024, scope='class_mlp1')
            class_mlp2 = tf.contrib.layers.conv2d(class_mlp1, num_outputs=512, scope='class_mlp2')
            class_mlp3 = tf.contrib.layers.conv2d(class_mlp2, num_outputs=256, scope='class_mlp3')
            class_mlp4 = tf.contrib.layers.conv2d(class_mlp3, num_outputs=256, scope='class_mlp4')
            class_mlp5 = tf.contrib.layers.conv2d(class_mlp4, num_outputs=128, scope='class_mlp5')
            logits = tf.contrib.layers.conv2d(class_mlp5, num_outputs=num_classes, scope='class_mlp6',activation_fn=None)

        logits=tf.squeeze(logits,axis=2,name='logits')

    return logits


def segmentation_classifier_v2(feats, point_feats, is_training, num_classes, reuse=False, use_bn=True):
    '''

    :param feats: n,k,f
    :param point_feats:
    :param is_training:
    :param num_classes:
    :param reuse:
    :return:
    '''
    normalizer_params={'scale':False,'is_training':is_training,'reuse':reuse}
    feats=tf.expand_dims(feats,axis=2)              # n,k,1,2048+6
    point_feats=tf.expand_dims(point_feats,axis=2)  # n,k,1,6
    bn=tf.contrib.layers.batch_norm if use_bn else None
    with tf.name_scope('segmentation_classifier'):
        with framework.arg_scope([tf.contrib.layers.conv2d],kernel_size=[1,1],stride=1,
                                 padding='VALID',activation_fn=tf.nn.relu,reuse=reuse,
                                 normalizer_fn=bn):

            normalizer_params['scope']='class_mlp1_bn'
            class_mlp1 = tf.contrib.layers.conv2d(
                feats, num_outputs=512, scope='class_mlp1',normalizer_params=normalizer_params)
            class_mlp1=tf.concat([class_mlp1,point_feats],axis=3)

            normalizer_params['scope']='class_mlp2_bn'
            class_mlp2 = tf.contrib.layers.conv2d(
                class_mlp1, num_outputs=256, scope='class_mlp2',normalizer_params=normalizer_params)
            class_mlp2=tf.concat([class_mlp2,point_feats],axis=3)
            # tf.cond(is_training,lambda:tf.nn.dropout(class_mlp2,0.7),lambda:class_mlp2)

            logits = tf.contrib.layers.conv2d(
                class_mlp2, num_outputs=num_classes, scope='class_mlp3',activation_fn=None,normalizer_fn=None)

        logits=tf.squeeze(logits,axis=2,name='logits')

    return logits


def model_classifier(feats, num_classes, reuse=False, is_training=tf.constant(False)):
    '''
    :param feats: n,f
    :param num_classes:
    :param is_training:
    :param reuse:
    :return:
    '''
    with tf.name_scope('segmentation_classifier'):
        with framework.arg_scope([tf.contrib.layers.fully_connected],activation_fn=tf.nn.relu,reuse=reuse):
            fc1=tf.contrib.layers.fully_connected(feats,num_outputs=512,scope='class_fc1')
            fc1 = tf.cond(is_training,
                          lambda: tf.nn.dropout(fc1, 0.7),
                          lambda: fc1)
            fc2=tf.contrib.layers.fully_connected(fc1,num_outputs=512,scope='class_fc2')
            fc3=tf.contrib.layers.fully_connected(fc2,num_outputs=256,scope='class_fc3')
            fc4=tf.contrib.layers.fully_connected(fc3,num_outputs=256,scope='class_fc4')
            fc4=tf.cond(is_training,
                        lambda: tf.nn.dropout(fc4,0.7),
                        lambda: fc4)

            logits=tf.contrib.layers.fully_connected(fc4,num_outputs=num_classes,scope='class_fc5',activation_fn=None)  # n, num_classes

    return logits


def model_classifier_v2(feats, num_classes, is_training, reuse=False, use_bn=True):
    '''
    :param feats: n,f
    :param num_classes:
    :param is_training:
    :param reuse:
    :return:
    '''
    normalizer_params={'scale':False,'is_training':is_training,'reuse':reuse}
    bn=tf.contrib.layers.batch_norm if use_bn else None
    with tf.name_scope('model_classifier'):
        with framework.arg_scope([tf.contrib.layers.fully_connected],
                                 activation_fn=tf.nn.relu,reuse=reuse,
                                 normalizer_fn=bn):

            normalizer_params['scope']='class_fc1_bn'
            fc1=tf.contrib.layers.fully_connected(
                feats,num_outputs=512,scope='class_fc1',normalizer_params=normalizer_params)
            # fc1=tf.cond(is_training,lambda: tf.nn.dropout(fc1,0.7),lambda: fc1)

            normalizer_params['scope']='class_fc2_bn'
            fc2=tf.contrib.layers.fully_connected(
                fc1,num_outputs=256,scope='class_fc2',normalizer_params=normalizer_params)
            fc2=tf.cond(is_training,lambda: tf.nn.dropout(fc2,0.7),lambda: fc2)

            logits=tf.contrib.layers.fully_connected(
                fc2,num_outputs=num_classes,scope='class_fc3',activation_fn=None,normalizer_fn=None)

    return logits