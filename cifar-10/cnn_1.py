

import tensorflow as tf

def cnn_archi( images ):

    with tf.name_scope( 'conv1_1' ) as scope:
        conv1_1 = tf.layers.conv2d( images, filters = 64,
                                    kernel_size = [5,5],
                                    padding = 'same',
                                    activation = tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1_1, pool_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same')

    with tf.name_scope('conv1_1') as scope:
        conv1_2 = tf.layers.conv2d(pool1, filters=64,
                                   kernel_size=[5, 5],
                                   padding='same',
                                   activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv1_2, pool_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same')


    # fc1
    with tf.name_scope('fc1') as scope:

        norm1_flat = tf.layers.flatten( pool2 )
        #dropout = tf.layers.dropout( norm1_flat, rate = 0.5 )
        fc1_out = tf.layers.dense( norm1_flat, units= 384, activation = tf.nn.relu)

    ## fc1
    with tf.name_scope('fc2') as scope:
        #dropout = tf.layers.dropout( fc1_out, rate = 0.5 )
        fc2_out = tf.layers.dense( fc1_out, units= 192, activation = tf.nn.relu)

    ## fc1
    with tf.name_scope('fc3') as scope:
        #dropout = tf.layers.dropout( fc2_out, rate = 0.5 )
        fc3_out = tf.layers.dense( fc2_out, units= 10 )

    return fc3_out