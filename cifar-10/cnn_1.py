
import numpy as np
import tensorflow as tf

import tensorflow.contrib as contrib

def cnn_archi( images ):

    with tf.name_scope( 'conv1_1' ) as scope:
        conv1_1 = tf.layers.conv2d( images, filters = 48,
                                    kernel_size = [3,3],
                                    padding = 'same',
                                    activation = tf.nn.relu)

    with tf.name_scope('conv1_1') as scope:
        conv1_2 = tf.layers.conv2d(conv1_1, filters=48,
                                   kernel_size=[3, 3],
                                   padding='same',
                                   activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(conv1_2, pool_size= (2,2), strides=(2,2), padding='same')


    with tf.name_scope( 'conv2_1' ) as scope:
        conv2_1 = tf.layers.conv2d( pool1, filters = 96,
                                    kernel_size = [3,3],
                                    padding = 'same',
                                    activation = tf.nn.relu)

    with tf.name_scope( 'conv2_2' ) as scope:
        conv2_2 = tf.layers.conv2d( conv2_1, filters = 96,
                                    kernel_size = [3,3],
                                    padding = 'same',
                                    activation = tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(conv2_2, pool_size=(2, 2), strides=(2,2), padding='same' )


    with tf.name_scope( 'conv3_1' ) as scope:
        conv3_1 = tf.layers.conv2d( pool2, filters = 192,
                                    kernel_size = [3,3],
                                    padding = 'same',
                                    activation = tf.nn.relu)

    with tf.name_scope( 'conv3_2' ) as scope:
        conv3_2 = tf.layers.conv2d( conv3_1, filters = 192,
                                    kernel_size = [2,2],
                                    padding = 'same',
                                    activation = tf.nn.relu)
                                    

    pool3 = tf.layers.max_pooling2d(conv3_2, pool_size=(2, 2), strides=(2,2) )

    '''
    with tf.name_scope( 'conv4_1' ) as scope:
        conv4_1 = tf.layers.conv2d( pool3_1, filters = 128,
                                    kernel_size = [2,2],
                                    padding = 'valid',
                                    activation = tf.nn.relu)
    pool4_1 = tf.layers.max_pooling2d(conv4_1, pool_size=(2, 2), strides=(1,1) )

    with tf.name_scope( 'conv5_1' ) as scope:
        conv5_1 = tf.layers.conv2d( pool4_1, filters = 128,
                                    kernel_size = [2,2],
                                    padding = 'valid',
                                    activation = tf.nn.relu)
    pool5_1 = tf.layers.max_pooling2d(conv5_1, pool_size=(2, 2), strides=(1,1) )

    with tf.name_scope( 'conv6_1' ) as scope:
        conv6_1 = tf.layers.conv2d( pool5_1, filters = 128,
                                    kernel_size = [2,2],
                                    padding = 'valid',
                                    activation = tf.nn.relu)
    pool6_1 = tf.layers.max_pooling2d(conv6_1, pool_size=(2, 2), strides=(1,1) )

    with tf.name_scope( 'conv7_1' ) as scope:
        conv7_1 = tf.layers.conv2d( pool6_1, filters = 128,
                                    kernel_size = [2,2],
                                    padding = 'valid',
                                    activation = tf.nn.relu)
    pool7_1 = tf.layers.max_pooling2d(conv7_1, pool_size=(2, 2), strides=(1,1) )
    '''
    # fc1
    with tf.name_scope('fc1') as scope:

        norm1_flat = tf.layers.flatten( pool3 )
        #dropout = tf.layers.dropout( norm1_flat, rate = 0.5 )
        fc1_out = tf.layers.dense( norm1_flat, units= 512, activation = tf.nn.relu)

    ## fc1
    with tf.name_scope('fc2') as scope:
        #dropout = tf.layers.dropout( fc1_out, rate = 0.5 )
        fc2_out = tf.layers.dense( fc1_out, units= 256, activation = tf.nn.relu)

    ## fc1
    with tf.name_scope('fc3') as scope:
        #dropout = tf.layers.dropout( fc2_out, rate = 0.5 )
        fc3_out = tf.layers.dense( fc2_out, units= 10 )

    return fc3_out