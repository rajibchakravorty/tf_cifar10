import os
import numpy as np
import cPickle
import random

from os.path import join
from skimage.io import imread, imsave
from skimage import color
import sys
sys.path.append( '../tf_utility' )

from prep_tfrecords import prep_tfrecord

do_save = True

def get_file_list_label( folder, label ):

    all_files = list( os.walk( folder ) )

    all_files = all_files[0][2]

    all_files = [ os.path.join( folder, x ) for x in all_files ]

    all_files = test_and_resave(all_files)

    file_list = [(x,label) for x in all_files ]

    return file_list

def test_and_resave( file_list ):

    rectified_list = list()

    for f in file_list:

        try:
            im = imread( f )

            if im is None:
                continue

            elif len( im.shape ) == 2:
                if do_save:
                    im_rectified = color.gray2rgb( im, None )
                    imsave( f, im_rectified )
                print f, im.shape
                #continue

            elif len( im.shape ) > 3:
                print '>3', f, im.shape
                continue

            elif im.shape[2] != 3:
                if do_save:
                    im_rectified = im[:,:,0:3]
                    imsave( f, im_rectified )
                print f, im.shape
                #continue
            else:
                if do_save:
                    imsave( f, im )

            rectified_list.append( f )
        except UserWarning:
            print f
            continue
        except:
            continue

    print 'Started with {0}, final list {1}'.format( len( file_list), len( rectified_list) )

    return rectified_list

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)

    images = dict['data']
    labels = dict['labels']

    return images, labels

def get_batch_data_lists( data_file, batch, image_path ):

    images, labels = unpickle( data_file )

    data_list = list()
    for idx, im in enumerate( images ):

        label = labels[idx]

        name = 'im_{0}_{1}_{2}.jpg'.format(batch, idx, label)

        image_file = join( image_path, name )

        im_rearranged = np.zeros( (32,32,3))
        im_rearranged[:,:,0] = np.reshape( im[0:1024], (32,32) )
        im_rearranged[:, :, 1] = np.reshape(im[1024:2048], (32, 32))
        im_rearranged[:, :, 2] = np.reshape(im[2048:], (32, 32))

        if np.max( im_rearranged ) > 1.:
            im_rearranged = im_rearranged/255.

        data_list.append ((image_file, label ) )

        if do_save:
            imsave( image_file, im_rearranged )

    return data_list

if __name__ == '__main__':

    data_location = '/opt/ml_data/cifar/cifar-10-batches-py'


    image_location = '/opt/ml_data/cifar/cifar-10/images/train'

    ##batch_1
    batch_file = join( data_location, 'data_batch_1' )
    batch_1_list = get_batch_data_lists(batch_file, 1, image_location)

    batch_file = join(data_location, 'data_batch_2')
    batch_2_list = get_batch_data_lists(batch_file, 2, image_location)

    batch_file = join(data_location, 'data_batch_3')
    batch_3_list = get_batch_data_lists(batch_file, 3, image_location)

    batch_file = join(data_location, 'data_batch_4')
    batch_4_list = get_batch_data_lists(batch_file, 4, image_location)

    batch_file = join(data_location, 'data_batch_5')
    batch_5_list = get_batch_data_lists(batch_file, 5, image_location)

    train_list = batch_1_list+batch_2_list+batch_3_list+batch_4_list+batch_5_list

    random.shuffle( train_list )

    valid_list = train_list[-5000:]
    train_list = train_list[0:-5000]

    batch_file = join(data_location, 'test_batch')
    test_list = get_batch_data_lists(batch_file, 'test', image_location)




    tfrecord_destfile = join( '/opt/ml_data/cifar/cifar-10', 'train.tfrecords' )
    prep_tfrecord( train_list, tfrecord_destfile )

    tfrecord_destfile = join( '/opt/ml_data/cifar/cifar-10', 'valid.tfrecords')
    prep_tfrecord(valid_list, tfrecord_destfile )

    tfrecord_destfile = join( '/opt/ml_data/cifar/cifar-10', 'test.tfrecords')
    prep_tfrecord(test_list, tfrecord_destfile )
