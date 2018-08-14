import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
import matplotlib.pyplot as plot
import cv2
slim_example_decoder = tf.contrib.slim.tfexample_decoder
path = 'D:/data/data/record/mydata_val.tfrecord'
with tf.Session() as sess:
    feature = {

        'image/encoded':tf.FixedLenFeature([], tf.string),
        'image/filename':tf.FixedLenFeature([], tf.string),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),

        'image/object/bbox/xmin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(tf.float32)

    }
    filename_queue = tf.train.string_input_producer([path])
    reader = tf.TFRecordReader()
    _,serialized_example= reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features=feature)
    image_path = features['image/filename']
    image_h = tf.cast(features['image/height'],tf.int32)
    image_w = tf.cast(features['image/width'],tf.int32)


    image_shape = tf.parallel_stack([image_h, image_w, 3])

    # 下面的方式会导致解析的维度reshape对应不上
    # image = tf.decode_raw(features['image/encoded'], tf.uint8)
    img = features['image/encoded']
    image = tf.image.decode_png(img, channels=3)


    image1 = tf.reshape(image, image_shape)

    xmin = tf.cast(features['image/object/bbox/xmin'],tf.float32)
    xmax = tf.cast(features['image/object/bbox/xmax'], tf.float32)
    ymin = tf.cast(features['image/object/bbox/ymin'], tf.float32)
    ymax = tf.cast(features['image/object/bbox/ymax'], tf.float32)

    # image_path, xmin, xmax, ymin,ymax = tf.train.batch([image_path, xmin, xmax, ymin,ymax],
    #                                                                batch_size=1
    #                                                              )
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10):
        path, x_min, y_min, x_max, y_max,image_height ,image_width,image_2= sess.run(
            [image_path, xmin, ymin, xmax, ymax,image_h,image_w,image1])
        print(path)
        print(x_min)
        print(y_min)
        print(x_max)
        print(y_max)
        print(image_height)
        print(image_width)

        print('-----------------------------')
        # cv2.imshow('image',image_2)
        # cv2.waitKey(20000)



    coord.request_stop()
    coord.join(threads)

