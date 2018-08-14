import os
import tensorflow as tf

from object_detection.core import standard_fields as fields
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)
slim = tf.contrib.slim
import cv2

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [32 x 32 x 3] color image.',
    'label': 'A single integer between 0 and 9',
}
slim_example_decoder = tf.contrib.slim.tfexample_decoder


@tf.contrib.framework.deprecated(None, 'Use object_detection/model_main.py.')
def main(_):

    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/key/sha256':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=1),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=1),
        # Image-level labels.
        'image/class/text':
            tf.VarLenFeature(tf.string),
        'image/class/label':
            tf.VarLenFeature(tf.int64),
        # Object boxes and classes.
        'image/object/bbox/xmin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(tf.int64),
        'image/object/class/text':
            tf.VarLenFeature(tf.string),
        'image/object/area':
            tf.VarLenFeature(tf.float32),
        'image/object/is_crowd':
            tf.VarLenFeature(tf.int64),
        'image/object/difficult':
            tf.VarLenFeature(tf.int64),
        'image/object/group_of':
            tf.VarLenFeature(tf.int64),
        'image/object/weight':
            tf.VarLenFeature(tf.float32),
    }
    image = slim_example_decoder.Image(
        image_key='image/encoded', format_key='image/format', channels=3)

    items_to_handlers = {
        fields.InputDataFields.image:
            image,
        fields.InputDataFields.source_id: (
            slim_example_decoder.Tensor('image/source_id')),
        fields.InputDataFields.key: (
            slim_example_decoder.Tensor('image/key/sha256')),
        fields.InputDataFields.filename: (
            slim_example_decoder.Tensor('image/filename')),
        # Object boxes and classes.
        fields.InputDataFields.groundtruth_boxes: (
            slim_example_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'],
                                             'image/object/bbox/')),
        fields.InputDataFields.groundtruth_area:
            slim_example_decoder.Tensor('image/object/area'),
        fields.InputDataFields.groundtruth_is_crowd: (
            slim_example_decoder.Tensor('image/object/is_crowd')),
        fields.InputDataFields.groundtruth_difficult: (
            slim_example_decoder.Tensor('image/object/difficult')),
        fields.InputDataFields.groundtruth_group_of: (
            slim_example_decoder.Tensor('image/object/group_of')),

        fields.InputDataFields.groundtruth_weights: (
            slim_example_decoder.Tensor('image/object/weight')),

        'height': (
            slim_example_decoder.Tensor('image/height')),

        'width': (
            slim_example_decoder.Tensor('image/width')),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)


    dataset = slim.dataset.Dataset(
    data_sources='D:/data/data/record/mydata_val.tfrecord',
    reader=tf.TFRecordReader,
    decoder=decoder,
    num_samples=500, items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)

    # print(dataset)


    provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=1,
            common_queue_capacity=20 * 2,
            common_queue_min=10 * 2,
            shuffle=True)
    # print(provider)


    [image,groundtruth_boxes,height,width,image_path] = provider.get(['image','groundtruth_boxes','height','width','filename'])

    image2 = image
    with tf.Session() as sess:
        # image = tf.train.batch(image, batch_size=1)

        # image1 = tf.image.resize_images(image, [1000, 1000])
        # image2 = tf.train.batch(
        #     reshape_list([image1]),
        #     batch_size=1,
        #     num_threads=2,
        #     capacity=5 * 1)


        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10):

            image3 ,boxes,image_path2,image_height,image_width= sess.run([image2,groundtruth_boxes,image_path,height,width])
            print(image_path2)

            print(np.shape(boxes))
            print(boxes[:, 0] * image_height)
            print(boxes[:, 1] * image_width)
            print(boxes[:, 2] * image_height)
            print(boxes[:, 3] * image_width)
            cv2.imshow('image', image3)
            cv2.waitKey(100000)

        coord.request_stop()
        coord.join(threads)



if __name__ == "__main__":
    tf.app.run()
