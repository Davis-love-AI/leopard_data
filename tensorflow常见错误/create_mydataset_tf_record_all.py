from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import hashlib
import io
import os
import numpy as np
import PIL.Image as pil
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_detection.utils.np_box_ops import iou
import csv
from six import raise_from
tf.logging.set_verbosity(tf.logging.INFO)
tf.app.flags.DEFINE_string('data_dir', 'D:/data/TF_train_data/test/data', 'Location of root directory for the ' 'data. Folder structure is assumed to be:' '<data_dir>/training/label_2 (annotations) and' '<data_dir>/data_object_image_2/training/image_2' '(images).')
tf.app.flags.DEFINE_string('output_path', 'C:/Users/wyl/Desktop/data/', 'Path to which TFRecord files'
 'will be written. The TFRecord with the training set' 'will be located at: <output_path>_train.tfrecord.' 'And the TFRecord with the validation set will be' 'located at: <output_path>_val.tfrecord')
tf.app.flags.DEFINE_string('classes_to_use', 'pedestrian,car,van,truck,person_sitting,person_sitting,tram,misc', 'Comma separated list of class names that will be' 'used. Adding the dontcare class will remove all' 'bboxs in the dontcare regions.')
tf.app.flags.DEFINE_string('label_map_path','D:/data/project/object_detection_google_0716/kitti_label_map.pbtxt','Path to label map proto.')
tf.app.flags.DEFINE_integer('validation_set_size', '500', 'Number of images to' 'be used as a validation set.')
FLAGS = tf.app.flags.FLAGS

def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_annotations(data_file):
    dir_list = os.listdir(data_file)
    result = {}
    for dir_list_sub in dir_list:
        label_dir = os.path.join(data_file, dir_list_sub, 'label')

        label_file_name = os.listdir(label_dir)
        label_file = os.path.join(label_dir, label_file_name[0])
        with open(label_file, 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            for line, row in enumerate(csv_reader):
                line += 1
                image_name, x1, y1, x2, y2, class_name = row[0:6]
                image_dir = os.path.join(data_file, dir_list_sub, 'image')
                image_path = os.path.join(image_dir, image_name)
                # print(image_path)
                if image_path not in result:
                    result[image_path] = []
                if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                    continue
                x1 = _parse(x1, float, 'line {}: malformed x1: {{}}'.format(line))
                y1 = _parse(y1, float, 'line {}: malformed y1: {{}}'.format(line))
                x2 = _parse(x2, float, 'line {}: malformed x2: {{}}'.format(line))
                y2 = _parse(y2, float, 'line {}: malformed y2: {{}}'.format(line))
                # Check that the bounding box is valid.
                if x2 <= x1:
                    raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
                if y2 <= y1:
                    raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

                    # check if the current class name is correctly present
                result[image_path].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
    return result


def convert_kitti_to_tfrecords(data_dir, output_path, classes_to_use, label_map_path, validation_set_size):
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    train_writer = tf.python_io.TFRecordWriter('%smydata_train.tfrecord' % output_path)
    val_writer = tf.python_io.TFRecordWriter('%smydata_val.tfrecord' % output_path)
    result = _read_annotations(data_dir)
    image_path = list(result.keys())

    for i in range(len(image_path)):
        is_validation_img = i < validation_set_size
        image_sigle_path = image_path[i]
        image_single_label_data = result[image_sigle_path]
        anno = read_annotation_file(image_single_label_data)

        annotation_for_image = filter_annotations(anno, classes_to_use)

        print(image_sigle_path)
        print(annotation_for_image)

        example = prepare_example(image_sigle_path, annotation_for_image, label_map_dict)
        if is_validation_img:
            val_writer.write(example.SerializeToString())
        else:
            train_writer.write(example.SerializeToString())
    train_writer.close()
    val_writer.close()

def prepare_example(image_path, annotations, label_map_dict):
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_png = fid.read()

    print('encoded_png',type(encoded_png))
    encoded_png_io = io.BytesIO(encoded_png)
    image = pil.open(encoded_png_io)
    image = np.asarray(image)
    key = hashlib.sha256(encoded_png).hexdigest()
    width = int(image.shape[1])
    height = int(image.shape[0])
    xmin_norm = annotations['2d_bbox_left'] / float(width)
    ymin_norm = annotations['2d_bbox_top'] / float(height)
    xmax_norm = annotations['2d_bbox_right'] / float(width)
    ymax_norm = annotations['2d_bbox_bottom'] / float(height)
    difficult_obj = [0] * len(xmin_norm)
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(image_path.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(image_path.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
        'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin_norm),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax_norm),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin_norm),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax_norm),
        'image/object/class/text': dataset_util.bytes_list_feature(
            [x.encode('utf8') for x in annotations['type']]),
        'image/object/class/label': dataset_util.int64_list_feature(
            [label_map_dict[x] for x in annotations['type']]),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj)
    }))
    return example

def filter_annotations(img_all_annotations, used_classes):
    img_filtered_annotations = {}
    # Filter the type of the objects.
    relevant_annotation_indices = [
        i for i, x in enumerate(img_all_annotations['type']) if x in used_classes
    ]

    # print('type(relevant_annotation_indices)',type(relevant_annotation_indices))
    # print(relevant_annotation_indices)
    for key in img_all_annotations.keys():
        img_filtered_annotations[key] = (
            img_all_annotations[key][relevant_annotation_indices])


    if 'dontcare' in used_classes:
        dont_care_indices = [i for i,
                                   x in enumerate(img_filtered_annotations['type'])
                             if x == 'dontcare']

        # bounding box format [y_min, x_min, y_max, x_max]
        all_boxes = np.stack([img_filtered_annotations['2d_bbox_top'],
                              img_filtered_annotations['2d_bbox_left'],
                              img_filtered_annotations['2d_bbox_bottom'],
                              img_filtered_annotations['2d_bbox_right']],
                             axis=1)

        ious = iou(boxes1=all_boxes,
                   boxes2=all_boxes[dont_care_indices])

        # Remove all bounding boxes that overlap with a dontcare region.
        if ious.size > 0:
            boxes_to_remove = np.amax(ious, axis=1) > 0.0
            for key in img_all_annotations.keys():
                img_filtered_annotations[key] = (
                    img_filtered_annotations[key][np.logical_not(boxes_to_remove)])

    return img_filtered_annotations


def read_annotation_file(image_single_label_data):
    anno = {}
    anno['type'] = np.array([x['class'].lower() for x in image_single_label_data])
    anno['2d_bbox_left'] = np.array([(float)(x['x1']) for x in image_single_label_data])
    anno['2d_bbox_top'] = np.array([(float)(x['y1']) for x in image_single_label_data])
    anno['2d_bbox_right'] = np.array([(float)(x['x2']) for x in image_single_label_data])
    anno['2d_bbox_bottom'] = np.array([(float)(x['y2']) for x in image_single_label_data])
    return anno


def main(_):
    convert_kitti_to_tfrecords(
        data_dir=FLAGS.data_dir,
        output_path=FLAGS.output_path,
        classes_to_use=FLAGS.classes_to_use.split(','),
        label_map_path=FLAGS.label_map_path,
        validation_set_size=FLAGS.validation_set_size)

if __name__ == '__main__':
    tf.app.run()
