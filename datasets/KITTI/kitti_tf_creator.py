# Original Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================


r""" 
Convert raw KITTI detection dataset to TFRecord for object_detection.
Converts KITTI detection dataset to TFRecords with a standard format allowing
  to use this dataset to train object detectors. The raw dataset can be
  downloaded from:
  http://kitti.is.tue.mpg.de/kitti/data_object_image_2.zip.
  http://kitti.is.tue.mpg.de/kitti/data_object_label_2.zip
  Permission can be requested at the main website.
  KITTI detection dataset contains 7481 training images. Using this code with
  the default settings will set aside the first 500 images as a validation set.
  This can be altered using the flags, see details below.
Example usage:
    python object_detection/dataset_tools/create_kitti_tf_record.py \
        --data_dir=/home/user/kitti \
        --output_path=/home/user/kitti.record
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import os
import random
random.seed(42) # Fixed for validation TFRecord

import numpy as np
import PIL.Image as pil
import tensorflow as tf

from object_detection.utils import dataset_util


def label_map_extractor(label_map_path):
  """ Extract a dictionary with class labels and IDs from txt file """
  ids = []
  names = []
  label_map = {}
  with open(label_map_path, 'r') as file:
      lines = file.readlines()
      for line in lines:
          if 'name' in line:
            names.append(line.split(":")[1].strip().strip("'"))             
          elif 'id' in line:
            ids.append(int(line.split(":")[1].strip()))
  for i in range(len(ids)): label_map[names[i]] = ids[i]
  return label_map


def generate_val_indices(data_dir, images, split = 0.2):
  """ Generate random indices for the val set and save to txt file """
  validation_set_size = int(len(os.listdir(data_dir+"/image_2/")) * split)  # Number of images to be used as a validation set.
  validation_set_indexes = random.sample(range(0, len(images)), validation_set_size)
  with open('/app/efficientdet_uncertainty/datasets/KITTI/vaL_index_list.txt', 'w') as file:
      for index in validation_set_indexes:
          file.write(f"{index}\n")


def kitti_active_tfrecords(data_dir, output_path, classes_to_use,
                               label_map_path, train_indices, current_iteration, train=True):
    """Active learning TF creator used in src/active_learning_loop.py. 
    Args:
      data_dir: The full path to the unzipped folder containing the unzipped data.
      output_path: The path to which TFRecord files will be written. 
      classes_to_use: List of strings naming the classes for which data should be
        converted.
      label_map_path: Path to label map proto.
      train_indices: Image indices to select for TFRecord.
      current_iteration: Save name with current active learning iteration number.
      train: Sets if its the validation or training TFRecord.
    """
    label_map_dict = label_map_extractor(label_map_path)
    train_count = 0
    annotation_dir = os.path.join(data_dir,
                                  'label_2')
    image_dir = os.path.join(data_dir,
                             'image_2')
    if train:
      train_writer = tf.io.TFRecordWriter(output_path + '/_train_'+str(current_iteration)+'.tfrecord')
    else:
      train_writer = tf.io.TFRecordWriter(output_path + '/_val_'+str(current_iteration)+'.tfrecord')

    images = np.asarray(sorted(tf.io.gfile.listdir(image_dir)))[train_indices]

    img_num = 0
    for img_name in images:
        img_name_without_ext = img_name.split('.')[0]
        img_anno = read_annotation_file(os.path.join(annotation_dir,
                                                     img_name_without_ext + '.txt'))

        image_path = os.path.join(image_dir, img_name)
        annotation_for_image = filter_annotations(img_anno, classes_to_use)

        example = prepare_example(image_path, annotation_for_image, label_map_dict)
        train_writer.write(example.SerializeToString())
        train_count += 1
        img_num += 1

    train_writer.close()
    print("Finished creating tfrecord")
    print("training images:", train_count)
    return train_count


def convert_kitti_to_tfrecords(data_dir, output_path, classes_to_use,
                               label_map_path, validation_indices):
    """Convert the KITTI detection dataset to TFRecords.
    Args:
      data_dir: The full path to the unzipped folder containing the unzipped data
        from data_object_image_2 and data_object_label_2.zip.
        Folder structure is assumed to be: data_dir/training/label_2 (annotations)
        and data_dir/data_object_image_2/training/image_2 (images).
      output_path: The path to which TFRecord files will be written. The TFRecord
        with the training set will be located at: <output_path>_train.tfrecord
        And the TFRecord with the validation set will be located at:
        <output_path>_val.tfrecord
      classes_to_use: List of strings naming the classes for which data should be
        converted. Use the same names as presented in the KIITI README file.
        Adding dontcare class will remove all other bounding boxes that overlap
        with areas marked as dontcare regions.
      label_map_path: Path to label map proto
      validation_indices: Which images should be used in the validation set.
    """
    label_map_dict = label_map_extractor(label_map_path)
    train_count = 0
    val_count = 0

    annotation_dir = os.path.join(data_dir,
                                  'label_2')

    image_dir = os.path.join(data_dir,
                             'image_2')

    train_writer = tf.io.TFRecordWriter('%s_train.tfrecord' %
                                        output_path)
    val_writer = tf.io.TFRecordWriter('%s_val.tfrecord' %
                                      output_path)

    images = sorted(tf.io.gfile.listdir(image_dir))

    img_num = 0
    for img_name in images:
        img_name_without_ext = img_name.split('.')[0]
        is_validation_img = img_num in validation_indices
        img_anno = read_annotation_file(os.path.join(annotation_dir,
                                                     img_name_without_ext + '.txt'))

        image_path = os.path.join(image_dir, img_name)
        annotation_for_image = filter_annotations(img_anno, classes_to_use)

        example = prepare_example(image_path, annotation_for_image, label_map_dict)
        if is_validation_img:
            val_writer.write(example.SerializeToString())
            val_count += 1
        else:
            train_writer.write(example.SerializeToString())
            train_count += 1
        img_num += 1

    train_writer.close()
    val_writer.close()
    print("finished creating tfrecords")
    print("training images:", train_count)
    print("val images:", val_count)


def prepare_example(image_path, annotations, label_map_dict):
    """Converts a dictionary with annotations for an image to tf.Example proto.
    Args:
      image_path: The complete path to image.
      annotations: A dictionary representing the annotation of a single object
        that appears in the image.
      label_map_dict: A map from string label names to integer ids.
    Returns:
      example: The converted tf.Example.
    """
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = pil.open(encoded_png_io)
    image = np.asarray(image)

    key = hashlib.sha256(encoded_png).hexdigest()

    width = int(image.shape[1])
    height = int(image.shape[0])
    try:
        xmin_norm = annotations['2d_bbox_left'] / float(width)
        ymin_norm = annotations['2d_bbox_top'] / float(height)
        xmax_norm = annotations['2d_bbox_right'] / float(width)
        ymax_norm = annotations['2d_bbox_bottom'] / float(height)
    except KeyError:
        print(' without objects!')

    difficult_obj = [0] * len(xmin_norm)
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(image_path[-10:].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(image_path[-10:-4].lstrip('0').encode('utf8') or '0'.encode('utf8')),
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
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
    }))

    return example


def filter_annotations(img_all_annotations, used_classes):
    """Filters out annotations from the unused classes and dontcare regions.
    Filters out the annotations that belong to classes we do now wish to use and
    (optionally) also removes all boxes that overlap with dontcare regions.
    Args:
      img_all_annotations: A list of annotation dictionaries. See documentation of
        read_annotation_file for more details about the format of the annotations.
      used_classes: A list of strings listing the classes we want to keep, if the
      list contains "dontcare", all bounding boxes with overlapping with dont
      care regions will also be filtered out.
    Returns:
      img_filtered_annotations: A list of annotation dictionaries that have passed
        the filtering.
    """
    dont_care_indices = [i for i, x in enumerate(img_all_annotations['type']) if x not in used_classes]

    if dont_care_indices:
        boxes_to_remove = []
        for i in range(len(img_all_annotations['type'])):
            if i in dont_care_indices:
                boxes_to_remove.append(True)
            else:
                boxes_to_remove.append(False)

        for key in img_all_annotations.keys():
            img_all_annotations[key] = (
                img_all_annotations[key][np.logical_not(boxes_to_remove)])

    return img_all_annotations


def read_annotation_file(filename):
    """Reads a KITTI annotation file.
    Converts a KITTI annotation file into a dictionary containing all the
    relevant information.
    Args:
      filename: the path to the annotataion text file.
    Returns:
      anno: A dictionary with the converted annotation information. See annotation
      README file for details on the different fields.
    """
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip().split(' ') for x in content]

    anno = {}
    anno['type'] = np.array([x[0].lower() for x in content])
    anno['truncated'] = np.array([float(x[1]) for x in content])
    anno['occluded'] = np.array([int(x[2]) for x in content])
    anno['alpha'] = np.array([float(x[3]) for x in content])

    anno['2d_bbox_left'] = np.array([float(x[4]) for x in content])
    anno['2d_bbox_top'] = np.array([float(x[5]) for x in content])
    anno['2d_bbox_right'] = np.array([float(x[6]) for x in content])
    anno['2d_bbox_bottom'] = np.array([float(x[7]) for x in content])

    return anno


if __name__ == "__main__":
  data_dir = "/app/efficientdet_uncertainty/datasets/KITTI/training"
  output_path = "/app/efficientdet_uncertainty/datasets/KITTI/tf/"  
  classes_to_use = ["car","van","truck","pedestrian","person_sitting","cyclist","tram"] # Not capital
  label_map_path = "/app/efficientdet_uncertainty/datasets/KITTI/kitti.pbtxt"  # Path to label map
  val_indices = []
  with open('/app/efficientdet_uncertainty/datasets/KITTI/vaL_index_list.txt', 'r') as file:
      for line in file:
          val_indices.append(int(line.strip()))
  print("Validation set size:", len(val_indices))

  convert_kitti_to_tfrecords(
      data_dir=data_dir,
      output_path=output_path,
      classes_to_use=classes_to_use,
      label_map_path=label_map_path,
      validation_indices=val_indices)


