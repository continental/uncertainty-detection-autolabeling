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
""" Create TFRecord with BDD dataset """

import hashlib
import io
import os
import ijson

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


def collect_bdd_images(data_dir):
    """Retrieve Images with actual labels and with the image existing. """
    json_path = os.path.join(data_dir, "labels/bdd100k_labels_images_train.json")
    image_dir = os.path.join(data_dir, "images/100k/train/")
    remaining_images = []
    with open(json_path, 'rb') as json_f:
        items = ijson.items(json_f, 'item')
        for item in items:
            img_name = item['name']
            img_path = os.path.join(image_dir, img_name)
            if os.path.isfile(img_path):
                if 'labels' in item:
                    remaining_images.append(img_path.split("/")[-1])
    with open(data_dir + "/labels/remaining_images.txt", 'w') as file:
        for im in remaining_images:
            file.write(f"{im}\n")
    return np.asarray(remaining_images)


def example_extractor(img_path, item, occurences_percls, label_map_dict, classes_to_use):
    """ Read image and corresponding labels """
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = pil.open(encoded_jpg_io)
    image = np.asarray(image)
    width = int(image.shape[1])
    height = int(image.shape[0])

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    labels = []
    occluded = []
    truncated = []
    current_labels = item['labels']
    for label in current_labels:
        category = label['category']  # Class
        if category in classes_to_use:
            occurences_percls[category] += 1
            labels.append(label_map_dict[category])
            classes.append(category.encode('utf8'))
            attributes = label['attributes']
            occluded.append(int(attributes['occluded'] == 'true'))
            truncated.append(int(attributes['truncated'] == 'true'))
            box2d = label['box2d']
            xmins.append(float(box2d['x1']) / width)
            ymins.append(float(box2d['y1']) / height)
            xmaxs.append(float(box2d['x2']) / width)
            ymaxs.append(float(box2d['y2']) / height)
    extracted_gt = [xmins, ymins, xmaxs, ymaxs, classes, labels, occluded, truncated]
    return encoded_jpg, width, height, extracted_gt


def example_builder(extracted_gt, encoded_jpg, item, height, width, img_name, unique_name):
    """ Build TF example based on given ground truth and image """
    xmins, ymins, xmaxs, ymaxs, classes, labels, occluded, truncated = extracted_gt
    difficult_obj = [0] * len(xmins)
    key = hashlib.sha256(encoded_jpg).hexdigest()
    att = item['attributes']
    weather, scene, timeofday = att['weather'], att['scene'], att['timeofday']
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(img_name.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(unique_name.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes),
        'image/object/class/label': dataset_util.int64_list_feature(labels),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/occluded': dataset_util.int64_list_feature(occluded),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/weather': dataset_util.bytes_feature(weather.encode('utf8')),
        'image/object/scene': dataset_util.bytes_feature(scene.encode('utf8')),
        'image/object/timeofday': dataset_util.bytes_feature(timeofday.encode('utf8')),
    }))
    # print(img_name, 'processed.')
    return tf_example


def bdd_active_tfrecords(data_dir, output_path, classes_to_use, label_map_path, train_indices, current_iteration,
                         train=True):
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
    if train:
        json_path = os.path.join(data_dir, "labels/bdd100k_labels_images_train.json")
        image_dir = os.path.join(data_dir, "images/100k/train/")
        writer = tf.io.TFRecordWriter(output_path + '_train_' + str(current_iteration) + '.tfrecord')

        if os.path.exists(data_dir + "/labels/remaining_images.txt"):
            remaining_images = []
            with open(data_dir + "/labels/remaining_images.txt", 'r') as file:
                for line in file:
                    remaining_images.append(str(line.strip()))
            remaining_images = np.asarray(remaining_images)
        else:
            remaining_images = collect_bdd_images(data_dir)
    else:
        json_path = os.path.join(data_dir, "labels/bdd100k_labels_images_val.json")
        image_dir = os.path.join(data_dir, "images/100k/val/")
        writer = tf.io.TFRecordWriter(output_path + '_val_' + str(current_iteration) + '.tfrecord')
        remaining_images = np.asarray(sorted(os.listdir(image_dir)))

    with open(json_path, 'rb') as json_f:
        items = ijson.items(json_f, 'item')
        img_counter = 0
        skipped_counter = 0
        occurences_percls = {}
        remaining_counter = 0
        unique_range = np.arange(0, 100000)
        for i in range(len(classes_to_use)): occurences_percls[classes_to_use[i]] = 0
        for item in items:  # Item is a dict, which contains name, attributes, timestamp and labels
            img_counter += 1
            if 'labels' not in item:
                skipped_counter += 1
                continue
            unique_name = str(unique_range[remaining_counter])
            img_name = item['name']
            img_path = os.path.join(image_dir, img_name)
            if os.path.isfile(img_path) and img_path.split("/")[-1] in remaining_images[train_indices]:
                encoded_jpg, width, height, extracted_gt = example_extractor(img_path, item, occurences_percls,
                                                                             label_map_dict, classes_to_use)
                if 0 == len(extracted_gt[0]):
                    skipped_counter += 1
                    # print("{0} has no object, skip it and continue.".format(img_name))
                    continue
                assert len(extracted_gt[0]) == len(extracted_gt[4]) == len(extracted_gt[5]) == len(
                    extracted_gt[6]) == len(extracted_gt[7]), 'Problem with list length'
                remaining_counter += 1
                tf_example = example_builder(extracted_gt, encoded_jpg, item, height, width, img_name, unique_name)
                writer.write(tf_example.SerializeToString())
            else:
                skipped_counter += 1
                img_counter += 1
        print('{0} images were processed and {1} were skipped. {2} remaining'.format(img_counter, skipped_counter,
                                                                                     remaining_counter))
        print(occurences_percls)
        writer.close()
        print("Finished creating tfrecord")
        return remaining_counter


def convert_bdd_to_tfrecords(data_dir, output_path, classes_to_use, label_map_path, train=True):
    """Convert the BDD dataset to TFRecords.
    Args:
      data_dir: The full path to the unzipped folder containing the unzipped data.
      output_path: The path to which TFRecord files will be written. The TFRecord
        with the training set will be located at: <output_path>_train100k.tfrecord
        And the TFRecord with the validation set will be located at:
        <output_path>_val100k.tfrecord
      classes_to_use: List of strings naming the classes for which data should be
      converted.
      label_map_path: Path to label map proto.
      train: Sets if its the validation or training TFRecord.
      """
    label_map_dict = label_map_extractor(label_map_path)
    if train:
        json_path = os.path.join(data_dir, "labels/bdd100k_labels_images_train.json")
        image_dir = os.path.join(data_dir, "images/100k/train/")
        writer = tf.io.TFRecordWriter(output_path + '_train100k.tfrecord')
    else:
        json_path = os.path.join(data_dir, "labels/bdd100k_labels_images_val.json")
        image_dir = os.path.join(data_dir, "images/100k/val/")
        writer = tf.io.TFRecordWriter(output_path + '_val100k.tfrecord')

    with open(json_path, 'rb') as json_f:
        items = ijson.items(json_f, 'item')
        img_counter = 0
        skipped_counter = 0
        occurences_percls = {}
        remaining_counter = 0
        unique_range = np.arange(0, 100000)
        for i in range(len(classes_to_use)): occurences_percls[classes_to_use[i]] = 0
        for item in items:  # Item is a dict, which contains name, attributes, timestamp and labels
            img_counter += 1
            if 'labels' not in item:
                skipped_counter += 1
                continue
            unique_name = str(unique_range[remaining_counter])
            img_name = item['name']
            img_path = os.path.join(image_dir, img_name)
            if os.path.isfile(img_path):
                encoded_jpg, width, height, extracted_gt = example_extractor(img_path, item, occurences_percls,
                                                                             label_map_dict, classes_to_use)
                if 0 == len(extracted_gt[0]):
                    skipped_counter += 1
                    # print("{0} has no object, skip it and continue.".format(img_name))
                    continue
                assert len(extracted_gt[0]) == len(extracted_gt[4]) == len(extracted_gt[5]) == len(
                    extracted_gt[6]) == len(extracted_gt[7]), 'Problem with list length'
                remaining_counter += 1
                tf_example = example_builder(extracted_gt, encoded_jpg, item, height, width, img_name, unique_name)
                writer.write(tf_example.SerializeToString())
            else:
                skipped_counter += 1
                img_counter += 1
        print('{0} images were processed and {1} were skipped. {2} remaining'.format(img_counter, skipped_counter,
                                                                                     remaining_counter))
        print(occurences_percls)
        writer.close()


if __name__ == "__main__":
    data_dir = "/app/efficientdet_uncertainty/datasets/BDD100K/bdd100k/"
    # collect_bdd_images(data_dir)
    output_path = "/app/efficientdet_uncertainty/datasets/BDD100K/tf/"
    classes_to_use = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light",
                      "traffic sign"]  # Not capital letters
    label_map_path = "/app/efficientdet_uncertainty/datasets/BDD100K/bdd.pbtxt"
    convert_bdd_to_tfrecords(data_dir, output_path, classes_to_use, label_map_path, train=True)
    convert_bdd_to_tfrecords(data_dir, output_path, classes_to_use, label_map_path, train=False)