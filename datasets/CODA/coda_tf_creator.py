# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

""" Create TFRecord with CODA dataset """


import hashlib
import io
import json
import os

import numpy as np
import PIL.Image as pil
import tensorflow as tf
from object_detection.utils import dataset_util

GPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def label_map_extractor(label_map_path):
    """Extract a dictionary with class labels and IDs from txt file"""
    ids = []
    names = []
    label_map = {}
    with open(label_map_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if "name" in line:
                names.append(line.split(":")[1].strip().strip("'"))
            elif "id" in line:
                ids.append(int(line.split(":")[1].strip()))
    for i in range(len(ids)):
        label_map[names[i]] = ids[i]
    return label_map


def coda_active_tfrecords(
    data_dir,
    output_path,
    classes_to_use,
    label_map_path,
    train_indices,
    current_iteration,
    train=False,
):
    """Active learning TF creator used in src/active_learning_loop.py.
    Args:
      data_dir: The full path to the unzipped folder containing the unzipped data.
      output_path: The path to which TFRecord files will be written.
      classes_to_use: List of strings naming the classes for which data should be
      converted.
      label_map_path: Path to label map proto.
      train_indices: Image indices to select for TFRecord.
      current_iteration: Save name with current active learning iteration number.
    Note: train (bool) is keyword for dataset KITTI, called train to maintain same
    usage for other datasets.
    """

    if train:
        classes_to_use, mod_ids, _, _ = extract_class_info("KITTI")
    else:
        classes_to_use, mod_ids, _, _ = extract_class_info("BDD")
    label_map_dict = label_map_extractor(label_map_path)

    with open(os.path.join(data_dir, "annotations.json")) as file:
        img_anno = json.load(file)["annotations"]

    image_dir = os.path.join(data_dir, "images")
    val_writer = tf.io.TFRecordWriter(
        output_path + "_val_" + str(current_iteration) + ".tfrecord"
    )

    images = sorted(tf.io.gfile.listdir(image_dir))

    result = []
    current_image_id = None
    current_annotations = []

    for annotation in img_anno:
        image_id = annotation["image_id"]

        if image_id != current_image_id:
            if current_annotations:
                result.append(current_annotations)
            current_annotations = []
            current_image_id = image_id

        current_annotations.append(annotation)

    if current_annotations:
        result.append(current_annotations)
    val_count = 0
    for img_num, img_name in enumerate(images):
        annotation_for_image = discard_classes(result[img_num], classes_to_use)
        if img_num in train_indices and len(annotation_for_image) != 0:
            image_path = os.path.join(image_dir, img_name)
            example = tf_exp(image_path, annotation_for_image, label_map_dict, mod_ids)
            val_writer.write(example.SerializeToString())
            val_count += 1

    val_writer.close()
    print("Finished creating tfrecords")
    print("Val images:", val_count)
    print("Total including skipped images:", img_num)


def convert_coda_to_tfrecords(
    data_dir, output_path, classes_to_use, label_map_path, mod_ids
):
    """Convert the CODA dataset to TFRecords.
    Args:
      data_dir: The full path to the unzipped folder containing the unzipped data.
      output_path: The path to which TFRecord with the validation. It will be located at:
        <output_path>_val.tfrecord
      classes_to_use: List of strings naming the classes for which data should be converted.
      label_map_path: Path to label map proto.
      mod_ids: Modified class IDs to map CODA to target dataset.
    """
    label_map_dict = label_map_extractor(label_map_path)

    with open(os.path.join(data_dir, "annotations.json")) as file:
        img_anno = json.load(file)["annotations"]

    image_dir = os.path.join(data_dir, "images")
    val_writer = tf.io.TFRecordWriter("%s_val.tfrecord" % output_path)

    images = sorted(tf.io.gfile.listdir(image_dir))

    result = []
    current_image_id = None
    current_annotations = []

    for annotation in img_anno:
        image_id = annotation["image_id"]

        if image_id != current_image_id:
            if current_annotations:
                result.append(current_annotations)
            current_annotations = []
            current_image_id = image_id

        current_annotations.append(annotation)

    if current_annotations:
        result.append(current_annotations)
    val_count = 0
    img_num = 0
    for img_name in images:
        image_path = os.path.join(image_dir, img_name)
        annotation_for_image = discard_classes(result[img_num], classes_to_use)
        if len(annotation_for_image) != 0:
            example = tf_exp(image_path, annotation_for_image, label_map_dict, mod_ids)
            val_writer.write(example.SerializeToString())
            val_count += 1
        img_num += 1

    val_writer.close()
    print("Finished creating tfrecords")
    print("Val images:", val_count)
    print("Total including skipped images:", img_num)


def tf_exp(image_path, annotations, label_map_dict, mod_ids):
    """Convert dict to TF example"""
    with tf.io.gfile.GFile(image_path, "rb") as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = np.asarray(pil.open(encoded_jpg_io))
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(image.shape[1])
    height = int(image.shape[0])

    left_values = []
    right_values = []
    top_values = []
    bottom_values = []

    cat_ids = []
    for annotation in annotations:  # [x, y, w, h].
        cat_ids.append(annotation["category_id"])
        bbox = annotation["bbox"]
        left_values.append(bbox[0])
        right_values.append(bbox[0] + bbox[2])
        top_values.append(bbox[1])
        bottom_values.append(bbox[1] + bbox[3])

    try:
        xmin_norm = np.asarray(left_values) / float(width)
        ymin_norm = np.asarray(top_values) / float(height)
        xmax_norm = np.asarray(right_values) / float(width)
        ymax_norm = np.asarray(bottom_values) / float(height)
    except KeyError:
        print("Without objects")

    difficult_obj = [0] * len(xmin_norm)
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/filename": dataset_util.bytes_feature(
                    image_path.split("/")[-1].encode("utf8")
                ),
                "image/source_id": dataset_util.bytes_feature(
                    str(annotations[0]["image_id"]).encode("utf8") or "0".encode("utf8")
                ),
                "image/key/sha256": dataset_util.bytes_feature(key.encode("utf8")),
                "image/encoded": dataset_util.bytes_feature(encoded_jpg),
                "image/format": dataset_util.bytes_feature("jpg".encode("utf8")),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmin_norm),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmax_norm),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymin_norm),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymax_norm),
                "image/object/class/text": dataset_util.bytes_list_feature(
                    [
                        next(
                            (
                                key
                                for key, value in label_map_dict.items()
                                if value == mod_ids[x]
                            ),
                            None,
                        ).encode("utf8")
                        for x in cat_ids
                    ]
                ),
                "image/object/class/label": dataset_util.int64_list_feature(
                    [mod_ids[x] for x in cat_ids]
                ),
                "image/object/difficult": dataset_util.int64_list_feature(
                    difficult_obj
                ),
            }
        )
    )

    return example


def discard_classes(annotations, used_classes):
    """Keeps detections with classes out of a selected list"""
    dont_care_indices = [
        i
        for i in range(len(annotations))
        if annotations[i]["category_id"] not in used_classes
    ]
    if len(dont_care_indices) != 0:
        boxes_to_remove = []
        for i in range(len(annotations)):
            if i in dont_care_indices:
                boxes_to_remove.append(True)
            else:
                boxes_to_remove.append(False)

        annotations = np.asarray(annotations)[np.logical_not(boxes_to_remove)]
    return annotations


def extract_class_info(slct_dataset):
    """Pre-defined information for both datasets KITTI and BDD and their connection to CODA."""
    with open(GPATH + "/CODA/annotations.json") as file:
        data = json.load(file)
    data_categories = data["categories"]
    if slct_dataset == "KITTI":
        search_names = [
            "car",
            "van",
            "truck",
            "pedestrian",
            "person_sitting",
            "cyclist",
            "tram",
        ]
        mod_ids = [4, 6, 1, 3]
        label_map_path = GPATH + "/CODA/coda_data.txt"
        output_path = GPATH + "/CODA/tf/"
    else:
        search_names = [
            "pedestrian",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            "traffic_light",
            "traffic_sign",
        ]
        mod_ids = [1, 3, 4, 5, 8, 7, 9, 10]
        label_map_path = GPATH + "/CODA/coda_data_BDD.txt"
        output_path = GPATH + "/CODA/tf_BDD/"

    # Iterate over the data to find the IDs for each name
    results = {}
    for item in data_categories:
        if item["name"] in search_names:
            results[item["name"]] = item["id"]

    # Print the results and get classes to use
    ids = []
    classes_to_use = []
    for name, id_ in results.items():
        print(f"The ID of {name} is {id_}.")
        ids.append(id_)
        classes_to_use.append(name)
    classes_to_use = np.asarray(classes_to_use)
    if "traffic_light" in classes_to_use:
        classes_to_use[np.where(classes_to_use == "traffic_light")[0]] = "traffic light"
    if "traffic_sign" in classes_to_use:
        classes_to_use[np.where(classes_to_use == "traffic_sign")[0]] = "traffic sign"

    with open(label_map_path, "w") as file:
        for i in range(len(ids)):
            file.write(f"item {{\n")
            file.write(f"  id: {mod_ids[i]}\n")
            file.write(f"  name: '{classes_to_use[i]}'\n")
            file.write(f"}}\n")
    mod_ids = {ids[x]: mod_ids[x] for x in range(len(mod_ids))}
    return ids, mod_ids, label_map_path, output_path


if __name__ == "__main__":
    ids, mod_ids, label_map_path, output_path = extract_class_info("KITTI")
    data_dir = GPATH + "/CODA/"
    convert_coda_to_tfrecords(
        data_dir=data_dir,
        output_path=output_path,
        classes_to_use=ids,
        label_map_path=label_map_path,
        mod_ids=mod_ids,
    )
