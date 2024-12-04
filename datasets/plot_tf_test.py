# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Plot TFRecord example """


import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib.patches import Rectangle

cuda_visible_devices = "0"  # os.getenv("MY_CUDA_VISIBLE_DEVICES")

if cuda_visible_devices is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# TFPATH = '/app/efficientdet_uncertainty/datasets/CODA/tf/_val.tfrecord'
# LABELPATH = '/app/efficientdet_uncertainty/datasets/CODA/coda_data.txt'
# SAVEPATH = '/app/efficientdet_uncertainty/datasets/CODA'
TFPATH = "/app/efficientdet_uncertainty/datasets/BDD100K/collage_crops/collage_crops_gt_randag_051/_train100k.tfrecord"
# TFPATH = '/app/efficientdet_uncertainty/datasets/BDD100K/tf/_train100k.tfrecord'
LABELPATH = "/app/efficientdet_uncertainty/datasets/BDD100K/bdd.pbtxt"
SAVEPATH = "/app/efficientdet_uncertainty/datasets/BDD100K"
TFPATH = "/app/efficientdet_uncertainty/datasets/KITTI/pseudo_labels/num_labeled_10/corrected_gt/_train.tfrecord"
LABELPATH = "/app/efficientdet_uncertainty/datasets/KITTI/kitti.pbtxt"
SAVEPATH = "/app/efficientdet_uncertainty/datasets/KITTI"


def plot_tf_image(image, boxes, labels):
    # Plot the image
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    colors = sns.color_palette("Paired", len(boxes))
    for x in range(len(boxes)):
        y1, x1, y2, x2 = boxes[x]
        label = labels[x]
        plt.gca().add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=colors[x],
                facecolor="none",
            )
        )
        plt.text(
            x1 + 3,
            y1 + 3,
            label,
            bbox=dict(facecolor=colors[x], alpha=0.5),
            fontsize=7,
            color="k",
        )
    plt.axis("off")
    plt.savefig(SAVEPATH + "/tf_test.png")
    plt.close()


def read_tf():
    """Reads tf record and plots one example with GT"""
    dataset = tf.data.TFRecordDataset(TFPATH)

    for record in dataset:
        example = tf.train.Example()
        example.ParseFromString(record.numpy())

        encoded_image = example.features.feature["image/encoded"].bytes_list.value[0]
        image = tf.image.decode_image(encoded_image).numpy()
        width = int(image.shape[1])
        height = int(image.shape[0])

        box_keys = [
            "image/object/bbox/ymin",
            "image/object/bbox/xmin",
            "image/object/bbox/ymax",
            "image/object/bbox/xmax",
        ]
        norm = [float(height), float(width), float(height), float(width)]
        label_key = "image/object/class/text"
        boxes = [
            np.asarray(example.features.feature[bbox_key].float_list.value) * norm[x]
            for x, bbox_key in enumerate(box_keys)
        ]

        boxes_combined = []
        for i in range(0, len(boxes[0])):
            box_comb = [boxes[j][i] for j in range(len(boxes))]
            boxes_combined.append(box_comb)

        label = example.features.feature[label_key].bytes_list.value
        labels = [x.decode("utf-8") for x in label]

        plot_tf_image(image, boxes_combined, labels)
        break


read_tf()
