# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Plot ground truth examples of BDD """


import io
import os

import ijson
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from object_detection.utils import label_map_util
from skimage import io

data_dir = "/app/efficientdet_uncertainty/datasets/BDD100K/bdd100k"
json_path = os.path.join(data_dir, "labels/bdd100k_labels_images_train.json")
image_dir = os.path.join(data_dir, "images/100k/train/")

classes_to_use = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]
label_map_path = "/app/efficientdet_uncertainty/datasets/BDD100K/bdd.pbtxt"
img_indx = 1

label_map_dict = label_map_util.get_label_map_dict(label_map_path)
colors = sns.color_palette("Paired", 9 * 2)
with open(json_path, "rb") as json_f:
    item = list(ijson.items(json_f, "item"))[img_indx]
    unique_range = np.arange(0, 100000)
    img_name = item["name"]
    img_path = os.path.join(image_dir, img_name)
    if os.path.isfile(img_path):
        image = np.array(io.imread(img_path), dtype=np.int32)
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

        current_labels = item["labels"]
        for label in current_labels:
            category = label["category"]
            if category in classes_to_use:
                labels.append(label_map_dict[category])
                classes.append(category.encode("utf8"))
                attributes = label["attributes"]
                occluded.append(int(attributes["occluded"] == "true"))
                truncated.append(int(attributes["truncated"] == "true"))
                box2d = label["box2d"]
                xmins.append(float(box2d["x1"]) / width)
                ymins.append(float(box2d["y1"]) / height)
                xmaxs.append(float(box2d["x2"]) / width)
                ymaxs.append(float(box2d["y2"]) / height)
        fig = plt.figure(figsize=(20, 20))
        plt.imshow(image)
        for i in range(len(xmins)):
            x1, y1, x2, y2 = map(
                float,
                [
                    xmins[i] * float(width),
                    ymins[i] * float(height),
                    xmaxs[i] * float(width),
                    ymaxs[i] * float(height),
                ],
            )
            plt.gca().add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor=colors[labels[i]],
                    facecolor="none",
                )
            )
            plt.text(
                x1 + 3,
                y1 + 3,
                str(classes[i])[2:-1],
                bbox=dict(facecolor=colors[labels[i]], alpha=0.5),
                fontsize=20,
                color="k",
            )

        plt.tight_layout()
        plt.savefig("/app/efficientdet_uncertainty/datasets/BDD100K/gt_test.png")
        plt.close()
