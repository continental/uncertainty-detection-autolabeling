# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Plot TFRecord example """


import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf
import seaborn as sns

# TFPATH = '/app/datasets/CODA/tf/_val.tfrecord'
# LABELPATH = '/app/datasets/CODA/coda_data.txt'
# SAVEPATH = '/app/datasets/CODA'
TFPATH = '/app/datasets/BDD100K/tf/_val100k.tfrecord'
# TFPATH = '/app/datasets/BDD100K/tf/_train100k.tfrecord'
LABELPATH = '/app/datasets/BDD100K/bdd.pbtxt'
SAVEPATH = '/app/datasets/BDD100K'

def plot_tf_image(image, boxes, labels):
    # Plot the image
    plt.figure(figsize=(10, 8))
    plt.imshow(image)    
    colors = sns.color_palette('Paired', len(boxes))
    for x in range(len(boxes)):
        y1, x1, y2, x2 = boxes[x]
        label = labels[x]
        plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                linewidth=2,
                                edgecolor=colors[x],
                                facecolor='none'))
        plt.text(x1 + 3, y1 + 3, label,
                bbox=dict(facecolor=colors[x], alpha=0.5),
                fontsize=7, color='k')
    plt.axis('off')
    plt.savefig(SAVEPATH + "/tf_test.png")
    plt.close()

def read_tf():   
    """ Reads tf record and plots one example with GT """ 
    dataset = tf.data.TFRecordDataset(TFPATH)
    
    for record in dataset:
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        
        encoded_image = example.features.feature['image/encoded'].bytes_list.value[0]
        image = tf.image.decode_image(encoded_image).numpy()
        width = int(image.shape[1])
        height = int(image.shape[0])

        box_keys = ['image/object/bbox/ymin','image/object/bbox/xmin','image/object/bbox/ymax','image/object/bbox/xmax']
        norm = [float(height), float(width), float(height), float(width)]
        label_key = 'image/object/class/text'
        boxes = [np.asarray(example.features.feature[bbox_key].float_list.value)*norm[x] for x, bbox_key in enumerate(box_keys)]

        boxes_combined = []
        for i in range(0, len(boxes[0])):
            box_comb = [boxes[j][i] for j in range(len(boxes))]
            boxes_combined.append(box_comb)

        label = example.features.feature[label_key].bytes_list.value
        labels = [x.decode('utf-8') for x in label]
                        
        plot_tf_image(image, boxes_combined, labels)
        break

read_tf()