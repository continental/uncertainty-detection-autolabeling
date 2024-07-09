# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Plot ground truth examples of KITTI """


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io
from matplotlib.patches import Rectangle

IMGPATH = '/app/datasets/KITTI/training/image_2/000004.png'
colors = sns.color_palette('Paired', 9 * 2)
categories = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']

img = np.array(io.imread(IMGPATH), dtype=np.int32)
with open(IMGPATH[:-18]+'label_2/'+IMGPATH[-10:-3]+'txt', 'r') as f:
  labels = f.readlines()

fig = plt.figure(figsize=(20,20))
plt.imshow(img)
width = int(img.shape[1])
height = int(img.shape[0])
for line in labels:
  line = line.split()
  label, _, _, _, x1, y1, x2, y2, _, _, _, _, _, _, _ = line
  x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
  if label != 'DontCare':
    plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2,
                                  edgecolor=colors[categories.index(label)],
                                  facecolor='none'))
    plt.text(x1 + 3, y1 + 3, label,
              bbox=dict(facecolor=colors[categories.index(label)], alpha=0.5),
              fontsize=7, color='k')
plt.tight_layout()
plt.savefig("/app/datasets/KITTI/gt_test.png")
plt.close()
