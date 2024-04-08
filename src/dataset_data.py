# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Define custom information for each dataset """


import os
import json

import label_util

def available_datasets(val=False):
  if val:
    return ["k", "b", "kc", "bc", "ks", "bs", "cbs", "cks"]
  else:
    return ["k", "b", "c",]

def get_ocl_trc(model_dir, img_names):
  """ Retrieves occlusion and truncation information from json label file """
  occlusions = []
  truncations = []
  if "KITTI" in model_dir:     
    for im_name in img_names:  
      f = open(model_dir.split("models")[0]+"/datasets/KITTI/training/label_2/"+im_name[-10:-3]+"txt","r")
      gtdata = f.readlines()
      occl = [gtdata[i].split(" ")[2] for i in range(len(gtdata))]
      trcs = [gtdata[i].split(" ")[1] for i in range(len(gtdata))]
      if occl: occlusions.append(occl)
      else: occlusions.append([-1]*100)
      if trcs: truncations.append(trcs)
      else: truncations.append([-1]*100)

  elif "BDD" in model_dir:   
    label_map = label_util.get_label_map("bdd")
    labels = [label_map[i] for i in range(1, 1 + len(label_map))]
    f = json.load(open(model_dir.split("models")[0]+"/datasets/BDD100K/bdd100k/labels/bdd100k_labels_images_val.json"))
    for j in range(len(f)):
        occl = [f[j]["labels"][i]["attributes"]["occluded"] for i in range(len(f[j]["labels"])) if
                f[j]["labels"][i]["category"] in labels]
        trcs = [f[j]["labels"][i]["attributes"]["truncated"] for i in range(len(f[j]["labels"])) if
                f[j]["labels"][i]["category"] in labels]
        if occl: occlusions.append(occl)
        else: occlusions.append([-1]*100)
        if trcs: truncations.append(trcs)
        else: truncations.append([-1]*100)
  else:
    for _ in img_names:  
      occlusions.append([-1]*100)
      truncations.append([-1]*100)

  return occlusions, truncations
        
def get_dataset_data(path, im_name=None):      
  """ Defines dataset specific information

  Args:
    path (str): Path including model name, which also includes dataset name

  Returns:
    label map, path to validation images, class names, image shape, path to image file
  """
  
  label_map = {}
  img_source_path = None
  img_shape = [0, 0]
  img_file = None
  class_names = []

  if "KITTI" in path:
    label_map = label_util.get_label_map("kitti") 
    img_source_path = '/KITTI/training/image_2/'
    img_shape = [375, 1220]
    for i in range(1, len(label_map)+1):
        class_names.append(label_map[i].capitalize())   
  elif "BDD" in path:
    label_map = label_util.get_label_map("bdd")  
    img_source_path = '/BDD100K/bdd100k/images/100k/val/'
    img_shape = [720, 1280]
    for i in range(1, len(label_map)+1):
        class_names.append(label_map[i].capitalize())    
  elif "CODA" in path:
    label_map = label_util.get_label_map("bdd")
    img_source_path = '/CODA/images/'
    img_shape = [1000, 1500]
  elif "tires" in path:
    label_map = label_util.get_label_map("tires")
    img_source_path = '/Tires/images/'
    img_shape = [1000, 1500]
    for i in range(1, len(label_map)+1):
        class_names.append(label_map[i].capitalize())    
  else:     
    print("No images found")
  if im_name is not None:                
    print(im_name)
    img_file = img_source_path+im_name

  return label_map, img_source_path, class_names, img_shape, img_file