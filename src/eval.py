# Original Copyright 2020 Google Research. All Rights Reserved.
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
""" Modified.
Eval libraries."""


from absl import logging
import argparse
import os

import yaml
import tensorflow as tf

import coco_metric
import dataloader
import hparams_config
import utils
import anchors
import efficientdet_keras
import label_util
import postprocess
import utils_keras
from utils_extra import mc_eval
from dataset_data import available_datasets

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU') 
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)


def main(dataset_name):
  """ Evaluate trained model on selected dataset """
  with open(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/configs/eval/eval_'+dataset_name+'.yaml') as f:
      eval_data = yaml.load(f, Loader=yaml.FullLoader)
  val_file_pattern = eval_data["val_file_pattern"]
  model_dir = eval_data["model_dir"]
  eval_samples = eval_data["eval_samples"]
  hparams = eval_data["hparams"]
  print("Eval samples: ", eval_samples)
  # Cloud TPU Cluster Resolvers
  tpu=None
  gcp_project=None
  tpu_zone=None
  val_json_file= None
  batch_size=8
  model_name = 'efficientdet-d0'
  logging.set_verbosity(logging.ERROR)
  config = hparams_config.get_efficientdet_config(model_name)
  config.override(hparams)
  config.val_json_file = val_json_file
  config.nms_configs.max_nms_inputs = anchors.MAX_DETECTION_POINTS
  config.drop_remainder = False  # eval all examples w/o drop.
  config.image_size = utils.parse_image_size(config['image_size'])

  if config.strategy == 'tpu':
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu, zone=tpu_zone, project=gcp_project)
    tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
    ds_strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
    logging.info('All devices: %s', tf.config.list_logical_devices('TPU'))
  elif config.strategy == 'gpus':
    ds_strategy = tf.distribute.MirroredStrategy()
    logging.info('All devices: %s', tf.config.list_physical_devices('GPU'))
  else:
    if tf.config.list_physical_devices('GPU'):
      ds_strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
    else:
      ds_strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')

  with ds_strategy.scope():
    # Network
    model = efficientdet_keras.EfficientDetNet(config=config)
    model.build((None, *config.image_size, 3))
    print(model.config)
    utils_keras.restore_ckpt(model,
                            tf.train.latest_checkpoint(model_dir),
                            config.moving_average_decay,
                            skip_mismatch=False)
    @tf.function
    def model_fn(images, labels):
      if config.mc_dropout: cls_outputs, box_outputs = mc_eval(model, images, config) # If mc dropout is activated
      else: cls_outputs, box_outputs = model(images, training=False)
      
      detections = postprocess.generate_detections(config,
                                                   cls_outputs,
                                                   box_outputs,
                                                   labels['image_scales'],
                                                   labels['source_ids'])
      detections = postprocess.transform_detections(detections)       
      tf.numpy_function(evaluator.update_state,
                        [labels['groundtruth_data'],
                         detections], [])
    # Evaluator for AP calculation.
    label_map = label_util.get_label_map(config.label_map)
    evaluator = coco_metric.EvaluationMetric(
        filename=config.val_json_file, label_map=label_map, apiou_curve=False)

    # dataset
    ds = dataloader.InputReader(
        val_file_pattern,
        is_training=False,
        max_instances_per_image=config.max_instances_per_image)(
            config, batch_size=batch_size)
    if eval_samples:
      ds = ds.take((eval_samples + batch_size - 1) // batch_size)

    ds = ds_strategy.experimental_distribute_dataset(ds)

    # evaluate all images.
    eval_sample = eval_samples or 5000
    pbar = tf.keras.utils.Progbar((eval_sample + batch_size - 1) // batch_size)
    for i, (images, labels) in enumerate(ds):
      ds_strategy.run(model_fn, (images, labels))
      pbar.update(i)

  # compute the final eval results.
  metrics = evaluator.result()
  metric_dict = {}
  for i, name in enumerate(evaluator.metric_names):
    metric_dict[name] = metrics[i]
  if label_map:
      for i, cid in enumerate(sorted(label_map.keys())):
          name = 'AP_/%s' % label_map[cid]
          metric_dict[name] = metrics[i + len(evaluator.metric_names)]

  print("\n")
  print(model_name + "\n")
  for met in str(metric_dict).split(","):
      print(str(met) + "\n", end="")


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    datasets = available_datasets(val=True)
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--dataset", type=int, choices=range(len(datasets)), help="Select a dataset")
    args = parser.parse_args()
    if args.dataset is not None:
        dataset_choice = args.dataset
    else:
        print("Command-line arguments not provided. Asking for user input.") 
        print("Available datasets:")
        for i, dataset in enumerate(datasets):
            print(f"{i}: {dataset}")
        try:  
          dataset_choice = int(input("Enter the number for dataset: "))
        except ValueError:
          print("Invalid input. Please enter valid number.")
          dataset_choice = None


    if dataset_choice is None:
        print("Please provide valid dataset.")
    else:
      # Check if the entered numbers are within valid range
      if 0 <= dataset_choice < len(datasets):
          selected_dataset = datasets[dataset_choice]
          main(selected_dataset)
      else:
          print("Invalid number selection. Please enter numbers within valid range.")

