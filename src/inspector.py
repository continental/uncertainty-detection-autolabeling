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
Tool to inspect a model."""


import argparse
import os

import dataloader
import hparams_config
import infer_lib
import numpy as np
import tensorflow as tf
import utils
import yaml
from absl import logging
from calibrate_model import Calibrate
from dataset_data import available_datasets
from infer_model import InferImages
from validate_model import Validate

SELECT_GPU = "5"


def main(
    mode,
    dataset_name="k",
    general_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
):
    """Perform calibrate, validate, export, inference for a selected dataset

    Args:
        mode (str): Select application
        dataset_name (str, optional): Dataset; defaults to "k", can be "k", "b", "c"
        general_path (str, optional): Path to workspace

    """
    # Open and load config file
    if len(dataset_name) > 5:
        if os.path.exists(
            general_path + "/configs/inference/AL/inference_" + dataset_name + ".yaml"
        ):
            yaml_path = (
                general_path
                + "/configs/inference/AL/inference_"
                + dataset_name
                + ".yaml"
            )
        else:
            yaml_path = (
                general_path
                + "/configs/inference/SSL/inference_"
                + dataset_name
                + ".yaml"
            )
    else:
        yaml_path = (
            general_path + "/configs/inference/inference_" + dataset_name + ".yaml"
        )

    with open(yaml_path) as f:
        infer_data = yaml.load(f, Loader=yaml.FullLoader)

    model_name = "efficientdet-d0"
    if mode == "export":
        model_dir = infer_data["model_dir"]
    else:
        model_dir = "_"

    saved_model_dir = infer_data["saved_model_dir"]  # Exporting path

    val_file_pattern = infer_data["val_file_pattern"]
    eval_samples = infer_data["eval_samples"]
    hparams = infer_data["hparams"]
    added_name = infer_data["added_name"] if "added_name" in infer_data else ""
    # For video
    input_video = infer_data["video_path"]
    video_save_path = (
        general_path + "/results/inference/videos/" + saved_model_dir.split("/")[-1]
    )

    batch_size = 1
    image_size = -1
    output_video = None
    tflite = ""  # ['', 'FP32', 'FP16', 'INT8']
    file_pattern = None
    num_calibration_steps = 500
    debug = False
    use_xla = False
    only_network = False
    tensorrt = ""  # ['', 'FP32', 'FP16', 'INT8']
    logging.set_verbosity(logging.INFO)

    tf.config.run_functions_eagerly(debug)
    tf.config.optimizer.set_jit(use_xla)
    devices = tf.config.list_physical_devices("GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)

    model_config = hparams_config.get_detection_config(model_name)
    model_config.override(hparams)  # Add custom overrides
    model_config.is_training_bn = False
    if image_size != -1:
        model_config.image_size = image_size
    model_config.image_size = utils.parse_image_size(model_config.image_size)

    model_params = model_config.as_dict()
    if "consist" in dataset_name:
        model_params["consistency_ssl"] = True
    # ---------------------------------------------------------------------
    if mode == "calibrate" or mode == "validate":
        # Get the validation dataset for validation or calibration
        params = dict(batch_size=1, eval_samples=eval_samples)
        model_config.override(params, True)

        val_dataset, img_names = dataloader.InputReader(
            val_file_pattern,
            is_training=False,
            use_fake_data=False,
            max_instances_per_image=model_config.max_instances_per_image,
            debug=False,
        )(model_config.as_dict(), names=True)

        val_dataset = val_dataset.take(eval_samples)
        strategy = tf.distribute.get_strategy()
        val_dataset = strategy.experimental_distribute_dataset(val_dataset)

        gt_classes = np.array([])
        gt_coords = np.array([])
        for _, labels in val_dataset:
            # Groundtruth annotations in a tensor with each row representing [y1, x1, y2, x2, is_crowd, area, class]
            gt_classes = (
                np.vstack((gt_classes, labels["groundtruth_data"][0][:, -1]))
                if gt_classes.size > 0
                else np.array([labels["groundtruth_data"][0][:, -1]])
            )
            gt_coords = (
                np.vstack((gt_coords, [labels["groundtruth_data"][0][:, 0:4]]))
                if gt_coords.size > 0
                else np.array([labels["groundtruth_data"][0][:, 0:4]])
            )
        driver = infer_lib.ServingDriver.create(
            model_dir,
            debug,
            saved_model_dir,
            model_name,
            batch_size or None,
            only_network,
            model_params,
        )

    if mode == "export":
        print("Exporting model")
        driver = infer_lib.KerasDriver(
            model_dir, debug, model_name, batch_size or None, only_network, model_params
        )
        if not saved_model_dir:
            raise ValueError("Please specify --saved_model_dir=")
        if tf.io.gfile.exists(saved_model_dir):
            tf.io.gfile.rmtree(saved_model_dir)
        driver.export(
            saved_model_dir, tensorrt, tflite, file_pattern, num_calibration_steps
        )
        print("Model are exported to %s" % saved_model_dir)

    elif mode == "inference" or mode == "auto-label" or mode == "SSAL":
        print("Predicting with model")
        driver = infer_lib.ServingDriver.create(
            model_dir,
            debug,
            saved_model_dir,
            model_name,
            batch_size or None,
            only_network,
            model_params,
        )
        InferImages(
            infer_data,
            driver,
            model_params,
            saved_model_dir,
            mode,
            added_name,
            general_path,
        ).iterate_infer()

    elif mode == "calibrate":
        print("Calibrating model")
        Calibrate(
            model_params,
            img_names,
            driver,
            gt_classes,
            gt_coords,
            saved_model_dir,
            general_path,
        ).calibrate_regclas()

    elif mode == "validate":
        print("Validating model")
        Validate(
            model_params,
            img_names,
            driver,
            gt_classes,
            gt_coords,
            saved_model_dir,
            general_path,
        ).launch_val()

    elif mode == "video":
        driver = infer_lib.ServingDriver.create(
            model_dir,
            debug,
            saved_model_dir,
            model_name,
            batch_size or None,
            only_network,
            model_params,
        )

        import cv2

        cap = cv2.VideoCapture(input_video)
        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
        if not cap.isOpened():
            print("Error opening input video: {}".format(input_video))

        out_ptr = None
        if output_video:
            frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
            out_ptr = cv2.VideoWriter(
                output_video,
                cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                cap.get(5),
                (frame_width, frame_height),
            )
        i = 0
        imgs_list = []
        min_score = model_config.nms_configs.score_thresh or 0.4
        while cap.isOpened():
            print("Frame number: " + str(i))
            ret, frame = cap.read()
            if not ret:
                break

            raw_frames = np.array([frame])
            detections_bs = driver.serve(raw_frames)
            if model_params["enable_softmax"]:
                boxes, scores, classes, _, logits = tf.nest.map_structure(
                    np.array, detections_bs
                )
            else:
                boxes, scores, classes, _ = tf.nest.map_structure(
                    np.array, detections_bs
                )

            if len(scores[0][scores[0] > min_score]) != 0:
                new_frame = driver.visualize(
                    raw_frames[0],
                    boxes[0],
                    classes[0],
                    scores[0],
                    logits[0],
                    min_score_thresh=min_score,
                    max_boxes_to_draw=model_config.nms_configs.max_output_size,
                )

                if out_ptr:  # Save whole video
                    out_ptr.write(new_frame)
                else:
                    import select
                    import sys

                    print(
                        "Press anything to stop saving frames. You have 2 seconds to answer!"
                    )
                    inpt, _, _ = select.select([sys.stdin], [], [], 2)
                    if inpt:
                        break
                    else:
                        imgs_list.append(video_save_path + "/frame_" + str(i) + ".png")
                        cv2.imwrite(
                            video_save_path + "/frame_" + str(i) + ".png", new_frame
                        )
                        i += 1
                    # TODO: Save uncertainty as well


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)

    datasets = available_datasets()
    modes = [
        "export",
        "inference",
        "calibrate",
        "validate",
        "video",
        "auto-label",
        "SSAL",
    ]
    print("Available datasets:")
    for i, dataset in enumerate(datasets):
        print(f"{i}: {dataset}")

    print("\nAvailable modes:")
    for i, mode in enumerate(modes):
        print(f"{i}: {mode}")

    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--dataset", help="Select a dataset")
    parser.add_argument(
        "--mode", type=int, choices=range(len(modes)), help="Select a mode"
    )
    parser.add_argument(
        "--general_path",
        type=str,
        help="Path to workspace",
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )

    args = parser.parse_args()
    general_path = None
    if args.dataset is not None and args.mode is not None:
        dataset_choice = args.dataset
        mode_choice = args.mode
        general_path = args.general_path
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = SELECT_GPU
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Command-line arguments not provided. Asking for user input.")
        try:
            dataset_choice = str(
                input("Enter the dataset letter or what is after inference_***.yaml: ")
            )
            mode_choice = int(input("Enter the number for mode: "))
        except ValueError:
            print("Invalid input. Please enter valid numbers.")
            dataset_choice = None
            mode_choice = None

    if dataset_choice is None or mode_choice is None:
        print("Please provide valid dataset and mode selections.")
    else:
        # Check if the entered numbers are within valid range
        if 0 <= mode_choice < len(modes):
            selected_dataset = dataset_choice
            selected_mode = modes[mode_choice]
            if general_path is None:
                main(selected_mode, selected_dataset)
            else:
                main(selected_mode, selected_dataset, general_path)
        else:
            print("Invalid number selection. Please enter numbers within valid range.")
