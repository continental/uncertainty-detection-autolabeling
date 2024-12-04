# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================

import argparse
import ast
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import uncertainty_analysis
import yaml
from utils_box import relativize_uncert

sys.path.insert(5, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.BDD100K.bdd_tf_creator import bdd_active_tfrecords
from datasets.KITTI.kitti_tf_creator import kitti_active_tfrecords


class STAC:
    """Custom implementation of student-teacher SSL approach based on STAC:
    Sohn, Kihyuk, et al. "A simple semi-supervised learning framework for object detection."
    arXiv preprint arXiv:2005.04757 (2020).
    """

    def __init__(
        self,
        dataset="KITTI",
        portion_labeled=10,
        tau=0.4,
        selection_strategy="score",
        teacher_strategy="",
        clw=20,
        added_name="_V",
        version_num=0,
        num_epochs=200,
        training_method="baseline",
        early_stopping=False,
        general_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        rand_seed=None,
    ):
        """Initializes STAC

        Args:
            dataset (str, optional): Dataset name. Defaults to "KITTI".
            portion_labeled (int, optional): Portion of labeled data. Defaults to 25%.
            tau (float, optional): Threshold to filter predictions based on. Default to 0.9.
            selection_strategy (str, optional): Data selection strategy. Defaults to "score" for classification score.
            teacher_strategy (str, optional): Teacher model strategy, e.g., rcf_cblog20_imscore. Defaults to "".
            clw (int, optional): Class weight for loss function. Defaults to 20.
            added_name (str, optional): Add name for saving. Defaults to "_V".
            version_num (int, optional): Training version. Defaults to 0.
            num_epochs (int, optional): Number of epochs per AL iteration. Defaults to 200.
            training_method (str, optional): Training method, lossatt, baseline or else MCdropout+LA. Defaults to "baseline".
            early_stopping (bool, optional): Activates early stopping instead of full training. Defaults to False.
            general_path (str, optional): Path to wdir. Defaults to parent folder of src.
            rand_seed (_type_, optional): Random seed for all random operations. Defaults to None.
        """
        self.config = (
            f"dataset: {dataset}, portion_labeled: {portion_labeled}, added_name: {added_name}, selection_strategy: {selection_strategy}, teacher_strategy: {teacher_strategy}, tau: {tau}, "
            f"version_num: {version_num}, num_epochs: {num_epochs}, training_method: {training_method}, rand_seed: {rand_seed}"
        )
        print(self.config)
        if rand_seed is not None:
            np.random.seed(rand_seed)
        self.clw = clw
        self.dataset = dataset
        self.teacher_strategy = teacher_strategy
        self.selection_strategy = selection_strategy
        self.self_train = True if "selftrain" in selection_strategy else False
        self.activate_pseudoscore = (
            True if "pseudoscore" in selection_strategy else False
        )
        self.tau = tau
        self.training_method = training_method
        self.general_path = general_path + "/"
        if self.dataset == "KITTI":
            # TFRecord config
            self.data_dir = self.general_path + "datasets/KITTI/training"
            self.data_dir_im = self.data_dir + "/image_2/"
            self.select_classes = [
                "car",
                "van",
                "truck",
                "pedestrian",
                "person_sitting",
                "cyclist",
                "tram",
            ]  # Not capital
            self.label_path = (
                self.general_path + "datasets/KITTI/kitti.pbtxt"
            )  # Path to label map
            # Path to dataset images
            self.im_names = [
                os.path.basename(path)
                for path in sorted(
                    glob.glob(os.path.join(self.data_dir, "image_2", "*"))
                )
            ]
            # Validation set indices
            self.val_indices = []
            with open(
                self.general_path + "datasets/KITTI/vaL_index_list.txt", "r"
            ) as file:
                for line in file:
                    self.val_indices.append(int(line.strip()))
            # Training parameters
            self.eval_samples = 1496
            self.val_tf = self.general_path + "datasets/KITTI/tf/_val.tfrecord"
            if "lossatt" in training_method:
                self.hparams = (
                    self.general_path + "configs/train/allclasses_lossatt.yaml"
                )
            elif "baseline" in training_method:
                self.hparams = self.general_path + "configs/train/allclasses_orig.yaml"
            else:
                self.hparams = (
                    self.general_path
                    + "configs/train/allclasses_mcdropout_lossatt_head.yaml"
                )

        elif self.dataset == "BDD100K":
            self.data_dir = self.general_path + "datasets/BDD100K/bdd100k/"
            self.data_dir_im = self.data_dir + "/images/100k/train/"
            self.select_classes = [
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
            self.label_path = self.general_path + "datasets/BDD100K/bdd.pbtxt"
            self.im_names = []
            with open(
                self.general_path
                + "datasets/BDD100K/bdd100k/labels/remaining_images.txt",
                "r",
            ) as file:
                for line in file:
                    self.im_names.append(str(line.strip()))
            self.val_indices = []  # Separate folder
            self.eval_samples = 10000
            self.val_tf = self.general_path + "datasets/BDD100K/tf/_val100k.tfrecord"
            if "lossatt" in training_method:
                self.hparams = (
                    self.general_path + "configs/train/allclasses_lossatt_BDD.yaml"
                )
            elif "baseline" in training_method:
                self.hparams = (
                    self.general_path + "configs/train/allclasses_orig_BDD.yaml"
                )
            else:
                self.hparams = (
                    self.general_path
                    + "configs/train/allclasses_mcdropout_lossatt_BDD_head.yaml"
                )
        len_train_ims = len(self.im_names) - len(self.val_indices)
        self.available_indices = [
            i for i in range(len(self.im_names)) if i not in self.val_indices
        ]
        # Universal training parameters
        self.batch_size = 8
        self.early_stopping = early_stopping
        self.num_epochs = num_epochs
        self.model_dir = self.general_path + "models/trained_models/STAC/"
        self.model_name = "efficientdet-d0"
        self.pretrained_ckpt = "efficientdet-d0"
        # Universal STAC config
        self.infer_results = (
            self.general_path + "/results/inference/"
        )  # Path to model inference results
        self.portion_labeled = portion_labeled
        self.num_labeled = int(portion_labeled / 100 * len_train_ims)
        print(
            f"# total images: {len_train_ims}, # labeled images: {self.num_labeled}, # unlabeled images {len_train_ims-self.num_labeled}"
        )
        self.version_num = version_num
        self.added_name = added_name + str(self.version_num)
        self.tf_path = (
            self.general_path
            + "datasets/"
            + self.dataset
            + "/tf_stac/num_labeled_"
            + str(portion_labeled)
            + "/V"
            + str(self.version_num)
            + "/"
        )
        if not os.path.exists(self.tf_path):
            os.makedirs(self.tf_path)

    def write_KITTI_pseudo_gt_txt(self):
        """
        Writes pseudo ground truth annotations in KITTI format to text files.

        Returns:
            int: The total number of detections written to the text files.
        """

        n_dets = 0
        os.makedirs(self.output_dir, exist_ok=True)
        if self.activate_pseudoscore:
            zipped_data = zip(
                self.pred_imgs_names,
                self.pred_classes,
                self.pred_boxes,
                self.pseudo_score,
            )
        else:
            zipped_data = zip(self.pred_imgs_names, self.pred_classes, self.pred_boxes)
        for zdata in zipped_data:
            output_path = os.path.join(self.output_dir, f"{zdata[0].split('.')[0]}.txt")
            with open(output_path, "w") as f:
                if self.activate_pseudoscore:
                    zipped_zdata = zip(zdata[1], zdata[2], zdata[3])
                else:
                    zipped_zdata = zip(zdata[1], zdata[2])
                for ddata in zipped_zdata:
                    # Assuming `box` is in the format [y_min, x_min, y_max, x_max]
                    # Dummy values are used for the other KITTI format fields
                    if self.activate_pseudoscore:
                        line = f"{ddata[0].capitalize()} 0.0 0 -10 {ddata[1][1]} {ddata[1][0]} {ddata[1][3]} {ddata[1][2]} 0.0 0.0 0.0 0.0 0.0 0.0 {np.round(ddata[2],2)}\n"
                    else:
                        line = f"{ddata[0].capitalize()} 0.0 0 -10 {ddata[1][1]} {ddata[1][0]} {ddata[1][3]} {ddata[1][2]} 0.0 0.0 0.0 0.0 0.0 0.0 -10\n"
                    f.write(line)
                    n_dets += 1
        return n_dets

    def write_BDD_pseudo_gt_json(self):
        """
        Writes pseudo ground truth labels in BDD format to a JSON file.

        Returns:
            int: The total number of detections.
        """
        n_dets = 0
        os.makedirs(self.output_dir, exist_ok=True)
        if self.activate_pseudoscore:
            zipped_data = zip(
                self.pred_imgs_names,
                self.pred_classes,
                self.pred_boxes,
                self.pseudo_score,
            )
        else:
            zipped_data = zip(self.pred_imgs_names, self.pred_classes, self.pred_boxes)
        pseudo_gt = []
        for zdata in zipped_data:
            image_data = {
                "name": zdata[0],
                "attributes": {
                    "weather": "overcast",
                    "timeofday": "daytime",
                    "scene": "city street",
                },
                "timestamp": 10000,  # Assuming a fixed timestamp, adjust as needed
                "labels": [],
            }

            if self.activate_pseudoscore:
                zipped_zdata = zip(zdata[1], zdata[2], zdata[3])
            else:
                zipped_zdata = zip(zdata[1], zdata[2])
            for label_id, ddata in enumerate(zipped_zdata):
                label_data = {
                    "id": str(label_id),
                    "attributes": {
                        "occluded": False,
                        "truncated": False,
                        "trafficLightColor": "NA",
                    },
                    "category": ddata[0],
                    "box2d": {
                        "x1": ddata[1][1],
                        "y1": ddata[1][0],
                        "x2": ddata[1][3],
                        "y2": ddata[1][2],
                    },
                }
                if self.activate_pseudoscore:
                    label_data["pseudo_score"] = np.round(ddata[2], 2)
                image_data["labels"].append(label_data)
                n_dets += 1

            pseudo_gt.append(image_data)

        output_path = os.path.join(self.output_dir, "pseudo_labels.json")
        with open(output_path, "w") as f:
            json.dump(pseudo_gt, f, indent=4)
        return n_dets

    def score_image(self, inference_path, student_model_directory):
        """Selects samples based on inference results and a scoring strategy"""

        def find_global_min_max(nested_list):
            """Finds the global minimum and maximum values in a nested list"""
            global_min = float("inf")
            global_max = float("-inf")

            def recurse(items):
                nonlocal global_min, global_max
                for item in items:
                    if isinstance(item, list):
                        recurse(item)
                    else:
                        global_min = min(global_min, item)
                        global_max = max(global_max, item)

            recurse(nested_list)
            return global_min, global_max

        def minmax_normalize(nested_list):
            """Normalizes a nested list using min-max normalization"""
            global_min, global_max = find_global_min_max(nested_list)

            def normalize(value):
                return (
                    (value - global_min) / (global_max - global_min)
                    if global_max - global_min > 0
                    else 0
                )

            def recurse_normalize(items):
                for i in range(len(items)):
                    if isinstance(items[i], list):
                        recurse_normalize(items[i])
                    else:
                        items[i] = normalize(items[i])

            recurse_normalize(nested_list)
            return nested_list

        f = open(inference_path, "r")
        dets = f.readlines()
        detections = [ast.literal_eval(d.replace("inf", "2e308")) for d in dets]
        per_image_score = []
        pred_classes = []
        pred_boxes = []
        pred_imgs_names = []
        pred_imgs_names.append(detections[0]["image_name"])
        img_name = detections[0]["image_name"]
        i = 0
        max_detections_per_image = 99
        # Get per image score
        while i < len(detections):
            curr_img_name = detections[i]["image_name"]
            if "calib" in self.selection_strategy:
                box_calib_mode = "iso_perclscoo_"
                class_calib_mode = "iso_percls_"
                if "box" in self.selection_strategy:
                    add_mode = box_calib_mode
                else:
                    add_mode = class_calib_mode
            else:
                if (
                    "box" in self.selection_strategy
                    or "class" in self.selection_strategy
                ):
                    add_mode = "uncalib_"
                else:
                    add_mode = ""
            if "alluncert" in self.selection_strategy:
                temp_ssl_score = [[], [], [], []]
            elif (
                "epuncert" in self.selection_strategy
                or "ental" in self.selection_strategy
            ):
                temp_ssl_score = [[], [], []]
            elif "combo" in self.selection_strategy:
                temp_ssl_score = [[], []]
            else:
                temp_ssl_score = []
            # Iterate over detections for current image
            temp_classes = []
            temp_boxes = []
            while curr_img_name == img_name:
                if "combo" in self.selection_strategy:
                    if "calib" in self.selection_strategy:
                        combo_uncert = self.opt_params[0] * detections[i][
                            "iso_percls_entropy"
                        ] + self.opt_params[1] * np.mean(
                            relativize_uncert(
                                [detections[i]["bbox"]],
                                [detections[i]["iso_perclscoo_albox"]],
                            )
                        )
                    else:
                        combo_uncert = self.opt_params[0] * detections[i][
                            "entropy"
                        ] + self.opt_params[1] * np.mean(
                            relativize_uncert(
                                [detections[i]["bbox"]],
                                [detections[i]["uncalib_albox"]],
                            )
                        )
                    temp_ssl_score[0].append(combo_uncert)
                    temp_ssl_score[-1].append(detections[i]["det_score"])

                elif "alluncert" in self.selection_strategy:
                    if "calib" in self.selection_strategy:
                        temp_ssl_score[0].append(
                            np.mean(
                                relativize_uncert(
                                    [detections[i]["bbox"]],
                                    [detections[i][box_calib_mode + "mcbox"]],
                                )
                            )
                        )
                        temp_ssl_score[1].append(
                            np.mean(
                                relativize_uncert(
                                    [detections[i]["bbox"]],
                                    [detections[i][box_calib_mode + "albox"]],
                                )
                            )
                        )
                        temp_ssl_score[2].append(
                            np.mean(detections[i][class_calib_mode + "mcclass"])
                        )
                    else:
                        temp_ssl_score[0].append(
                            np.mean(
                                relativize_uncert(
                                    [detections[i]["bbox"]],
                                    [detections[i]["uncalib_mcbox"]],
                                )
                            )
                        )
                        temp_ssl_score[1].append(
                            np.mean(
                                relativize_uncert(
                                    [detections[i]["bbox"]],
                                    [detections[i]["uncalib_albox"]],
                                )
                            )
                        )
                        temp_ssl_score[2].append(
                            np.mean(detections[i]["uncalib_mcclass"])
                        )
                    temp_ssl_score[-1].append(detections[i]["det_score"])

                elif "epuncert" in self.selection_strategy:
                    if "calib" in self.selection_strategy:
                        temp_ssl_score[0].append(
                            np.mean(
                                relativize_uncert(
                                    [detections[i]["bbox"]],
                                    [detections[i][box_calib_mode + "mcbox"]],
                                )
                            )
                        )
                        temp_ssl_score[1].append(
                            np.mean(detections[i][class_calib_mode + "mcclass"])
                        )
                    else:
                        temp_ssl_score[0].append(
                            np.mean(
                                relativize_uncert(
                                    [detections[i]["bbox"]],
                                    [detections[i]["uncalib_mcbox"]],
                                )
                            )
                        )
                        temp_ssl_score[1].append(
                            np.mean(detections[i]["uncalib_mcclass"])
                        )
                    temp_ssl_score[-1].append(detections[i]["det_score"])

                elif "ental" in self.selection_strategy:
                    if "calib" in self.selection_strategy:
                        temp_ssl_score[0].append(
                            np.mean(
                                relativize_uncert(
                                    [detections[i]["bbox"]],
                                    [detections[i][box_calib_mode + "albox"]],
                                )
                            )
                        )
                        temp_ssl_score[1].append(
                            detections[i][class_calib_mode + "entropy"]
                        )
                    else:
                        temp_ssl_score[0].append(
                            np.mean(
                                relativize_uncert(
                                    [detections[i]["bbox"]],
                                    [detections[i]["uncalib_albox"]],
                                )
                            )
                        )
                        temp_ssl_score[1].append(detections[i]["entropy"])
                    temp_ssl_score[-1].append(detections[i]["det_score"])

                else:
                    if (
                        add_mode + self.selection_strategy.split("_")[-1]
                        in detections[i]
                    ):
                        sel_metric = detections[i][
                            add_mode + self.selection_strategy.split("_")[-1]
                        ]
                        if (
                            "box" in self.selection_strategy
                            and "norm" in self.selection_strategy
                        ):
                            temp_ssl_score.append(
                                np.mean(
                                    relativize_uncert(
                                        [detections[i]["bbox"]], [sel_metric]
                                    )
                                )
                            )
                        elif isinstance(sel_metric, float):
                            temp_ssl_score.append(sel_metric)
                        else:
                            temp_ssl_score.append(np.mean(sel_metric))
                    else:
                        temp_ssl_score.append(detections[i]["det_score"])
                temp_classes.append(detections[i]["class"])
                temp_boxes.append(detections[i]["bbox"])
                i += 1  # Check if last detection
                if i == len(detections):
                    break
                curr_img_name = detections[i]["image_name"]

            if i < len(detections):
                pred_imgs_names.append(curr_img_name)
            per_image_score.append(temp_ssl_score[:max_detections_per_image])
            pred_classes.append(temp_classes[:max_detections_per_image])
            pred_boxes.append(temp_boxes[:max_detections_per_image])
            img_name = curr_img_name

        if "combo" in self.selection_strategy:
            uncerts = [
                [pt[i] for pt in per_image_score]
                for i in range(len(per_image_score[0]))
            ]

            filter_based_on_sigmoid = [
                np.asarray(pis) > self.tau for pis in uncerts[-1]
            ]

            uncerts = minmax_normalize(uncerts[0])

            per_image_score = [
                np.asarray(uncerts[u]) * filter_based_on_sigmoid[u]
                for u in range(len(uncerts))
            ]
            filter_based_on_score = [
                (np.asarray(pis) <= np.mean(self.opt_thrs)) * (np.asarray(pis) > 0)
                for pis in per_image_score
            ]

        elif "alluncert" in self.selection_strategy:
            uncerts = [
                [pt[i] for pt in per_image_score]
                for i in range(len(per_image_score[0]))
            ]

            filter_based_on_sigmoid = [
                np.asarray(pis) > self.tau for pis in uncerts[-1]
            ]

            uncerts = uncerts[:-1]
            uncerts = minmax_normalize(
                [
                    list(
                        1
                        / np.mean([uncerts[0][i], uncerts[1][i], uncerts[2][i]], axis=0)
                    )
                    for i in range(len(uncerts[0]))
                ]
            )

            per_image_score = [
                np.asarray(uncerts[u]) * filter_based_on_sigmoid[u]
                for u in range(len(uncerts))
            ]
            filter_based_on_score = [pis > self.tau for pis in per_image_score]
        elif (
            "epuncert" in self.selection_strategy or "ental" in self.selection_strategy
        ):
            uncerts = [
                [pt[i] for pt in per_image_score]
                for i in range(len(per_image_score[0]))
            ]

            filter_based_on_sigmoid = [
                np.asarray(pis) > self.tau for pis in uncerts[-1]
            ]

            uncerts = uncerts[:-1]
            uncerts = minmax_normalize(
                [
                    list(1 / np.mean([uncerts[0][i], uncerts[1][i]], axis=0))
                    for i in range(len(uncerts[0]))
                ]
            )

            per_image_score = [
                np.asarray(uncerts[u]) * filter_based_on_sigmoid[u]
                for u in range(len(uncerts))
            ]
            filter_based_on_score = (
                filter_based_on_sigmoid  # [pis > self.tau for pis in per_image_score]
            )
        else:
            filter_based_on_score = [
                np.asarray(pis) > self.tau for pis in per_image_score
            ]
        pred_boxes = [
            np.asarray(pb)[fs]
            for pb, fs in zip(pred_boxes, filter_based_on_score)
            if sum(fs) > 0
        ]
        pred_classes = [
            np.asarray(pc)[fs]
            for pc, fs in zip(pred_classes, filter_based_on_score)
            if sum(fs) > 0
        ]
        pred_imgs_names = np.asarray(pred_imgs_names)[
            [np.any(fd) for fd in filter_based_on_score]
        ]
        if self.activate_pseudoscore:
            per_image_score = [
                np.asarray(pb)[fs]
                for pb, fs in zip(per_image_score, filter_based_on_score)
                if sum(fs) > 0
            ]
            return (pred_imgs_names, pred_classes, pred_boxes, per_image_score)
        else:
            return (pred_imgs_names, pred_classes, pred_boxes)

    def train(self, model_directory, num_examples_per_epoch, labeled_only=False):
        """Run model training

        Args:
            model_directory (str): Path to model directory
            num_examples_per_epoch (int): Number of images in training TFRecord
            labeled_only (bool, optional): Train only on labeled data. Defaults to False.
        """
        print("Running Training")
        script_folder = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_folder)

        if self.self_train and "teacher" not in model_directory:
            self.num_epochs = 20  # TODO: implement smart num epochs
            self.self_train_iter = 3

            self.pretrained_ckpt = self.model_directory
            for i in range(self.self_train_iter):
                print("Self-training iteration #" + str(i))
                model_name = "EXP_" + model_directory.split("/")[-1]
                infer_res_path = (
                    self.infer_results + "/SSAL/" + model_name + "/prediction_data.txt"
                )
                exp_model_path = (
                    self.general_path + "models/exported_models/SSL/" + model_name
                )
                calibration_path = (
                    self.general_path + "/results/calibration/" + model_name
                )
                if os.path.exists(model_directory) and not self.early_stopping:
                    print("Checking if training is over (no early stopping)")
                    while not os.path.exists(
                        model_directory + "/ckpt-" + str(self.num_epochs) + ".index"
                    ):
                        command = (
                            f"nohup python -m train_flags_SSL --ssl_method=STAC --train_file_pattern_labeled={self.train_tf_init} --train_file_pattern_unlabeled={self.train_tf_pseudo} --val_file_pattern={self.val_tf} --ratio={self.ratio} --model_name={self.model_name} --model_dir={model_directory} --batch_size={self.batch_size} --eval_samples={self.eval_samples} --num_epochs={self.num_epochs} --num_examples_per_epoch={num_examples_per_epoch} --pretrained_ckpt={self.pretrained_ckpt} --hparams={self.hparams} >> "
                            + model_directory
                            + "/train_log_"
                            + str(self.ratio)
                            + ".out"
                        )
                        subprocess.run(command, shell=True)
                        if os.path.exists(infer_res_path):
                            shutil.rmtree(
                                infer_res_path
                            )  # If inference done on model without full training
                        if os.path.exists(exp_model_path):
                            shutil.rmtree(exp_model_path)
                        if os.path.exists(calibration_path):
                            shutil.rmtree(calibration_path)
                else:
                    command = (
                        f"nohup python -m train_flags_SSL --ssl_method=STAC --train_file_pattern_labeled={self.train_tf_init} --train_file_pattern_unlabeled={self.train_tf_pseudo} --val_file_pattern={self.val_tf} --ratio={self.ratio} --model_name={self.model_name} --model_dir={model_directory} --batch_size={self.batch_size} --eval_samples={self.eval_samples} --num_epochs={self.num_epochs} --num_examples_per_epoch={num_examples_per_epoch} --pretrained_ckpt={self.pretrained_ckpt} --hparams={self.hparams} >> "
                        + model_directory
                        + "/train_log_"
                        + str(self.ratio)
                        + ".out"
                    )
                    if not self.early_stopping:
                        while not os.path.exists(
                            model_directory + "/ckpt-" + str(self.num_epochs) + ".index"
                        ):
                            subprocess.run(command, shell=True)
                    else:
                        subprocess.run(command, shell=True)

                if i < self.self_train_iter - 1:
                    self.model_directory = model_directory
                    self.pretrained_ckpt = self.model_directory
                    model_directory = model_directory.split(".")[0] + "." + (str(i + 1))

                    self.output_dir = (
                        self.general_path
                        + "datasets/"
                        + self.dataset
                        + "/pseudo_labels/num_labeled_"
                        + str(self.portion_labeled)
                        + "/selftrain_"
                        + self.teacher_setup
                        + self.selection_strategy
                        + "_thr_0"
                        + str(self.tau).split(".")[-1]
                        + self.added_name
                        + "."
                        + str(i)
                    )
                    self.train_tf_pseudo = (
                        self.tf_path
                        + "_train_"
                        + self.teacher_setup
                        + "selftrain_pseudo_thr_0"
                        + str(self.tau).split(".")[-1]
                        + "_"
                        + self.selection_strategy
                        + self.added_name
                        + "."
                        + str(i + 1)
                        + ".tfrecord"
                    )

                    pseudo_indices, n_dets = self.predict_teacher(
                        exp_model_path,
                        infer_res_path,
                        calibration_path,
                        model_name,
                        model_directory,
                    )
                    num_examples_per_epoch = self.num_labeled + len(pseudo_indices)
                    print_command = (
                        "Actual number of images for student is "
                        + str(self.num_labeled)
                        + " labeled and "
                        + str(len(pseudo_indices))
                        + " for unlabeled, total is "
                        + str(self.num_labeled + len(pseudo_indices))
                        + " Number of pseudo-dets is "
                        + str(n_dets)
                    )
                    print(print_command + "\n" + self.config)

                    os.makedirs(model_directory, exist_ok=True)
                    with open(model_directory + "/number_images.txt", "w") as text_file:
                        text_file.write(print_command)
                        text_file.write("\n")
                        text_file.write(self.config)

        else:
            if labeled_only:
                command = (
                    f"nohup python -m train_flags --train_file_pattern={self.train_tf_init} --val_file_pattern={self.val_tf} --model_name={self.model_name} --model_dir={model_directory} --batch_size={self.batch_size} --eval_samples={self.eval_samples} --num_epochs={self.num_epochs} --num_examples_per_epoch={num_examples_per_epoch} --pretrained_ckpt={self.pretrained_ckpt} --hparams={self.hparams} >> "
                    + model_directory
                    + "/train_log.out"
                )
            else:
                command = (
                    f"nohup python -m train_flags_SSL --ssl_method=STAC --train_file_pattern_labeled={self.train_tf_init} --train_file_pattern_unlabeled={self.train_tf_pseudo} --val_file_pattern={self.val_tf} --ratio={self.ratio} --model_name={self.model_name} --model_dir={model_directory} --batch_size={self.batch_size} --eval_samples={self.eval_samples} --num_epochs={self.num_epochs} --num_examples_per_epoch={num_examples_per_epoch} --pretrained_ckpt={self.pretrained_ckpt} --hparams={self.hparams} >> "
                    + model_directory
                    + "/train_log_"
                    + str(self.ratio)
                    + ".out"
                )

            os.makedirs(model_directory, exist_ok=True)
            if not self.early_stopping:
                while not os.path.exists(
                    model_directory + "/ckpt-" + str(self.num_epochs) + ".index"
                ):
                    subprocess.run(command, shell=True)
            else:
                subprocess.run(command, shell=True)

    def exp_calib_val_infer(
        self, prediction_indices, exporting_path, calibration_path, opt_params_path
    ):
        """Perform export/calibration/validation/inference

        Args:
            prediction_indices (list): List with current prediction indices
            exporting_path (str): Model exporting path
            calibration_path (str): Path to calibration results
            opt_params_path (str): Path to optimal parameters for combo uncertainty thresholding
        """
        infer_indices_txt = (
            self.tf_path
            + self.dataset
            + "_"
            + str(self.portion_labeled)
            + "_infer_indices"
            + self.added_name
            + ".txt"
        )
        if not os.path.exists(infer_indices_txt):
            all_ims = [
                os.path.basename(path)
                for path in sorted(glob.glob(self.data_dir_im + "/*"))
            ]
            prediction_indices = np.asarray(
                [all_ims.index(name) for name in self.im_names]
            )[
                prediction_indices
            ]  # Remap to original indices
            with open(infer_indices_txt, "w") as file:
                for index in prediction_indices:
                    file.write(f"{index}\n")

        data = {}
        data["eval_samples"] = self.eval_samples
        data["hparams"] = self.hparams
        data["infer_indices"] = infer_indices_txt
        data["infer_folder"] = self.data_dir_im
        data["model_dir"] = self.model_directory
        data["saved_model_dir"] = exporting_path
        data["val_file_pattern"] = self.val_tf
        data["video_path"] = self.general_path + "datasets/videos/xyz.mp4"
        # Write the modified data back to the YAML file
        if not os.path.exists(self.general_path + "/configs/inference/SSL/"):
            os.makedirs(self.general_path + "/configs/inference/SSL/")
        yaml_name = self.dataset + "_" + self.teacher_setup + self.added_name[1:]
        with open(
            self.general_path
            + "configs/inference/SSL/inference_"
            + yaml_name
            + ".yaml",
            "w",
        ) as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)
        print("Running Export/Calibration/Validation/Inference")
        # Export and infer, or infer if model already exported
        script_folder = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_folder)
        commands = [
            "python -m inspector --dataset "
            + yaml_name
            + " --mode 6 --general_path "
            + self.general_path
        ]

        if "combo" in self.selection_strategy and not os.path.exists(opt_params_path):
            commands.insert(
                0,
                "python -m inspector --dataset "
                + yaml_name
                + " --mode 3 --general_path "
                + self.general_path,
            )

        if "calib" in self.selection_strategy and not os.path.exists(calibration_path):
            commands.insert(
                0,
                "python -m inspector --dataset "
                + yaml_name
                + " --mode 2 --general_path "
                + self.general_path,
            )

        if not os.path.exists(exporting_path):
            commands.insert(
                0,
                "python -m inspector --dataset "
                + yaml_name
                + " --mode 0 --general_path "
                + self.general_path,
            )
        for cmd in commands:
            result = subprocess.run(cmd, shell=True)
            try:
                result.check_returncode()
            except subprocess.CalledProcessError:
                print(f"Command '{cmd}' failed. Stopping further execution.")
                break
        else:
            print("All commands executed successfully.")

    def get_tfrecord_generator(self):
        """Selects TFRecord generator based on dataset"""
        if self.dataset == "KITTI":
            return kitti_active_tfrecords
        elif self.dataset == "BDD100K":
            return bdd_active_tfrecords

    def predict_teacher(
        self,
        exp_model_path,
        infer_res_path,
        calibration_path,
        model_name,
        student_model_directory,
    ):
        """Predicts pseudo-labels using teacher model"""
        if "calib" in self.selection_strategy:
            opt_params_path = (
                self.general_path
                + "/results/validation/"
                + model_name
                + "/thresholding/calib/"
            )
        else:
            opt_params_path = (
                self.general_path
                + "/results/validation/"
                + model_name
                + "/thresholding/orig/"
            )
        val_path = (
            self.general_path
            + "/results/validation/"
            + model_name
            + "/validate_results.txt"
        )

        no_path = (
            (not os.path.exists(infer_res_path))
            or ("combo" in self.selection_strategy and not os.path.exists(val_path))
            or (
                "calib" in self.selection_strategy
                and not os.path.exists(calibration_path)
            )
        )
        # If any of the necessary files is non-existent, use the rest of the images as unlabeled
        if no_path:
            self.exp_calib_val_infer(
                self.available_indices[self.num_labeled :],
                exp_model_path,
                calibration_path,
                opt_params_path,
            )
        # If using combo uncertainty as a selection strategy get the optimal combination parameters
        if "combo" in self.selection_strategy:
            if uncertainty_analysis.FIX_CD:
                fix_cd = "cd"
            else:
                fix_cd = "fd"
            combo_data_txt = (
                fix_cd
                + "_"
                + str(uncertainty_analysis.FPR_TPR)
                + "_iou_"
                + str(np.min(uncertainty_analysis.IOU_THRS))
                + "_"
                + str(np.max(uncertainty_analysis.IOU_THRS))
                + ".txt"
            )
            if not os.path.exists(
                opt_params_path + "/optimal_params_" + combo_data_txt
            ):
                if "calib" in self.selection_strategy:
                    uncertainty_analysis.MainUncertViz(
                        model_name, general_path=self.general_path
                    )
                else:
                    uncertainty_analysis.MainUncertViz(
                        model_name, calib=False, general_path=self.general_path
                    )
            with open(
                opt_params_path + "/optimal_params_" + combo_data_txt, "r"
            ) as file:
                self.opt_params = [float(x.strip("[]")) for x in file.read().split(",")]

            with open(opt_params_path + "/optimal_thrs_" + combo_data_txt, "r") as file:
                self.opt_thrs = [float(x.strip("[]")) for x in file.read().split()]

        curr_learn_path = (
            self.general_path
            + "datasets/"
            + self.dataset
            + "/pseudo_labels/num_labeled_"
            + str(self.portion_labeled)
            + "/"
            + self.teacher_setup
            + self.selection_strategy
            + ".txt"
        )
        if os.path.exists(curr_learn_path) and "curr" in self.selection_strategy:
            output_dir_new = self.output_dir.split(
                self.selection_strategy.split("_")[0]
            )
            output_dir_new[-1] = output_dir_new[-1][
                1:
            ]  # to remove "_" at the beginning
            self.output_dir = "".join(output_dir_new)
            with open(curr_learn_path, "r") as file:
                lines = file.read().split("---\n")
                common_im_names = lines[0].strip().split("\n")
                rare_im_names = lines[1].strip().split("\n")
            common_im_names = np.asarray([im.split(".")[0] for im in common_im_names])
            rare_im_names = np.asarray([im.split(".")[0] for im in rare_im_names])

        # Select pseudo-labeled images
        if not os.path.exists(self.output_dir):
            extracted_data = self.score_image(infer_res_path, student_model_directory)
            if self.activate_pseudoscore:
                (
                    self.pred_imgs_names,
                    self.pred_classes,
                    self.pred_boxes,
                    self.pseudo_score,
                ) = extracted_data
            else:
                self.pred_imgs_names, self.pred_classes, self.pred_boxes = (
                    extracted_data
                )
            self.pred_classes = [
                [self.select_classes[int(c) - 1] for c in c_im]
                for c_im in self.pred_classes
            ]
            # Write predictions as GT
            if "KITTI" in self.dataset:
                n_dets = self.write_KITTI_pseudo_gt_txt()
            else:
                n_dets = self.write_BDD_pseudo_gt_json()
        else:
            if "KITTI" in self.dataset:
                self.pred_imgs_names = os.listdir(self.output_dir)
                self.pred_imgs_names = [p for p in self.pred_imgs_names if "txt" in p]
                n_dets = 0
                # Loop through each file in the directory
                for filename in self.pred_imgs_names:
                    if filename.endswith(".txt"):  # Check if the file is a .txt file
                        file_path = os.path.join(self.output_dir, filename)

                        with open(file_path, "r") as file:
                            for line in file:
                                if line.strip():  # Check if the line is not empty
                                    n_dets += 1

            else:
                with open(self.output_dir + "/pseudo_labels.json", "r") as file:
                    bdd_dets = json.load(file)
                    self.pred_imgs_names = [jitem["name"] for jitem in bdd_dets]
                    n_dets = np.sum([len(jitem["labels"]) for jitem in bdd_dets])

        cleaned_im_names = np.asarray([im.split(".")[0] for im in self.im_names])

        pseudo_indices = [
            np.where(pim.split(".")[0] == cleaned_im_names)[0][0]
            for pim in self.pred_imgs_names
        ]
        self.ratio = self.num_labeled
        if os.path.exists(curr_learn_path):
            common_tf_pseudo = self.train_tf_pseudo.split(".tfrecord")
            common_tf_pseudo = common_tf_pseudo[0] + "_common.tfrecord"
            common_indices = [
                np.where(pim.split(".")[0] == cleaned_im_names)[0][0]
                for pim in common_im_names
            ]
            if not os.path.exists(common_tf_pseudo):
                self.get_tfrecord_generator()(
                    data_dir=self.data_dir,
                    output_path=self.tf_path,
                    classes_to_use=self.select_classes,
                    label_map_path=self.label_path,
                    train_indices=common_indices,
                    current_iteration=self.train_tf_pseudo.split("_train_")[-1].split(
                        ".tf"
                    )[0]
                    + "_common",
                    pseudo=self.output_dir,
                )

            rare_tf_pseudo = self.train_tf_pseudo.split(".tfrecord")
            rare_tf_pseudo = rare_tf_pseudo[0] + "_rare.tfrecord"
            rare_indices = [
                np.where(pim.split(".")[0] == cleaned_im_names)[0][0]
                for pim in rare_im_names
            ]
            if not os.path.exists(rare_tf_pseudo):
                self.get_tfrecord_generator()(
                    data_dir=self.data_dir,
                    output_path=self.tf_path,
                    classes_to_use=self.select_classes,
                    label_map_path=self.label_path,
                    train_indices=rare_indices,
                    current_iteration=self.train_tf_pseudo.split("_train_")[-1].split(
                        ".tf"
                    )[0]
                    + "_rare",
                    pseudo=self.output_dir,
                )
        else:
            if not os.path.exists(self.train_tf_pseudo):
                self.get_tfrecord_generator()(
                    data_dir=self.data_dir,
                    output_path=self.tf_path,
                    classes_to_use=self.select_classes,
                    label_map_path=self.label_path,
                    train_indices=pseudo_indices,
                    current_iteration=self.train_tf_pseudo.split("_train_")[-1].split(
                        ".tf"
                    )[0],
                    pseudo=self.output_dir,
                )

        return pseudo_indices, n_dets

    def run(self):
        """Run STAC"""
        # Initial tfrecord with labeled data
        if self.teacher_strategy:
            teacher_setup = self.teacher_strategy.split("curriculum")
            if "learning" in self.teacher_strategy:
                teacher_added_name = teacher_setup[1].split("learning")[-1]
                self.teacher_setup = teacher_setup[0] + "cl_"
                self.train_tf_init = (
                    self.tf_path
                    + "/_train_"
                    + self.teacher_setup
                    + "curriculum_learning"
                    + teacher_added_name
                    + ".tfrecord"
                )
            else:
                self.teacher_setup = teacher_setup[0]
                teacher_added_name = ""
                if "rcc" in self.teacher_strategy:
                    self.tf_path = (
                        self.general_path
                        + "datasets/"
                        + self.dataset
                        + "/collage_crops/num_labeled_"
                        + str(self.portion_labeled)
                        + "/"
                        + self.teacher_setup
                        + "/"
                    )
                    if self.dataset == "KITTI":
                        self.train_tf_init = self.tf_path + "/_train.tfrecord"

                    else:
                        self.train_tf_init = self.tf_path + "/_train100k.tfrecord"
                else:
                    self.train_tf_init = (
                        self.tf_path + "/_train_" + self.teacher_setup + ".tfrecord"
                    )

            if "imscore" in self.teacher_strategy:
                teacher_scores = (
                    self.general_path
                    + "datasets/"
                    + self.dataset
                    + "/pseudo_labels/num_labeled_"
                    + str(self.portion_labeled)
                    + "/"
                    + self.teacher_setup.split("imscore")[0]
                    + "imscore"
                )
            else:
                teacher_scores = None
            teacher_name = (
                "_STAC_teacher"
                + str(self.portion_labeled)
                + "_"
                + self.teacher_setup
                + teacher_added_name
                + self.added_name
            )
            tf_added_name = self.teacher_setup
        else:
            self.train_tf_init = (
                self.tf_path
                + "_train_init_"
                + self.added_name.split("_")[-1]
                + ".tfrecord"
            )
            teacher_scores = None
            teacher_name = "_STAC_teacher" + str(self.portion_labeled) + self.added_name
            self.teacher_setup = ""
            teacher_setup = ["_"]
            tf_added_name = (
                self.teacher_setup + "init_" + self.added_name.split("_")[-1]
            )
        curr_learn_init_path = (
            self.general_path
            + "datasets/"
            + self.dataset
            + "/pseudo_labels/num_labeled_"
            + str(self.portion_labeled)
            + "/"
            + teacher_setup[0].split("_")[0]
            + "_curriculum_learning"
            + str(self.clw)
            + ".txt"
        )
        if (
            os.path.exists(curr_learn_init_path)
            and "curriculum_learning" in self.teacher_strategy
        ):
            cleaned_im_names = np.asarray([im.split(".")[0] for im in self.im_names])
            output_dir_new = self.tf_path + "/_train_" + self.teacher_setup
            with open(curr_learn_init_path, "r") as file:
                lines = file.read().split("---\n")
                common_im_names = lines[0].strip().split("\n")
                rare_im_names = lines[1].strip().split("\n")
            common_im_names = np.asarray([im.split(".")[0] for im in common_im_names])
            rare_im_names = np.asarray([im.split(".")[0] for im in rare_im_names])

            common_tf_pseudo = output_dir_new + "common.tfrecord"
            common_indices = [
                np.where(pim.split(".")[0] == cleaned_im_names)[0][0]
                for pim in common_im_names
            ]
            if not os.path.exists(common_tf_pseudo):
                self.get_tfrecord_generator()(
                    data_dir=self.data_dir,
                    output_path=self.tf_path,
                    classes_to_use=self.select_classes,
                    label_map_path=self.label_path,
                    train_indices=common_indices,
                    current_iteration=self.teacher_setup + "common",
                    pseudo=teacher_scores,
                )

            rare_tf_pseudo = output_dir_new + "rare.tfrecord"
            rare_indices = [
                np.where(pim.split(".")[0] == cleaned_im_names)[0][0]
                for pim in rare_im_names
            ]
            if not os.path.exists(rare_tf_pseudo):
                self.get_tfrecord_generator()(
                    data_dir=self.data_dir,
                    output_path=self.tf_path,
                    classes_to_use=self.select_classes,
                    label_map_path=self.label_path,
                    train_indices=rare_indices,
                    current_iteration=self.teacher_setup + "rare",
                    pseudo=teacher_scores,
                )
        else:
            if not os.path.exists(self.train_tf_init):
                with open(
                    self.tf_path
                    + "_train_init_"
                    + self.added_name.split("_")[-1]
                    + ".txt",
                    "w",
                ) as text_file:
                    text_file.write(
                        " ".join(map(str, self.available_indices[: self.num_labeled]))
                    )
                self.get_tfrecord_generator()(
                    data_dir=self.data_dir,
                    output_path=self.tf_path,
                    classes_to_use=self.select_classes,
                    label_map_path=self.label_path,
                    train_indices=self.available_indices[: self.num_labeled],
                    current_iteration=tf_added_name,
                    pseudo=teacher_scores,
                )

        # Check if model already trained else trian on labeled data
        self.model_directory = self.model_dir + self.dataset + teacher_name
        infer_res_path = (
            self.infer_results
            + "/SSAL/"
            + "EXP_"
            + self.dataset
            + teacher_name
            + "/prediction_data.txt"
        )
        exp_model_path = (
            self.general_path
            + "models/exported_models/SSL/EXP_"
            + self.dataset
            + teacher_name
        )
        calibration_path = (
            self.general_path
            + "/results/calibration/EXP_"
            + self.dataset
            + teacher_name
        )
        if os.path.exists(self.model_directory) and not self.early_stopping:
            print("Checking if training is over (no early stopping)")
            while not os.path.exists(
                self.model_directory + "/ckpt-" + str(self.num_epochs) + ".index"
            ):
                self.train(self.model_directory, self.num_labeled, labeled_only=True)
                if os.path.exists(infer_res_path):
                    shutil.rmtree(
                        infer_res_path
                    )  # If inference done on model without full training
                if os.path.exists(exp_model_path):
                    shutil.rmtree(exp_model_path)
                if os.path.exists(calibration_path):
                    shutil.rmtree(calibration_path)
            print("Training is indeed over")
        else:
            self.train(self.model_directory, self.num_labeled, labeled_only=True)

        if self.self_train:
            student_mode = "selftrain"
            tf_add = student_mode + "_"
            v_add = ".0"
        else:
            student_mode = "student"
            v_add = tf_add = ""

        if self.teacher_strategy and teacher_added_name:
            self.teacher_setup = self.teacher_setup[:-1] + teacher_added_name + "_"
        # Predict with teacher model
        self.output_dir = (
            self.general_path
            + "datasets/"
            + self.dataset
            + "/pseudo_labels/num_labeled_"
            + str(self.portion_labeled)
            + "/"
            + self.teacher_setup
            + self.selection_strategy
            + "_thr_0"
            + str(self.tau).split(".")[-1]
            + self.added_name
            + v_add
        )
        self.train_tf_pseudo = (
            self.tf_path
            + "_train_"
            + self.teacher_setup
            + tf_add
            + "pseudo_thr_0"
            + str(self.tau).split(".")[-1]
            + "_"
            + self.selection_strategy
            + self.added_name
            + v_add
            + ".tfrecord"
        )

        student_model_directory = (
            self.model_dir
            + self.dataset
            + "_STAC_"
            + self.teacher_setup
            + student_mode
            + str(self.portion_labeled)
            + "_thr_0"
            + str(self.tau).split(".")[-1]
            + "_"
            + self.selection_strategy
            + self.added_name
            + v_add
        )
        pseudo_indices, n_dets = self.predict_teacher(
            exp_model_path,
            infer_res_path,
            calibration_path,
            "EXP_" + self.dataset + teacher_name,
            student_model_directory,
        )

        # Train student with randaug on unlabeled images
        print_command = (
            "Actual number of images for student is "
            + str(self.num_labeled)
            + " labeled and "
            + str(len(pseudo_indices))
            + " for unlabeled, total is "
            + str(self.num_labeled + len(pseudo_indices))
            + " Number of pseudo-dets is "
            + str(n_dets)
        )
        print(print_command + "\n" + self.config)

        os.makedirs(student_model_directory, exist_ok=True)
        with open(student_model_directory + "/number_images.txt", "a") as text_file:
            text_file.write(print_command)
            text_file.write("\n")
            text_file.write(self.config)

        self.train(student_model_directory, self.num_labeled + len(pseudo_indices))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initializes STAC")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument(
        "--portion_labeled", type=int, default=25, help="Portion of labeled data"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.4,
        help="Threshold to filter predictions based on",
    )
    parser.add_argument(
        "--selection_strategy",
        type=str,
        default="score",
        help="Data selection strategy",
    )
    parser.add_argument(
        "--teacher_strategy",
        type=str,
        default="",
        help="Teacher selection strategy",
    )
    parser.add_argument(
        "--clw",
        type=int,
        default=10,
        help="Class weighting score",
    )
    parser.add_argument(
        "--added_name", type=str, default="_V", help="Add name for saving"
    )
    parser.add_argument("--version_num", type=int, default=0, help="Training version")
    parser.add_argument(
        "--num_epochs", type=int, default=200, help="Number of epochs per AL iteration"
    )
    parser.add_argument(
        "--training_method", type=str, default="baseline", help="Training method"
    )
    parser.add_argument(
        "--general_path",
        type=str,
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="Path to working directory",
    )
    parser.add_argument(
        "--early_stopping",
        type=bool,
        default=False,
        help="Activates early stopping instead of full training",
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=None,
        help="Random seed for all random operations",
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")

    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.dataset is None:
        select_gpu = "0"
        print("selected GPU: ", select_gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = select_gpu
        STAC().run()
    else:
        stac = STAC(
            dataset=args.dataset,
            portion_labeled=args.portion_labeled,
            tau=args.tau,
            selection_strategy=args.selection_strategy,
            added_name=args.added_name,
            version_num=args.version_num,
            num_epochs=args.num_epochs,
            training_method=args.training_method,
            early_stopping=args.early_stopping,
            general_path=args.general_path,
            rand_seed=args.rand_seed,
            clw=args.clw,
            teacher_strategy=args.teacher_strategy,
        )
        stac.run()
