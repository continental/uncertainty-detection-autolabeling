# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Specific for paper: Enhancing Active Learning for Object Detection through Similarity
Active learning main loop to train """


import argparse
import ast
import os
import shutil
import subprocess
import sys

import imagehash
import numpy as np
import tensorflow as tf
import uncertainty_analysis
import yaml
from PIL import Image
from utils_box import relativize_uncert

sys.path.insert(7, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.BDD100K.bdd_tf_creator import bdd_active_tfrecords
from datasets.KITTI.kitti_tf_creator import kitti_active_tfrecords


class ActiveLearning:
    """Active learning loop"""

    def __init__(
        self,
        dataset="KITTI",
        scoring_strategy="random",
        iteration_budget=[5, 5, 5, 10, 20, 30, 25],
        added_name="_V",
        version_num=0,
        num_epochs=200,
        early_stopping=False,
        general_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        rand_seed=None,
        hash_method="p",
        prune_thr=0,
    ):
        """Initializes active learning loop

        Args:
            dataset (str, optional): Dataset name. Defaults to "KITTI".
            scoring_strategy (str, optional): AL scoring strategy. Defaults to "random".
            iteration_budget (list, optional): Iteration budget to be added in each iteration. Defaults to [5,5,5,10,20,30,25].
            added_name (str, optional): Add name for saving. Defaults to "_V".
            version_num (int, optional): Training version. Defaults to 0.
            num_epochs (int, optional): Number of epochs per AL iteration. Defaults to 200.
            early_stopping (bool, optional): Activates early stopping instead of full training. Defaults to False.
            general_path (str, optional): Path to wdir. Defaults to parent folder of src.
            rand_seed (_type_, optional): Random seed for all random operations. Defaults to None.
            hash_method (str, optional): Hashing methods (perceptual vs wavelet). Defaults to "p".
            prune_thr (int, optional): Hamming distance threshold for pruning. Defaults to 0.
        """
        print(
            f"dataset: {dataset}, selection_mode: {scoring_strategy}, iteration_budget: {iteration_budget}, added_name: {added_name}, "
            f"version_num: {version_num}, num_epochs: {num_epochs}, rand_seed: {rand_seed}, hash_method: {hash_method}, prune_thr: {prune_thr}"
        )
        if rand_seed is not None:
            np.random.seed(rand_seed)
        self.dataset = dataset
        self.general_path = general_path + "/"
        self.scoring_strategy = scoring_strategy
        self.iteration_budget = iteration_budget
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
            self.im_names = sorted(
                tf.io.gfile.listdir(os.path.join(self.data_dir, "image_2"))
            )
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
            if "lossatt" in scoring_strategy:
                self.hparams = (
                    self.general_path + "configs/train/allclasses_lossatt.yaml"
                )
            elif "baseline" in scoring_strategy:
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
            if "lossatt" in scoring_strategy:
                self.hparams = (
                    self.general_path + "configs/train/allclasses_lossatt_BDD.yaml"
                )
            elif "baseline" in scoring_strategy:
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
        self.current_iteration = 0
        self.hash_method = hash_method
        # Universal training parameters
        self.batch_size = 8
        self.early_stopping = early_stopping
        self.num_epochs = num_epochs
        self.model_dir = self.general_path + "models/trained_models/AL/"
        self.model_name = "efficientdet-d0"
        self.pretrained_ckpt = "efficientdet-d0"
        if "prune" in scoring_strategy:
            train_im_names, len_train_ims, kept_inds = self.extract_hash_matrix(
                len_train_ims
            )
        # Universal AL config
        self.infer_results = (
            self.general_path + "/results/inference/"
        )  # Path to model inference results
        self.ims_per_iter = [
            int(len_train_ims * self.iteration_budget[i] / 100)
            for i in range(len(self.iteration_budget))
        ]
        self.ims_per_iter[-1] = len_train_ims - sum(self.ims_per_iter[:-1])
        print_budget = f"# added images per iteration: {self.ims_per_iter}, actual # per iter {[sum(self.ims_per_iter[:i+1]) for i in range(len(self.ims_per_iter))]}"
        if "prune" in scoring_strategy:
            print_budget += f", pruned: {len(train_im_names)-len(kept_inds)}"
        print(print_budget)
        self.num_per_iter = self.ims_per_iter[0]
        self.version_num = version_num
        self.added_name = added_name + str(self.version_num)
        self.tf_active_path = (
            self.general_path
            + "datasets/"
            + self.dataset
            + "/tf_active/"
            + self.scoring_strategy
            + "/V"
            + str(self.version_num)
            + "/"
        )
        if not os.path.exists(self.tf_active_path):
            os.makedirs(self.tf_active_path)

    def extract_hash_matrix(self, len_train_ims):
        """Calculates hash values and hash matrix and returns the adjusted iteration budget based on the removed images via distance threshold"""
        train_im_names = [self.im_names[i] for i in self.available_indices]
        # Random pruning
        if "rand" in self.scoring_strategy:
            subset_size = int((1 - prune_thr) * len_train_ims)
            print("Random pruning:" + str(subset_size))
            kept_inds = np.random.choice(
                np.arange(len_train_ims), size=subset_size, replace=False
            )
        else:
            # Get or generate hash values
            pruning_path = self.data_dir + "/" + hash_method
            hash_values_path = pruning_path + "_hash_ims.npy"
            if os.path.exists(hash_values_path):
                self.hash_values = np.load(hash_values_path, allow_pickle=True)
            else:
                hashes = []
                for im_name in train_im_names:
                    im = Image.open(self.data_dir_im + im_name).resize((512, 256))
                    if self.hash_method == "p":
                        hash_value = imagehash.phash(im)
                    else:
                        hash_value = imagehash.whash(im)
                    hashes.append(hash_value)
                    im.close()
                self.hash_values = np.asarray(hashes)
                np.save(hash_values_path, self.hash_values, allow_pickle=True)

            if os.path.exists(
                pruning_path + "_pruning_kept_inds_thr" + str(prune_thr) + ".npy"
            ):
                kept_inds = np.load(
                    pruning_path + "_pruning_kept_inds_thr" + str(prune_thr) + ".npy",
                    allow_pickle=True,
                )
            else:
                if os.path.exists(pruning_path + "_distance_matrix.npy"):
                    self.dist_matrix = np.load(
                        pruning_path + "_distance_matrix.npy", allow_pickle=True
                    )
                else:
                    self.dist_matrix = np.asarray(
                        [
                            self.hash_values - self.hash_values[i]
                            for i in range(len(self.hash_values))
                        ]
                    )
                    np.save(
                        pruning_path + "_distance_matrix.npy",
                        self.dist_matrix,
                        allow_pickle=True,
                    )

                if os.path.exists(
                    pruning_path + "_prune_duplicates_thr" + str(prune_thr) + ".npy"
                ):
                    duplicates = np.load(
                        pruning_path
                        + "_prune_duplicates_thr"
                        + str(prune_thr)
                        + ".npy",
                        allow_pickle=True,
                    )
                else:
                    duplicates = []
                    max_dist = np.max(self.dist_matrix)
                    for i in range(len(self.hash_values)):
                        duplicate_indices = np.where(
                            self.dist_matrix[i] <= max_dist * prune_thr
                        )[0]
                        if len(duplicate_indices) > 1:
                            duplicates.append(duplicate_indices)
                    np.save(
                        pruning_path
                        + "_prune_duplicates_thr"
                        + str(prune_thr)
                        + ".npy",
                        duplicates,
                        allow_pickle=True,
                    )

                # plot_samples([Image.open(self.data_dir_im+im_name).resize((512, 256)) for im_name in np.asarray(self.im_names)[duplicates[0]]])
                select_representative = [
                    np.random.choice(duplicates[i]) for i in range(len(duplicates))
                ]
                removed_ind = np.unique(
                    np.concatenate(
                        [
                            duplicates[i][duplicates[i] != select_representative[i]]
                            for i in range(len(duplicates))
                        ]
                    )
                )
                kept_inds = [
                    i for i in np.arange(len_train_ims) if i not in removed_ind
                ]
                np.save(
                    pruning_path + "_pruning_kept_inds_thr" + str(prune_thr) + ".npy",
                    kept_inds,
                    allow_pickle=True,
                )
                self.hash_values = self.hash_values[kept_inds]

        self.available_indices = [
            self.available_indices[new_idx] for new_idx in kept_inds
        ]
        # One time prune, no iterations
        if "full_prune" in self.scoring_strategy:
            self.iteration_budget = [100]
        else:
            new_iter_budget = (
                np.asarray(self.iteration_budget)
                * (len_train_ims)
                / len(self.available_indices)
            )
            self.iteration_budget = new_iter_budget[new_iter_budget.cumsum() <= 100]
        len_train_ims = len(self.available_indices)
        return train_im_names, len_train_ims, kept_inds

    @staticmethod
    def min_max_scaler(data):
        """Scales the data from 0 to 1 based on min and max"""
        return [(x - min(data)) / (max(data) - min(data)) for x in data]

    @staticmethod
    def z_score_normalization(data):
        """Standardize the data"""
        return (data - np.mean(data)) / np.std(data)

    def generate_random(self, current_indices):
        """Generates random indices for training, which are neither already selected nor in the validation set"""
        # Keep generating random values until collected unique values that are not in val_indices
        train_indices = []
        while len(train_indices) < self.num_per_iter:
            random_index = np.random.choice(range(len(self.im_names)))
            if (
                random_index in self.available_indices
                and random_index not in train_indices
                and random_index not in current_indices
            ):
                train_indices.append(random_index)
        return train_indices

    def get_tfrecord_generator(self):
        """Selects TFRecord generator based on dataset"""
        if self.dataset == "KITTI":
            return kitti_active_tfrecords
        elif self.dataset == "BDD100K":
            return bdd_active_tfrecords

    def generate_tfrecord(self):
        """Generates training indices then TFRecord based on the indices"""
        # Check if iteration exists before
        current_indices = []
        if self.current_iteration > 0:
            with open(
                self.tf_active_path
                + self.dataset
                + "_train_indices_"
                + self.scoring_strategy
                + "_"
                + str(self.current_iteration - 1)
                + self.added_name
                + ".txt",
                "r",
            ) as file:
                for line in file:
                    current_indices.append(int(line.strip()))
        if self.current_iteration < len(self.iteration_budget) - 1:
            # Random select
            if "random" in self.scoring_strategy or self.current_iteration == 0:
                train_indices = self.generate_random(current_indices)
            # AL strategy
            else:
                train_indices = self.infer_collect(
                    current_indices
                )  # > 0 and before last iteration
            train_indices = current_indices + train_indices
        else:
            train_indices = self.available_indices  # Last iteration train on all
        with open(
            self.tf_active_path
            + self.dataset
            + "_train_indices_"
            + self.scoring_strategy
            + "_"
            + str(self.current_iteration)
            + self.added_name
            + ".txt",
            "w",
        ) as file:
            for index in train_indices:
                file.write(f"{index}\n")
        # Generate TFRecord if not already generated
        if os.path.exists(
            self.tf_active_path
            + "_train_"
            + str(self.current_iteration)
            + self.added_name
            + ".tfrecord"
        ):
            return sum(self.ims_per_iter[: self.current_iteration + 1])
        else:
            return self.get_tfrecord_generator()(
                data_dir=self.data_dir,
                output_path=self.tf_active_path,
                classes_to_use=self.select_classes,
                label_map_path=self.label_path,
                train_indices=train_indices,
                current_iteration=str(self.current_iteration) + self.added_name,
            )

    def exp_calib_val_infer(
        self, current_indices, exporting_path, calibration_path, opt_params_path
    ):
        """Perform export/calibration/validation/inference

        Args:
            current_indices (list): List with current training indices
            exporting_path (str): Model exporting path
            calibration_path (str): Path to calibration results
            opt_params_path (str): Path to optimal parameters for combo uncertainty thresholding
        """
        infer_indices_txt = (
            self.tf_active_path
            + self.dataset
            + "_infer_indices_"
            + self.scoring_strategy
            + "_"
            + str(self.current_iteration - 1)
            + self.added_name
            + ".txt"
        )
        available_indices = np.arange(len(self.im_names))
        prediction_indices = [
            ind
            for ind in available_indices
            if ind not in current_indices and ind in self.available_indices
        ]  # Prediction/unlabeled indices
        all_ims = sorted(tf.io.gfile.listdir(self.data_dir_im))
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
        data["model_dir"] = (
            self.model_dir
            + self.dataset
            + "_"
            + self.scoring_strategy
            + "_"
            + str(self.current_iteration - 1)
            + self.added_name
        )
        data["saved_model_dir"] = exporting_path
        data["val_file_pattern"] = self.val_tf
        data["video_path"] = self.general_path + "datasets/videos/xyz.mp4"
        # Write the modified data back to the YAML file
        if not os.path.exists(self.general_path + "/configs/inference/AL/"):
            os.makedirs(self.general_path + "/configs/inference/AL/")
        yaml_name = (
            self.dataset
            + "_"
            + self.scoring_strategy
            + "_"
            + str(self.current_iteration - 1)
            + self.added_name
        )
        with open(
            self.general_path + "configs/inference/AL/inference_" + yaml_name + ".yaml",
            "w",
        ) as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)
        print("Running Export/Calibration/Validation/Inference")
        # Export and infer, or infer if model already exported
        script_folder = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_folder)
        commands = [
            f"python -m inspector --dataset "
            + yaml_name
            + " --mode 6 --general_path "
            + self.general_path
        ]

        if "combo" in self.scoring_strategy and not os.path.exists(opt_params_path):
            commands.insert(
                0,
                f"python -m inspector --dataset "
                + yaml_name
                + " --mode 3 --general_path "
                + self.general_path,
            )

        if "calib" in self.scoring_strategy and not os.path.exists(calibration_path):
            commands.insert(
                0,
                f"python -m inspector --dataset "
                + yaml_name
                + " --mode 2 --general_path "
                + self.general_path,
            )

        if not os.path.exists(exporting_path):
            commands.insert(
                0,
                f"python -m inspector --dataset "
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

    def score_image(self, inference_path):
        """Selects samples based on inference results and an AL strategy"""
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
        # Get per image score
        while i < len(detections):
            curr_img_name = detections[i]["image_name"]
            temp_al_score = []
            if "alluncert" in self.scoring_strategy or "sota" in self.scoring_strategy:
                temp_al_score = [[], [], []]
            elif (
                "epuncert" in self.scoring_strategy or "ental" in self.scoring_strategy
            ):
                temp_al_score = [[], []]

            # Iterate over detections for current image
            temp_classes = []
            temp_boxes = []
            while curr_img_name == img_name:
                if "calib" in self.scoring_strategy:
                    box_calib_mode = "iso_perclscoo_"
                    class_calib_mode = "iso_percls_"
                    if "box" in self.scoring_strategy:
                        add_mode = box_calib_mode
                    else:
                        add_mode = class_calib_mode
                else:
                    if (
                        "box" in self.scoring_strategy
                        or "class" in self.scoring_strategy
                    ):
                        add_mode = "uncalib_"
                    else:
                        add_mode = ""

                if "combo" in self.scoring_strategy:
                    if "calib" in self.scoring_strategy:
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
                    temp_al_score.append(combo_uncert)

                elif (
                    "alluncert" in self.scoring_strategy
                    or "sota" in self.scoring_strategy
                ):
                    if "calib" in self.scoring_strategy:
                        temp_al_score[0].append(
                            np.mean(
                                relativize_uncert(
                                    [detections[i]["bbox"]],
                                    [detections[i][box_calib_mode + "mcbox"]],
                                )
                            )
                        )
                        temp_al_score[1].append(
                            np.mean(
                                relativize_uncert(
                                    [detections[i]["bbox"]],
                                    [detections[i][box_calib_mode + "albox"]],
                                )
                            )
                        )
                        temp_al_score[2].append(
                            np.mean(detections[i][class_calib_mode + "mcclass"])
                        )
                    else:
                        temp_al_score[0].append(
                            np.mean(
                                relativize_uncert(
                                    [detections[i]["bbox"]],
                                    [detections[i]["uncalib_mcbox"]],
                                )
                            )
                        )
                        temp_al_score[1].append(
                            np.mean(
                                relativize_uncert(
                                    [detections[i]["bbox"]],
                                    [detections[i]["uncalib_albox"]],
                                )
                            )
                        )
                        temp_al_score[2].append(
                            np.mean(detections[i]["uncalib_mcclass"])
                        )

                elif "epuncert" in self.scoring_strategy:
                    if "calib" in self.scoring_strategy:
                        temp_al_score[0].append(
                            np.mean(
                                relativize_uncert(
                                    [detections[i]["bbox"]],
                                    [detections[i][box_calib_mode + "mcbox"]],
                                )
                            )
                        )
                        temp_al_score[1].append(
                            np.mean(detections[i][class_calib_mode + "mcclass"])
                        )
                    else:
                        temp_al_score[0].append(
                            np.mean(
                                relativize_uncert(
                                    [detections[i]["bbox"]],
                                    [detections[i]["uncalib_mcbox"]],
                                )
                            )
                        )
                        temp_al_score[1].append(
                            np.mean(detections[i]["uncalib_mcclass"])
                        )

                elif "ental" in self.scoring_strategy:
                    if "calib" in self.scoring_strategy:
                        temp_al_score[0].append(
                            np.mean(
                                relativize_uncert(
                                    [detections[i]["bbox"]],
                                    [detections[i][box_calib_mode + "albox"]],
                                )
                            )
                        )
                        temp_al_score[1].append(
                            detections[i][class_calib_mode + "entropy"]
                        )
                    else:
                        temp_al_score[0].append(
                            np.mean(
                                relativize_uncert(
                                    [detections[i]["bbox"]],
                                    [detections[i]["uncalib_albox"]],
                                )
                            )
                        )
                        temp_al_score[1].append(detections[i]["entropy"])

                else:
                    if add_mode + self.scoring_strategy.split("_")[-1] in detections[i]:
                        sel_metric = detections[i][
                            add_mode + self.scoring_strategy.split("_")[-1]
                        ]
                        if (
                            "box" in self.scoring_strategy
                            and "norm" in self.scoring_strategy
                        ):
                            temp_al_score.append(
                                np.mean(
                                    relativize_uncert(
                                        [detections[i]["bbox"]], [sel_metric]
                                    )
                                )
                            )
                        elif type(sel_metric) == float:
                            temp_al_score.append(sel_metric)
                        else:
                            temp_al_score.append(np.mean(sel_metric))
                    else:
                        temp_al_score.append(detections[i]["det_score"])
                temp_classes.append(detections[i]["class"])
                temp_boxes.append(detections[i]["bbox"])
                i += 1  # Check if last detection
                if i == len(detections):
                    break
                curr_img_name = detections[i]["image_name"]
                pred_imgs_names.append(detections[i]["image_name"])

            # Aggregate detections to one score per image
            if isinstance(temp_al_score[0], list):
                if "mean" in self.scoring_strategy:
                    per_image_score.append([np.mean(tm) for tm in temp_al_score])
                else:
                    per_image_score.append([np.max(tm) for tm in temp_al_score])
            else:
                if "mean" in self.scoring_strategy:
                    per_image_score.append(np.mean(temp_al_score))
                else:
                    per_image_score.append(np.max(temp_al_score))

            pred_classes.append(temp_classes)
            pred_boxes.append(temp_boxes)
            img_name = curr_img_name

        # Combine uncertainties if multiple approaches
        if isinstance(per_image_score[0], list) and all(
            len(sublist) == len(per_image_score[0]) for sublist in per_image_score
        ):  # Check if all sublists have same shape to add them together
            if "highep_lowal" in self.scoring_strategy:
                per_image_score = np.asarray(
                    [
                        self.min_max_scaler(np.asarray(per_image_score)[:, i])
                        for i in range(len(per_image_score[0]))
                    ]
                )
                ep = np.sum([per_image_score[i] for i in [0, 2]], axis=0)
                al = per_image_score[1]
                per_image_score = (
                    ep - al
                )  # When this is sorted for highest uncertainty, the higher the ep in comparison to al the higher the diference
            elif "sota" in self.scoring_strategy:
                per_image_score = np.max(
                    [
                        self.z_score_normalization(np.asarray(per_image_score)[:, i])
                        for i in range(len(per_image_score[0]))
                    ],
                    axis=0,
                )
            else:
                per_image_score = np.sum(
                    [
                        self.min_max_scaler(np.asarray(per_image_score)[:, i])
                        for i in range(len(per_image_score[0]))
                    ],
                    axis=0,
                )
        return per_image_score, pred_classes, np.unique(pred_imgs_names)

    def select_images(self, inference_path):
        """Selects AL batch of images for labeling"""
        # Get per image score
        per_image_al_score, pred_classes, pred_imgs_names = self.score_image(
            inference_path
        )
        # Add class balancing weight
        if "perc" in self.scoring_strategy:
            class_names = np.unique(np.concatenate(pred_classes))
            n_ideal_classes = np.arange(np.max(class_names)) + 1
            class_distribution = [
                sum(np.concatenate(pred_classes) == cls) for cls in class_names
            ]
            weights = np.asarray(
                [
                    sum(class_distribution) / class_distribution[class_idx]
                    for class_idx in range(len(class_names))
                ]
            )
            weights = np.insert(
                weights,
                [int(i - 1) for i in n_ideal_classes if i not in class_names],
                0,
            )  # To compensate for missing classes
            # Mean class weight based score per image, then multiply with AL per image score
            per_image_cls_score = [
                np.mean(
                    [
                        weights[int(np.unique(im_cls)[i] - 1)]
                        for i in range(len(np.unique(im_cls)))
                    ]
                )
                for im_cls in pred_classes
            ]
            per_image_al_score = np.multiply(per_image_cls_score, per_image_al_score)

        if "nee" in self.scoring_strategy:
            n = 5
            batch_size = self.num_per_iter // n
            remainder = self.num_per_iter % n
            selected_indices = []
            sorted_indices = np.argsort(per_image_al_score)
            bins = np.array_split(sorted_indices, n)
            # Perform exploration and exploitation for (n-1) bins
            for i in range(n - 1):
                bin_indices = bins[i][-batch_size:]
                selected_indices.extend(bin_indices)
            # For the last bin, select the lowest scored images for exploitation, distribute the remainder evenly
            last_bin_indices = bins[-1][: batch_size + remainder]
            selected_indices.extend(last_bin_indices)
            selected_images = [
                x.split(".")[0] for x in pred_imgs_names[selected_indices]
            ]
        else:
            sorted_images = [
                x.split(".")[0]
                for _, x in sorted(
                    zip(per_image_al_score, pred_imgs_names), key=lambda pair: pair[0]
                )
            ]
            if "bottomk" in self.scoring_strategy:
                selected_images = sorted_images[
                    : self.num_per_iter
                ]  # Bottom-k highest entropy
            else:
                selected_images = sorted_images[
                    -self.num_per_iter :
                ]  # Top-k highest entropy
        new_training_indices = [
            index
            for index, item in enumerate(self.im_names)
            if item.split(".")[0] in selected_images
        ]  # Map back the indices
        return new_training_indices

    def infer_collect(self, current_indices):
        """Selects training indices based on AL strategy"""
        # Read train indices and extract unlabeled indices for inference
        inference_path = (
            self.infer_results
            + "/AL/"
            + "EXP_"
            + self.dataset
            + "_"
            + self.scoring_strategy
            + "_"
            + str(self.current_iteration - 1)
            + self.added_name
            + "/prediction_data.txt"
        )
        calibration_path = (
            self.general_path
            + "/results/calibration/EXP_"
            + self.dataset
            + "_"
            + self.scoring_strategy
            + "_"
            + str(self.current_iteration - 1)
            + self.added_name
        )
        exporting_path = (
            self.general_path
            + "models/exported_models/AL/EXP_"
            + self.dataset
            + "_"
            + self.scoring_strategy
            + "_"
            + str(self.current_iteration - 1)
            + self.added_name
        )
        if "calib" in self.scoring_strategy:
            opt_params_path = (
                self.general_path
                + "/results/validation/EXP_"
                + self.dataset
                + "_"
                + self.scoring_strategy
                + "_"
                + str(self.current_iteration - 1)
                + self.added_name
                + "/thresholding/calib/"
            )
        else:
            opt_params_path = (
                self.general_path
                + "/results/validation/EXP_"
                + self.dataset
                + "_"
                + self.scoring_strategy
                + "_"
                + str(self.current_iteration - 1)
                + self.added_name
                + "/thresholding/orig/"
            )
        no_path = (
            not os.path.exists(inference_path)
            or (
                "combo" in self.scoring_strategy and not os.path.exists(opt_params_path)
            )
            or (
                "calib" in self.scoring_strategy
                and not os.path.exists(calibration_path)
            )
        )
        # If any of the necessary files is non-existent
        if no_path:
            self.exp_calib_val_infer(
                current_indices, exporting_path, calibration_path, opt_params_path
            )
        # If using combo uncertainty as a selection strategy get the optimal combination parameters
        if "combo" in self.scoring_strategy:
            combo_data_txt = (
                opt_params_path
                + "/optimal_params_"
                + FIX_CD
                + "_"
                + str(FPR_TPR)
                + "_iou_"
                + str(np.min(IOU_THRS))
                + "_"
                + str(np.max(IOU_THRS))
                + ".txt"
            )
            if not os.path.exists(combo_data_txt):
                combo_path = (
                    "EXP_"
                    + self.dataset
                    + "_"
                    + self.scoring_strategy
                    + "_"
                    + str(self.current_iteration - 1)
                    + self.added_name
                )
                if "calib" in self.scoring_strategy:
                    uncertainty_analysis.MainUncertViz(
                        combo_path, general_path=self.general_path
                    )
                else:
                    uncertainty_analysis.MainUncertViz(
                        combo_path, calib=False, general_path=self.general_path
                    )
            with open(combo_data_txt, "r") as file:
                self.opt_params = [float(x.strip("[]")) for x in file.read().split(",")]
        return self.select_images(inference_path)

    def train(self, num_examples_per_epoch):
        """Run model training

        Args:
            num_examples_per_epoch (int): Number of images in training TFRecord
        """
        print("Running Training")
        train_tf = (
            self.tf_active_path
            + "_train_"
            + str(self.current_iteration)
            + self.added_name
            + ".tfrecord"
        )
        if "full_" in self.scoring_strategy:
            model_iter = 6
        else:
            model_iter = self.current_iteration
        model_directory = (
            self.model_dir
            + self.dataset
            + "_"
            + self.scoring_strategy
            + "_"
            + str(model_iter)
            + self.added_name
        )

        script_folder = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_folder)

        if not os.path.exists(model_directory):  # To save train log
            os.makedirs(model_directory)
        command = (
            f"nohup python -m train_flags --train_file_pattern={train_tf} --val_file_pattern={self.val_tf} --model_name={self.model_name} --model_dir={model_directory} --batch_size={self.batch_size} --eval_samples={self.eval_samples} --num_epochs={self.num_epochs} --num_examples_per_epoch={num_examples_per_epoch} --pretrained_ckpt={self.pretrained_ckpt} --hparams={self.hparams} >> "
            + model_directory
            + "/train_log_"
            + str(self.current_iteration)
            + ".out"
        )
        subprocess.run(command, shell=True)

    def run(self):
        """Main function to run AL iterations"""
        for i in range(0, len(self.iteration_budget)):
            print("Running iteration " + str(i))
            self.current_iteration = i
            # Continue a failed training before jumping to the next iteration otherwise inference is wrong
            model_directory = (
                self.model_dir
                + self.dataset
                + "_"
                + self.scoring_strategy
                + "_"
                + str(i)
                + self.added_name
            )
            if os.path.exists(model_directory) and not self.early_stopping:
                print("Checking if training is over (no early stopping)")
                while not os.path.exists(
                    model_directory + "/ckpt-" + str(self.num_epochs) + ".index"
                ):
                    num_im = sum(self.ims_per_iter[: self.current_iteration + 1])
                    self.train(num_im)
                    infer_res_path = (
                        self.infer_results
                        + "/AL/"
                        + "EXP_"
                        + self.dataset
                        + "_"
                        + self.scoring_strategy
                        + "_"
                        + str(self.current_iteration)
                        + self.added_name
                        + "/prediction_data.txt"
                    )
                    exp_model_path = (
                        self.general_path
                        + "models/exported_models/AL/EXP_"
                        + self.dataset
                        + "_"
                        + self.scoring_strategy
                        + "_"
                        + str(self.current_iteration)
                        + self.added_name
                    )
                    if os.path.exists(infer_res_path):
                        os.remove(
                            infer_res_path
                        )  # If inference done on model without full training
                    if os.path.exists(exp_model_path):
                        shutil.rmtree(exp_model_path)
                print("Training is indeed over")
                continue
            else:
                if self.current_iteration > 0:
                    print(
                        "Checking if training of previous model is over (no early stopping)"
                    )
                    prev_model_directory = (
                        self.model_dir
                        + self.dataset
                        + "_"
                        + self.scoring_strategy
                        + "_"
                        + str(i - 1)
                        + self.added_name
                    )
                    self.current_iteration = i - 1
                    while not os.path.exists(
                        prev_model_directory
                        + "/ckpt-"
                        + str(self.num_epochs)
                        + ".index"
                    ):
                        num_im = sum(self.ims_per_iter[: self.current_iteration + 1])
                        self.train(num_im)
                        infer_res_path = (
                            self.infer_results
                            + "/AL/"
                            + "EXP_"
                            + self.dataset
                            + "_"
                            + self.scoring_strategy
                            + "_"
                            + str(self.current_iteration)
                            + self.added_name
                            + "/prediction_data.txt"
                        )
                        exp_model_path = (
                            self.general_path
                            + "models/exported_models/AL/EXP_"
                            + self.dataset
                            + "_"
                            + self.scoring_strategy
                            + "_"
                            + str(self.current_iteration)
                            + self.added_name
                        )
                        if os.path.exists(infer_res_path):
                            os.remove(
                                infer_res_path
                            )  # If inference done on model without full training
                        if os.path.exists(exp_model_path):
                            shutil.rmtree(exp_model_path)
                    print("Training previous model is indeed over")
                    self.current_iteration = i
            self.num_per_iter = self.ims_per_iter[i]
            num_im = self.generate_tfrecord()
            # Copy warm-up model if available as it is the same to accelerate training
            warm_up_model = (
                model_directory.split(self.dataset)[0]
                + self.dataset
                + "_entropy_0"
                + self.added_name
            )
            if (
                "prune" not in self.scoring_strategy
                and self.current_iteration == 0
                and os.path.exists(warm_up_model)
                and os.path.exists(
                    warm_up_model + "/ckpt-" + str(self.num_epochs) + ".index"
                )
            ):
                shutil.copytree(warm_up_model, model_directory)
                warm_up_infer = (
                    self.infer_results
                    + "/AL/"
                    + "EXP_"
                    + self.dataset
                    + "_entropy_0"
                    + self.added_name
                )
                if os.path.exists(warm_up_infer + "/prediction_data.txt"):
                    shutil.copytree(
                        warm_up_infer,
                        self.infer_results
                        + "/AL/"
                        + "EXP_"
                        + self.dataset
                        + "_"
                        + self.scoring_strategy
                        + "_0"
                        + self.added_name,
                    )
            else:
                self.train(num_im)
            print("Iteration " + str(self.current_iteration) + " over.")
        print("AL iterations are over.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--dataset", type=str, help="Select a dataset")
    parser.add_argument("--selection_mode", type=str, help="Select AL scoring function")
    parser.add_argument("--iteration_budget", help="Select numner of AL iterations")
    parser.add_argument("--added_name", type=str, help="Add a name for saving")
    parser.add_argument("--version_num", type=int, help="Select training version")
    parser.add_argument(
        "--num_epochs", type=int, help="Select numnber of training epochs"
    )
    parser.add_argument(
        "--early_stopping", type=bool, help="Select if fixed or with early stopping"
    )
    parser.add_argument("--general_path", type=str, help="Path to working space")
    parser.add_argument(
        "--rand_seed", help="Integer to fix the seed at, otherwise not fixed"
    )
    parser.add_argument("--hash_method", help="Hashing method, fixed to perceptual")
    parser.add_argument("--opt_params", help="Hyperparameters for combo uncert")
    parser.add_argument("--prune_thr", help="Hyperparameters for combo uncert")

    args = parser.parse_args()

    if args.prune_thr != "null" and args.prune_thr is not None:
        prune_thr = float(args.prune_thr)
    else:
        prune_thr = 0
    if args.rand_seed != "null" and args.rand_seed is not None:
        rand_seed = int(args.rand_seed)
    else:
        rand_seed = None
    if args.hash_method != "null" and args.hash_method is not None:
        hash_method = args.hash_method
    else:
        hash_method = "p"
    # Parameters for combo uncertainty if used
    if args.opt_params != "null" and args.opt_params is not None:
        opt_params = eval(args.opt_params)
        uncertainty_analysis.FPR_TPR = FPR_TPR = opt_params[0]
        uncertainty_analysis.FIX_CD = opt_params[1]
        if uncertainty_analysis.FIX_CD:
            FIX_CD = "cd"
        else:
            FIX_CD = "fd"
        uncertainty_analysis.IOU_THRS = IOU_THRS = eval(opt_params[2])
    else:
        from hparams_config import default_detection_configs

        uncertainty_analysis.FPR_TPR = FPR_TPR = default_detection_configs().thr_fpr_tpr
        uncertainty_analysis.FIX_CD = default_detection_configs().thr_cd
        if uncertainty_analysis.FIX_CD:
            FIX_CD = "cd"
        else:
            FIX_CD = "fd"
        uncertainty_analysis.IOU_THRS = IOU_THRS = (
            default_detection_configs().thr_iou_thrs
        )
    print("Opt_params: ", FPR_TPR, FIX_CD, IOU_THRS)

    if None in [
        args.dataset,
        args.selection_mode,
        args.iteration_budget,
        args.added_name,
        args.version_num,
        args.num_epochs,
    ]:
        selected_gpu = "6"
        print("selected GPU: ", selected_gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu
        ActiveLearning(
            dataset="KITTI",
            scoring_strategy="entropy",
            version_num=0,
            added_name="_seed21_V",
            hash_method="p",
            rand_seed=21,
            prune_thr=0.1,
        ).run()
    else:
        ActiveLearning(
            dataset=args.dataset,
            scoring_strategy=args.selection_mode,
            iteration_budget=eval(args.iteration_budget),
            added_name=args.added_name,
            version_num=args.version_num,
            num_epochs=args.num_epochs,
            early_stopping=args.early_stopping,
            general_path=args.general_path,
            rand_seed=rand_seed,
            hash_method=hash_method,
            prune_thr=prune_thr,
        ).run()
