# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Validate model on validation dataset """


import os
import shutil
import time

import cv2
import numpy as np
import tensorflow as tf
from add_corruption import add_weather, apply_corruption
from dataset_data import get_dataset_data, get_ocl_trc
from scipy.stats import iqr
from utils_box import CalibrateBoxUncert, calc_iou_np, calc_rmse
from utils_class import CalibrateClass, stable_softmax
from utils_extra import ValidUncertPlot, add_array_dict, gt_box_assigner, update_arrays


class Validate:
    """Class for the validation on a given dataset"""

    def __init__(
        self,
        model_params,
        img_names,
        driver,
        gt_classes,
        gt_boxes,
        model_dir,
        general_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ):
        """Constructs all the necessary attributes for the validation

        Args:
          model_params (dict): Model parameters
          img_names (list): Image names
          driver (object): Driver for serving single or batch images
          gt_classes (array): Ground truth classes
          gt_boxes (array): Ground truth bounding boxes
          model_dir (str): Path with model name, to import calibration models
          general_path (str): Path to working space

        """
        self.general_path = general_path
        self.saving_path = (
            general_path + "/results/validation/" + model_dir.split("/")[-1]
        )
        self.model_params = model_params
        self.img_names = img_names
        self.driver = driver
        self.gt_classes = gt_classes
        self.gt_boxes = gt_boxes
        self.model_dir = model_dir

        if os.path.exists(self.saving_path):
            shutil.rmtree(self.saving_path)
        os.makedirs(self.saving_path)

        runtime_txt = self.saving_path + "/validationstep_runtime.txt"
        self.runtime_txt = runtime_txt

        self.current_index = 0
        self.im_name = ""

        self.filtered_boxes, self.filtered_classes, self.filtered_scores = (
            np.array([]),
            np.array([]),
            np.array([]),
        )
        if self.model_params["enable_softmax"]:
            (
                self.filtered_classes_logits,
                self.filtered_classes_probab,
                self.filtered_entropy,
            ) = (np.array([]), np.array([]), np.array([]))
            if self.model_params["calibrate_classification"]:
                (
                    self.filtered_iso_all_probab,
                    self.filtered_ts_all_probab,
                    self.filtered_iso_all_entropy,
                    self.filtered_ts_all_entropy,
                ) = (np.array([]), np.array([]), np.array([]), np.array([]))
                (
                    self.filtered_iso_percls_probab,
                    self.filtered_ts_percls_probab,
                    self.filtered_iso_percls_entropy,
                    self.filtered_ts_percls_entropy,
                ) = (np.array([]), np.array([]), np.array([]), np.array([]))
            if (
                self.model_params["mc_classheadrate"]
                or self.model_params["mc_dropoutrate"]
            ):
                self.filtered_mcclass = np.array([])
            if self.model_params["calibrate_classification"]:
                (
                    self.filtered_iso_all_mcclass,
                    self.filtered_ts_all_mcclass,
                    self.filtered_iso_percls_mcclass,
                    self.filtered_ts_percls_mcclass,
                ) = (np.array([]), np.array([]), np.array([]), np.array([]))
        if self.model_params["mc_boxheadrate"] or self.model_params["mc_dropoutrate"]:
            self.filtered_mcbox = np.array([])
            if self.model_params["calibrate_regression"]:
                (
                    self.filtered_iso_all_mcbox,
                    self.filtered_ts_all_mcbox,
                    self.filtered_iso_percoo_mcbox,
                    self.filtered_ts_percoo_mcbox,
                    self.filtered_iso_perclscoo_mcbox,
                    self.filtered_iso_perclscoo_mcbox_rel,
                ) = (
                    np.array([]),
                    np.array([]),
                    np.array([]),
                    np.array([]),
                    np.array([]),
                    np.array([]),
                )
        if self.model_params["loss_attenuation"]:
            self.filtered_albox = np.array([])
            if self.model_params["calibrate_regression"]:
                (
                    self.filtered_iso_all_albox,
                    self.filtered_ts_all_albox,
                    self.filtered_iso_percoo_albox,
                    self.filtered_ts_percoo_albox,
                    self.filtered_iso_perclscoo_albox,
                    self.filtered_iso_perclscoo_albox_rel,
                ) = (
                    np.array([]),
                    np.array([]),
                    np.array([]),
                    np.array([]),
                    np.array([]),
                    np.array([]),
                )

        (
            self.filtered_gt_classes,
            self.filtered_gt_boxes,
            self.filtered_names,
            self.filtered_occlusions,
            self.filtered_truncations,
        ) = (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))

        self.occlusions, self.truncations = get_ocl_trc(model_dir, img_names)

    def _process_val_image(self, image, augment_method=False, aug_lvl=0):
        """Processes one validation image"""
        st = time.time()
        detections = self.driver.serve(image)
        elapsed_time = time.time() - st
        with open(self.runtime_txt, "a") as f:
            f.write(str(elapsed_time) + "\n")
        if self.model_params["enable_softmax"]:
            boxes, scores, classes, _, logits = tf.nest.map_structure(
                np.array, detections
            )
            probab_logits = [stable_softmax(logits[0])]
            entropy = -np.sum(
                probab_logits[0]
                * np.nan_to_num(np.log2(np.maximum(probab_logits[0], 10**-7))),
                axis=1,
            )
        else:
            boxes, scores, classes, _ = tf.nest.map_structure(np.array, detections)
            logits = [None]

        if (
            self.model_params["mc_boxheadrate"] or self.model_params["mc_dropoutrate"]
        ) and not self.model_params["loss_attenuation"]:
            mc_boxhead = np.nan_to_num(boxes[:, :, 4:])
            al_boxhead = None
        elif (
            self.model_params["mc_boxheadrate"] or self.model_params["mc_dropoutrate"]
        ) and self.model_params["loss_attenuation"]:
            al_boxhead = np.nan_to_num(boxes[:, :, 4:8])
            mc_boxhead = np.nan_to_num(boxes[:, :, 8:])
        elif (
            not (
                self.model_params["mc_boxheadrate"]
                or self.model_params["mc_dropoutrate"]
            )
            and self.model_params["loss_attenuation"]
        ):
            al_boxhead = np.nan_to_num(boxes[:, :, 4:])
            mc_boxhead = None
        else:
            al_boxhead = None
            mc_boxhead = None
        if self.model_params["mc_classheadrate"] or self.model_params["mc_dropoutrate"]:
            mc_classhead = np.nan_to_num(classes[:, :, 1:])
            classes = classes[:, :, 0]
        else:
            mc_classhead = None

        if mc_boxhead is not None or al_boxhead is not None:
            boxes = boxes[:, :, :4]

        if self.model_params["calibrate_regression"]:
            if mc_boxhead is not None:
                (
                    _,
                    iso_all_mcbox,
                    ts_all_mcbox,
                    ts_percoo_mcbox,
                    iso_percoo_mcbox,
                    iso_perclscoo_mcbox,
                    rel_iso_perclscoo_mcbox,
                ) = CalibrateBoxUncert(
                    self.model_params,
                    self.saving_path.split("/")[-1] + "/regression/mcdropout/",
                    general_path=self.general_path,
                ).calibrate_boxuncert(
                    mc_boxhead[0], classes[0], boxes[0]
                )
            if al_boxhead is not None:
                (
                    _,
                    iso_all_albox,
                    ts_all_albox,
                    ts_percoo_albox,
                    iso_percoo_albox,
                    iso_perclscoo_albox,
                    rel_iso_perclscoo_albox,
                ) = CalibrateBoxUncert(
                    self.model_params,
                    self.saving_path.split("/")[-1] + "/regression/aleatoric/",
                    general_path=self.general_path,
                ).calibrate_boxuncert(
                    al_boxhead[0], classes[0], boxes[0]
                )
        if self.model_params["calibrate_classification"]:
            if mc_classhead is not None:
                (
                    _,
                    _,
                    ts_all_probab,
                    ts_all_mcclass,
                    ts_all_entropy,
                    ts_perc_probab,
                    ts_perc_mcclass,
                    ts_perc_entropy,
                    iso_all_probab,
                    iso_all_mcclass,
                    iso_all_entropy,
                    iso_perc_probab,
                    iso_perc_mcclass,
                    iso_perc_entropy,
                ) = CalibrateClass(
                    logits[0],
                    self.saving_path.split("/")[-1],
                    self.model_params["calib_method_class"],
                    mc_classhead[0],
                    general_path=self.general_path,
                ).calibrate_class()
            else:
                (
                    _,
                    ts_all_probab,
                    ts_all_entropy,
                    ts_perc_probab,
                    ts_perc_entropy,
                    iso_all_probab,
                    iso_all_entropy,
                    iso_perc_probab,
                    iso_perc_entropy,
                ) = CalibrateClass(
                    logits[0],
                    self.saving_path.split("/")[-1],
                    self.model_params["calib_method_class"],
                    general_path=self.general_path,
                ).calibrate_class()

        # Augment boxes
        if augment_method and (aug_lvl == 1 or aug_lvl == 2):
            keep_boxes = np.where(
                np.all((self.gt_boxes[self.current_index] != [-1, -1, -1, -1]), axis=-1)
            )[0]
            filtered_gt_boxes = []
            for ind in range(len(self.gt_boxes[self.current_index])):
                if ind in keep_boxes:
                    box = self.gt_boxes[self.current_index][ind]
                    # Flip the box
                    if aug_lvl == 1:
                        filtered_gt_boxes.append(
                            [
                                image.shape[1] - box[2],
                                box[1],
                                image.shape[1] - box[0],
                                box[3],
                            ]
                        )
                    elif aug_lvl == 2:
                        filtered_gt_boxes.append(
                            [
                                box[0],
                                image.shape[2] - box[3],
                                box[2],
                                image.shape[2] - box[1],
                            ]
                        )
                else:
                    filtered_gt_boxes.append(self.gt_boxes[self.current_index][ind])
            filtered_gt_boxes = np.asarray(filtered_gt_boxes)
        else:
            filtered_gt_boxes = self.gt_boxes[self.current_index]

        # Collect detections
        for i in np.where(self.gt_classes[self.current_index] > 0)[0]:
            if augment_method:
                self.filtered_names = np.append(
                    self.filtered_names,
                    self.im_name[:-4] + "_" + augment_method + self.im_name[-4:],
                )
            else:
                self.filtered_names = np.append(self.filtered_names, self.im_name)

            self.filtered_gt_classes = np.append(
                self.filtered_gt_classes, self.gt_classes[self.current_index][i]
            )
            self.filtered_gt_boxes = update_arrays(
                self.filtered_gt_boxes, filtered_gt_boxes, i
            )

            self.filtered_occlusions = np.append(
                self.filtered_occlusions, self.occlusions[self.current_index][i]
            )
            self.filtered_truncations = np.append(
                self.filtered_truncations, self.truncations[self.current_index][i]
            )
            # Reorder based on lowest MSE/highest IoU
            correct_index = gt_box_assigner(
                self.model_params["assign_gt_box"], filtered_gt_boxes, boxes[0], i
            )

            self.filtered_classes = np.append(
                self.filtered_classes, classes[0][correct_index]
            )
            if self.model_params["enable_softmax"]:
                self.filtered_classes_logits = update_arrays(
                    self.filtered_classes_logits, logits[0], correct_index
                )
                self.filtered_classes_probab = update_arrays(
                    self.filtered_classes_probab, probab_logits[0], correct_index
                )
                self.filtered_entropy = np.append(
                    self.filtered_entropy, entropy[correct_index]
                )
                if self.model_params["calibrate_classification"]:
                    self.filtered_iso_all_probab = update_arrays(
                        self.filtered_iso_all_probab, iso_all_probab, correct_index
                    )
                    self.filtered_ts_all_probab = update_arrays(
                        self.filtered_ts_all_probab, ts_all_probab, correct_index
                    )
                    self.filtered_iso_percls_probab = update_arrays(
                        self.filtered_iso_percls_probab, iso_perc_probab, correct_index
                    )
                    self.filtered_ts_percls_probab = update_arrays(
                        self.filtered_ts_percls_probab, ts_perc_probab, correct_index
                    )
                    if iso_all_entropy.size > 0:
                        self.filtered_iso_all_entropy = np.append(
                            self.filtered_iso_all_entropy,
                            iso_all_entropy[correct_index],
                        )
                    if ts_all_entropy.size > 0:
                        self.filtered_ts_all_entropy = np.append(
                            self.filtered_ts_all_entropy, ts_all_entropy[correct_index]
                        )
                    if iso_perc_entropy.size > 0:
                        self.filtered_iso_percls_entropy = np.append(
                            self.filtered_iso_percls_entropy,
                            iso_perc_entropy[correct_index],
                        )
                    if ts_perc_entropy.size > 0:
                        self.filtered_ts_percls_entropy = np.append(
                            self.filtered_ts_percls_entropy,
                            ts_perc_entropy[correct_index],
                        )

            if mc_classhead is not None:
                self.filtered_mcclass = update_arrays(
                    self.filtered_mcclass, mc_classhead[0], correct_index
                )
                if self.model_params["calibrate_classification"]:
                    self.filtered_iso_all_mcclass = update_arrays(
                        self.filtered_iso_all_mcclass, iso_all_mcclass, correct_index
                    )
                    self.filtered_ts_all_mcclass = update_arrays(
                        self.filtered_ts_all_mcclass, ts_all_mcclass, correct_index
                    )
                    self.filtered_iso_percls_mcclass = update_arrays(
                        self.filtered_iso_percls_mcclass,
                        iso_perc_mcclass,
                        correct_index,
                    )
                    self.filtered_ts_percls_mcclass = update_arrays(
                        self.filtered_ts_percls_mcclass, ts_perc_mcclass, correct_index
                    )
            if al_boxhead is not None:
                self.filtered_albox = update_arrays(
                    self.filtered_albox, al_boxhead[0], correct_index
                )
                if self.model_params["calibrate_regression"]:
                    self.filtered_iso_all_albox = update_arrays(
                        self.filtered_iso_all_albox, iso_all_albox, correct_index
                    )
                    self.filtered_ts_all_albox = update_arrays(
                        self.filtered_ts_all_albox, ts_all_albox, correct_index
                    )
                    self.filtered_ts_percoo_albox = update_arrays(
                        self.filtered_ts_percoo_albox, ts_percoo_albox, correct_index
                    )
                    self.filtered_iso_percoo_albox = update_arrays(
                        self.filtered_iso_percoo_albox, iso_percoo_albox, correct_index
                    )
                    self.filtered_iso_perclscoo_albox = update_arrays(
                        self.filtered_iso_perclscoo_albox,
                        iso_perclscoo_albox,
                        correct_index,
                    )
                    self.filtered_iso_perclscoo_albox_rel = update_arrays(
                        self.filtered_iso_perclscoo_albox_rel,
                        rel_iso_perclscoo_albox,
                        correct_index,
                    )

            if mc_boxhead is not None:
                self.filtered_mcbox = update_arrays(
                    self.filtered_mcbox, mc_boxhead[0], correct_index
                )
                if self.model_params["calibrate_regression"]:
                    self.filtered_iso_all_mcbox = update_arrays(
                        self.filtered_iso_all_mcbox, iso_all_mcbox, correct_index
                    )
                    self.filtered_ts_all_mcbox = update_arrays(
                        self.filtered_ts_all_mcbox, ts_all_mcbox, correct_index
                    )
                    self.filtered_ts_percoo_mcbox = update_arrays(
                        self.filtered_ts_percoo_mcbox, ts_percoo_mcbox, correct_index
                    )
                    self.filtered_iso_percoo_mcbox = update_arrays(
                        self.filtered_iso_percoo_mcbox, iso_percoo_mcbox, correct_index
                    )
                    self.filtered_iso_perclscoo_mcbox = update_arrays(
                        self.filtered_iso_perclscoo_mcbox,
                        iso_perclscoo_mcbox,
                        correct_index,
                    )
                    self.filtered_iso_perclscoo_mcbox_rel = update_arrays(
                        self.filtered_iso_perclscoo_mcbox_rel,
                        rel_iso_perclscoo_mcbox,
                        correct_index,
                    )

            self.filtered_boxes = update_arrays(
                self.filtered_boxes, boxes[0][:, :4], correct_index
            )
            self.filtered_scores = np.append(
                self.filtered_scores, scores[0][correct_index]
            )

        if not augment_method:
            self.current_index += 1

    def launch_val(self):
        """Main function to perform validation validation images"""

        self.current_index = 0
        for im_name in self.img_names:
            self.im_name = im_name
            image_npath = get_dataset_data(self.model_dir, im_name=im_name)[-1]
            image_file = tf.io.read_file(
                self.model_dir.split("models")[0] + "/datasets/" + image_npath
            )
            im = tf.io.decode_image(image_file, channels=3, expand_animations=False)
            im = tf.expand_dims(im, axis=0)
            # Augmentation
            if self.model_params["infer_augment"]:
                modes = self.model_params["infer_augment"]
                if "heq" in modes:
                    img_yuv = cv2.cvtColor(np.asarray(im[0]), cv2.COLOR_BGR2YUV)
                    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                    aug_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                    # cv2.imwrite("aug_image_heq.png", aug_image)
                    self._process_val_image(
                        tf.expand_dims(aug_image, axis=0), "histeq", 0
                    )
                if "alb" in modes:
                    weather_types = ["snow", "fog", "rain", "noise"]
                    for weather_type in weather_types:
                        aug_image = add_weather(im[0], weather_type)
                        self._process_val_image(
                            tf.expand_dims(aug_image, axis=0), weather_type, 0
                        )
                if "aug" in modes:
                    corruptions = ["ns", "mb", "ct", "br"]
                    for corruption_type in corruptions:
                        augims = apply_corruption(corruption_type, np.asarray(im[0]))
                        for ind_im, aug_image in enumerate(augims):
                            self._process_val_image(
                                tf.expand_dims(aug_image, axis=0),
                                f"{corruption_type}{ind_im}",
                                ind_im,
                            )
                if "flip" in modes:
                    flip_types = [0, 1]  # Vertical and horizontal flips
                    for flip_type in flip_types:
                        aug_image = cv2.flip(np.asarray(im[0]), flip_type)
                        flip_name = "vflip" if flip_type == 0 else "hflip"
                        self._process_val_image(
                            tf.expand_dims(aug_image, axis=0),
                            flip_name,
                            1 if flip_type == 0 else 2,
                        )
            self._process_val_image(im)  # It must be post augmentation

        # Save prediction data on validation set
        detections_data = []
        with open(self.saving_path + "/validate_results.txt", "w") as f:
            for i in range(len(self.filtered_boxes)):
                data_dict = {
                    "image_name": self.filtered_names[i],
                    "score": self.filtered_scores[i],
                    "bbox": list(self.filtered_boxes[i]),
                    "gt_bbox": list(self.filtered_gt_boxes[i]),
                    "gt_occl": self.filtered_occlusions[i],
                    "gt_trunc": self.filtered_truncations[i],
                    "class": self.filtered_classes[i],
                    "gt_class": self.filtered_gt_classes[i],
                }

                if self.model_params["enable_softmax"]:
                    data_dict["logits"] = list(self.filtered_classes_logits[i])
                    data_dict["probab"] = list(self.filtered_classes_probab[i])
                    data_dict["entropy"] = self.filtered_entropy[i]
                    if self.model_params["calibrate_classification"]:
                        data_dict = add_array_dict(
                            data_dict, self.filtered_iso_all_probab, "iso_all_probab", i
                        )
                        data_dict = add_array_dict(
                            data_dict, self.filtered_ts_all_probab, "ts_all_probab", i
                        )
                        data_dict = add_array_dict(
                            data_dict,
                            self.filtered_ts_percls_probab,
                            "ts_percls_probab",
                            i,
                        )
                        data_dict = add_array_dict(
                            data_dict,
                            self.filtered_iso_percls_probab,
                            "iso_percls_probab",
                            i,
                        )
                        data_dict = add_array_dict(
                            data_dict,
                            self.filtered_iso_all_entropy,
                            "iso_all_entropy",
                            i,
                        )
                        data_dict = add_array_dict(
                            data_dict, self.filtered_ts_all_entropy, "ts_all_entropy", i
                        )
                        data_dict = add_array_dict(
                            data_dict,
                            self.filtered_ts_percls_entropy,
                            "ts_percls_entropy",
                            i,
                        )
                        data_dict = add_array_dict(
                            data_dict,
                            self.filtered_iso_percls_entropy,
                            "iso_percls_entropy",
                            i,
                        )
                if (
                    self.model_params["mc_classheadrate"]
                    or self.model_params["mc_dropoutrate"]
                ):
                    data_dict["uncalib_mcclass"] = list(self.filtered_mcclass[i])
                    if self.model_params["calibrate_classification"]:
                        data_dict = add_array_dict(
                            data_dict,
                            self.filtered_iso_all_mcclass,
                            "iso_all_mcclass",
                            i,
                        )
                        data_dict = add_array_dict(
                            data_dict, self.filtered_ts_all_mcclass, "ts_all_mcclass", i
                        )
                        data_dict = add_array_dict(
                            data_dict,
                            self.filtered_ts_percls_mcclass,
                            "ts_percls_mcclass",
                            i,
                        )
                        data_dict = add_array_dict(
                            data_dict,
                            self.filtered_iso_percls_mcclass,
                            "iso_percls_mcclass",
                            i,
                        )
                if (
                    self.model_params["mc_boxheadrate"]
                    or self.model_params["mc_dropoutrate"]
                ):
                    data_dict["uncalib_mcbox"] = list(self.filtered_mcbox[i])
                    if self.model_params["calibrate_regression"]:
                        data_dict = add_array_dict(
                            data_dict, self.filtered_iso_all_mcbox, "iso_all_mcbox", i
                        )
                        data_dict = add_array_dict(
                            data_dict, self.filtered_ts_all_mcbox, "ts_all_mcbox", i
                        )
                        data_dict = add_array_dict(
                            data_dict,
                            self.filtered_ts_percoo_mcbox,
                            "ts_percoo_mcbox",
                            i,
                        )
                        data_dict = add_array_dict(
                            data_dict,
                            self.filtered_iso_percoo_mcbox,
                            "iso_percoo_mcbox",
                            i,
                        )
                        data_dict = add_array_dict(
                            data_dict,
                            self.filtered_iso_perclscoo_mcbox,
                            "iso_perclscoo_mcbox",
                            i,
                        )
                        data_dict = add_array_dict(
                            data_dict,
                            self.filtered_iso_perclscoo_mcbox_rel,
                            "rel_iso_perclscoo_mcbox",
                            i,
                        )
                if self.model_params["loss_attenuation"]:
                    data_dict["uncalib_albox"] = list(self.filtered_albox[i])
                    if self.model_params["calibrate_regression"]:
                        data_dict = add_array_dict(
                            data_dict, self.filtered_iso_all_albox, "iso_all_albox", i
                        )
                        data_dict = add_array_dict(
                            data_dict, self.filtered_ts_all_albox, "ts_all_albox", i
                        )
                        data_dict = add_array_dict(
                            data_dict,
                            self.filtered_ts_percoo_albox,
                            "ts_percoo_albox",
                            i,
                        )
                        data_dict = add_array_dict(
                            data_dict,
                            self.filtered_iso_percoo_albox,
                            "iso_percoo_albox",
                            i,
                        )
                        data_dict = add_array_dict(
                            data_dict,
                            self.filtered_iso_perclscoo_albox,
                            "iso_perclscoo_albox",
                            i,
                        )
                        data_dict = add_array_dict(
                            data_dict,
                            self.filtered_iso_perclscoo_albox_rel,
                            "rel_iso_perclscoo_albox",
                            i,
                        )

                detections_data.append(data_dict)
                f.write(str(data_dict) + "\n")

        with open(self.saving_path + "/average_score.txt", "w") as f:
            f.write(str(np.mean(self.filtered_scores)))
        with open(self.runtime_txt) as f:
            times = f.readlines()
            times = np.asarray(
                [times[i].split("\n")[0] for i in range(len(times))]
            ).astype(np.float32)
            times = times[times < 1]
        # Calculate upper bounds for outliers
        q3 = np.percentile(times, 75)
        iqr_value = iqr(times)
        upper_bound = q3 + 50 * iqr_value
        filtered_times = [x for x in times if x <= upper_bound]

        with open(self.runtime_txt, "w") as file:
            file.write(
                "Mean time in ms: {:.3f}\n".format(np.mean(filtered_times) * 1000)
            )
            file.write("STD time in ms: {:.3f}\n".format(np.std(filtered_times) * 1000))
            file.write(
                "Median time in ms: {:.3f}\n".format(np.median(filtered_times) * 1000)
            )

        if (
            not (
                self.model_params["mc_boxheadrate"]
                or self.model_params["mc_dropoutrate"]
            )
            and not self.model_params["loss_attenuation"]
        ):
            with open(self.saving_path + "/model_performance.txt", "w") as file:
                file.write(
                    "Misclassification rate: {}\n".format(
                        len(
                            np.where(self.filtered_gt_classes != self.filtered_classes)[
                                0
                            ]
                        )
                        / len(self.filtered_gt_classes)
                    )
                )
                file.write(
                    "mIoU: {}\n".format(
                        np.mean(
                            calc_iou_np(self.filtered_gt_boxes, self.filtered_boxes)
                        )
                    )
                )
                file.write(
                    "RMSE: {}\n".format(
                        float(calc_rmse(self.filtered_gt_boxes, self.filtered_boxes))
                    )
                )
        else:
            if self.model_params["loss_attenuation"]:
                if not os.path.exists(self.saving_path + "/aleatoric"):
                    os.makedirs(self.saving_path + "/aleatoric")
                if self.model_params["calibrate_regression"]:
                    calibs = {
                        "ts_all": self.filtered_ts_all_albox,
                        "ts_percoo": self.filtered_ts_percoo_albox,
                        "iso_all": self.filtered_iso_all_albox,
                        "iso_percoo": self.filtered_iso_percoo_albox,
                        "iso_perclscoo": self.filtered_iso_perclscoo_albox,
                        "rel_iso_perclscoo": self.filtered_iso_perclscoo_albox_rel,
                    }
                    calibrated_uncert = calibs[self.model_params["calib_method_box"]]
                    if calibrated_uncert.size == 0:
                        calibrated_uncert = None
                else:
                    calibrated_uncert = None
                ValidUncertPlot(
                    self.filtered_gt_boxes,
                    self.filtered_boxes,
                    self.filtered_gt_classes,
                    self.filtered_classes,
                    self.filtered_albox,
                    calibrated_uncert,
                    self.saving_path + "/aleatoric",
                    self.model_params,
                )

            if (
                self.model_params["mc_boxheadrate"]
                or self.model_params["mc_dropoutrate"]
            ):
                if not os.path.exists(self.saving_path + "/mcdropout"):
                    os.makedirs(self.saving_path + "/mcdropout")
                if self.model_params["calibrate_regression"]:
                    calibs = {
                        "ts_all": self.filtered_ts_all_mcbox,
                        "ts_percoo": self.filtered_ts_percoo_mcbox,
                        "iso_all": self.filtered_iso_all_mcbox,
                        "iso_percoo": self.filtered_iso_percoo_mcbox,
                        "iso_perclscoo": self.filtered_iso_perclscoo_mcbox,
                        "rel_iso_perclscoo": self.filtered_iso_perclscoo_mcbox_rel,
                    }
                    calibrated_uncert = calibs[self.model_params["calib_method_box"]]
                    if calibrated_uncert.size == 0:
                        calibrated_uncert = None
                else:
                    calibrated_uncert = None
                ValidUncertPlot(
                    self.filtered_gt_boxes,
                    self.filtered_boxes,
                    self.filtered_gt_classes,
                    self.filtered_classes,
                    self.filtered_mcbox,
                    calibrated_uncert,
                    self.saving_path + "/mcdropout",
                    self.model_params,
                )
        print("Validation is finished on all the validation dataset")
