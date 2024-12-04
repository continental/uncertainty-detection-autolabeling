# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================

import copy
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import yaml
from PIL import Image

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ssl_utils.parent import Parent_SSL, generate_commands_and_create_dirs
from utils_box import calc_iou_np


class AdvancedLabels(Parent_SSL):
    def __init__(
        self,
        glc_method,
        synthetic=False,
        added_name_glc="",
        iou_consist=0.90,
        md_max_inter=0,
        md_dropped_gt=0.20,
        mistakes_per_image=1,
        mistake_upper_size=100,
        mistake_lower_size=10,
        correct_boxes_to_modify=0.20,
        correct_boxes_width_height=0.10,
        correct_max_inter=1.0,
        correct_score=0.40,
        labeled_portion=0,
        num_labeled=0,
        v_number=0,
        *args,
        **kwargs,
    ):
        """
        Initializes an instance of the AdvancedLabels class. Requires predictions with consistency_ssl activated in src/hparams_config.py
        Args:
            glc_method (str): The method to use for fixing GT labels. ["mistakes", "md", "noisy"]
            synthetic (bool, optional): Whether to use synthetic augmentations to monitor effect on GT. Defaults to False.
            added_name_glc (str, optional): Additional name for saving paths. Defaults to "".
            iou_consist (float, optional): IoU consistency threshold between predictions and their augmented variants. Defaults to 0.90.
            md_max_inter (int, optional): Maximum intersection threshold between prediction and GT. Defaults to 0.
            md_dropped_gt (float, optional): Percentage of ground truth boxes to drop for synthetic MD. Defaults to 0.20.
            mistakes_per_image (int, optional): Number of mistakes to introduce per image. Defaults to 1.
            mistake_upper_size (int, optional): Upper size limit for introduced mistakes. Defaults to 100.
            mistake_lower_size (int, optional): Lower size limit for introduced mistakes. Defaults to 10.
            correct_boxes_to_modify (float, optional): Percentage of correct boxes to modify for synthetic noise. Defaults to 0.20.
            correct_boxes_width_height (float, optional): Percentage of width and height to modify for synthetic noise. Defaults to 0.10.
            correct_max_inter (float, optional): Maximum intersection between prediction and GT. Defaults to 1.0.
            correct_score (float, optional): Score threshold for selecting boxes as correct. Defaults to 0.40.
            labeled_portion (int, optional): Portion of labeled data to use for pseudo labels. Defaults to 0.
            num_labeled (int, optional): Number of labeled data. Defaults to 0.
            v_number (int, optional): Model version number. Defaults to 0.
            *args: Variable length argument list. Belongs to parent class.
            **kwargs: Arbitrary keyword arguments. Belongs to parent class.
        """

        super().__init__(*args, **kwargs)  # Call parent constructor
        self.added_name_glc = added_name_glc
        self.v_number = v_number
        self.consist_intersection = iou_consist
        self.md_max_inter = md_max_inter
        self.md_dropped_gt = md_dropped_gt
        self.mistakes_per_image = mistakes_per_image
        self.mistakes_upper_size = mistake_upper_size
        self.mistakes_lower_size = mistake_lower_size
        self.correct_boxes_to_modify = correct_boxes_to_modify
        self.correct_boxes_width_height = correct_boxes_width_height
        self.correct_max_inter = correct_max_inter
        self.correct_score = correct_score
        self.num_labeled = num_labeled
        self.labeled_portion = labeled_portion
        inf_res = (
            self.inference_path.split("EXP")[0]
            + "consist_EXP"
            + self.inference_path.split("EXP")[-1]
        )
        if not os.path.exists(inf_res):
            labeled_inf_path = (
                self.dataset_path
                + "/tf_stac/"
                + "/".join(self.labeled_indices_path.split("/")[:-1])
                + "/consist_train_init_V"
                + str(v_number)
                + ".txt"
            )
            new_indices = self.labeled_indices
            if self.dataset == "BDD100K":
                import glob

                imnames = sorted(glob.glob(self.gt_images_folder + "/*"))
                imnames = np.asarray([i.split("/")[-1] for i in imnames])
                imnames_index_map = {name: idx for idx, name in enumerate(imnames)}
                map_indices = [
                    imnames_index_map[a]
                    for a in self.labeled_imnames_all
                    if a in imnames_index_map
                ]
                new_indices = np.asarray(map_indices)[
                    self.labeled_indices
                ]  # Bdd images without labels, inspector doesn't check them

            with open(labeled_inf_path, "w") as file:
                # Convert each integer to a string and join them with a newline character
                file.write("\n".join(map(str, new_indices)))

            yaml_name = self.dataset + "_consist_V0"
            yaml_path = (
                self.general_path
                + "/configs/inference/SSL/inference_"
                + yaml_name
                + ".yaml"
            )
            with open(yaml_path, "r") as yaml_file:
                data = yaml.safe_load(yaml_file)

            data["infer_indices"] = labeled_inf_path
            data["model_dir"] = (
                self.general_path + "/models/trained_models/STAC/" + self.model_name[4:]
            )  # without EXP_
            data["saved_model_dir"] = (
                self.general_path + "/models/exported_models/SSL/" + self.model_name
            )
            data["added_name"] = "consist"
            # Write the modified data back to the YAML file
            if not os.path.exists(self.general_path + "/configs/inference/SSL/"):
                os.makedirs(self.general_path + "/configs/inference/SSL/")
            with open(
                yaml_path,
                "w",
            ) as yaml_file:
                yaml.dump(data, yaml_file, default_flow_style=False)

            command = f"PYTHONPATH=/{self.general_path}/src/ python -m inspector --dataset {yaml_name} --mode 6 --general_path {self.general_path}/"
            subprocess.run(command, shell=True)

        (
            pred_im_names,
            self.score_perim,
            self.pred_cls,
            self.pred_box,
            self.ciou_perim,
            ccls_perim,
        ) = self.read_predictions(inf_res, "score", True)
        all_gt_objects = []
        if self.dataset == "KITTI":
            self.clean_perd_im_names = np.asarray(
                [i.split(".")[0] + ".txt" for i in pred_im_names]
            )
        else:
            self.clean_perd_im_names = np.asarray(
                [i.split(".")[0] + ".jpg" for i in pred_im_names]
            )

        for image_data in self.clean_perd_im_names:
            all_gt_objects.append(
                self.read_annotations()(
                    os.path.join(self.gt_labels_folder, image_data),
                    self.used_classes,
                )
            )
        if self.dataset == "KITTI":
            self.gt_box = [
                [[b["bbox"][1], b["bbox"][0], b["bbox"][3], b["bbox"][2]] for b in im]
                for im in all_gt_objects
            ]
        else:
            self.gt_box = [[b["bbox"] for b in im] for im in all_gt_objects]
        self.gt_cls = [[b["class"] for b in im] for im in all_gt_objects]

        self.ious = [
            [calc_iou_np([gt], self.pred_box[i]) for gt in self.gt_box[i]]
            for i in range(len(self.gt_box))
        ]
        self.matched_det = [np.argmax(iou, axis=-1) for iou in self.ious]
        self.ious_gt = [np.max(iou, axis=-1) for iou in self.ious]

        if "md" in glc_method:
            self.mds(synthetic=synthetic)
        if "mistakes" in glc_method:
            self.mistakes(synthetic=synthetic)
        if "noisy" in glc_method:
            self.noisy_boxes(synthetic=synthetic)

    def corrected_gt(
        self,
        folder_name="/gt/",
        remove_mistakes=False,
        wrong_gt=None,
        correct_boxes=False,
        corrected_gt_boxes=None,
        add_missingboxes=False,
        missing_gt_boxes=None,
        drop_gt=None,
        add_mistakes=None,
        remove_added_mistakes=None,
    ):
        """
        Generates corrected ground truth labels.
        Args:
            folder_name (str, optional): Folder name for the corrected ground truth labels. Defaults to "/gt/".
            remove_mistakes (bool, optional): Whether to remove mistakes from the ground truth labels. Defaults to False.
            wrong_gt (list, optional): List of indices of wrong ground truth labels to remove. Defaults to None.
            correct_boxes (bool, optional): Whether to correct the bounding boxes in the ground truth labels. Defaults to False.
            corrected_gt_boxes (list, optional): List of corrected bounding boxes for the ground truth labels. Defaults to None.
            add_missingboxes (bool, optional): Whether to add missing boxes to the ground truth labels. Defaults to False.
            missing_gt_boxes (list, optional): List of missing ground truth boxes to add. Defaults to None.
            drop_gt (list, optional): List of indices of ground truth labels to drop. Defaults to None.
            add_mistakes (list, optional): List of mistakes to add to the ground truth labels. Defaults to None.
            remove_added_mistakes (list, optional): List of indices of added mistakes to remove. Defaults to None.
        """
        corrected_gt_path = (
            self.dataset_path + "/pseudo_labels/" + self.added_name + folder_name
        )
        if os.path.exists(corrected_gt_path):
            shutil.rmtree(corrected_gt_path)
        os.makedirs(corrected_gt_path)
        sys.path.append(os.path.abspath(self.general_path))
        val_indices = []
        if self.dataset == "KITTI":
            for i in range(len(self.clean_perd_im_names)):
                gt_file_path = self.gt_labels_folder + "/" + self.clean_perd_im_names[i]
                filtered_pred_lines = []
                with open(gt_file_path, "r") as file:
                    lines = file.readlines()
                    j = 0
                    for line in lines:
                        parts = line.strip().split(" ")
                        if len(parts) > 0 and parts[0] in self.used_classes:
                            should_append = True
                            if (remove_mistakes and j in wrong_gt[i]) or (
                                drop_gt is not None and (i, j) in drop_gt
                            ):
                                should_append = False
                            if correct_boxes and should_append:
                                parts[4] = str(corrected_gt_boxes[i][j][1])
                                parts[5] = str(corrected_gt_boxes[i][j][0])
                                parts[6] = str(corrected_gt_boxes[i][j][3])
                                parts[7] = str(corrected_gt_boxes[i][j][2])
                            if should_append:
                                filtered_pred_lines.append(parts)
                            j += 1

                if add_missingboxes:
                    for b in range(len(self.pred_box[i])):
                        if missing_gt_boxes[i][b]:
                            parts = [
                                self.used_classes[int(self.pred_cls[i][b]) - 1],
                                "-1",
                                "-1",
                                "-10",
                                str(self.pred_box[i][b][1]),
                                str(self.pred_box[i][b][0]),
                                str(self.pred_box[i][b][3]),
                                str(self.pred_box[i][b][2]),
                                "-1",
                                "-1",
                                "-1",
                                "-1000",
                                "-1000",
                                "-1000",
                                "-10",
                            ]
                            filtered_pred_lines.append(parts)
                if add_mistakes is not None:
                    for k, box in enumerate(add_mistakes[i]):
                        if (
                            remove_added_mistakes is None
                            or k not in remove_added_mistakes[i]
                        ):
                            cls = np.random.choice(self.used_classes)
                            parts = [
                                cls,
                                "-1",
                                "-1",
                                "-10",
                                str(box[0]),
                                str(box[1]),
                                str(box[2]),
                                str(box[3]),
                                "-1",
                                "-1",
                                "-1",
                                "-1000",
                                "-1000",
                                "-1000",
                                "-10",
                            ]
                            filtered_pred_lines.append(parts)

                file_path = corrected_gt_path + "/" + self.clean_perd_im_names[i]
                with open(file_path, "w") as file:
                    file.writelines(
                        [" ".join(item) + "\n" for item in filtered_pred_lines]
                    )

            with open(
                self.general_path + "/datasets/KITTI/vaL_index_list.txt", "r"
            ) as file:
                for line in file:
                    val_indices.append(int(line.strip()))

            from datasets.KITTI.kitti_tf_creator import kitti_custom_to_tfrecords

            kitti_custom_to_tfrecords(
                corrected_gt_path,
                corrected_gt_path,
                self.general_path + "/datasets/KITTI/kitti.pbtxt",
                [s.lower() for s in self.used_classes],
                get_orig=False,
                data_dir_orig=self.general_path + "/datasets/KITTI/training/",
                validation_indices=val_indices,
                train_indices=self.labeled_indices,
            )
        else:
            filtered_data = []
            i = 0
            for item in self.bdd_data:
                if (
                    "labels" in item
                    and isinstance(item["labels"], list)
                    and item["name"] in self.clean_perd_im_names
                ):
                    filtered_detections = []
                    filtered_item = item.copy()
                    j = 0
                    for obj in filtered_item["labels"]:
                        if obj["category"] in self.used_classes:
                            should_append = True
                            if (remove_mistakes and j in wrong_gt[i]) or (
                                drop_gt is not None and (i, j) in drop_gt
                            ):
                                should_append = False
                            if correct_boxes and should_append:
                                obj["box2d"]["x1"] = corrected_gt_boxes[i][j][1]
                                obj["box2d"]["y1"] = corrected_gt_boxes[i][j][0]
                                obj["box2d"]["x2"] = corrected_gt_boxes[i][j][3]
                                obj["box2d"]["y2"] = corrected_gt_boxes[i][j][2]
                            if should_append:
                                filtered_detections.append(obj)
                            j += 1

                    if add_missingboxes:
                        for b in range(len(self.pred_box[i])):
                            if missing_gt_boxes[i][b]:
                                obj = {
                                    "id": str(j + b),
                                    "attributes": {
                                        "occluded": False,
                                        "truncated": False,
                                        "trafficLightColor": "G",
                                    },
                                    "category": self.used_classes[
                                        int(self.pred_cls[i][b] - 1)
                                    ],
                                    "box2d": {
                                        "x1": self.pred_box[i][b][1],
                                        "y1": self.pred_box[i][b][0],
                                        "x2": self.pred_box[i][b][3],
                                        "y2": self.pred_box[i][b][2],
                                    },
                                }
                                filtered_detections.append(obj)

                    if add_mistakes is not None:
                        for k, box in enumerate(add_mistakes[i]):
                            if (
                                remove_added_mistakes is None
                                or k not in remove_added_mistakes[i]
                            ):
                                cls = np.random.choice(self.used_classes)
                                obj = {
                                    "id": str(j + k),
                                    "attributes": {
                                        "occluded": False,
                                        "truncated": False,
                                        "trafficLightColor": "G",
                                    },
                                    "category": cls,
                                    "box2d": {
                                        "x1": box[0],
                                        "y1": box[1],
                                        "x2": box[2],
                                        "y2": box[3],
                                    },
                                }
                                filtered_detections.append(obj)

                    filtered_item["labels"] = filtered_detections
                    filtered_data.append(filtered_item)
                    i += 1
            with open(
                corrected_gt_path + "bdd100k_labels_images_train.json", "w"
            ) as json_file:
                json.dump(filtered_data, json_file, indent=4)

            from datasets.BDD100K.bdd_tf_creator import bdd_custom_to_tfrecords

            bdd_custom_to_tfrecords(
                corrected_gt_path,
                corrected_gt_path,
                self.general_path + "/datasets/BDD100K/bdd.pbtxt",
                [s.lower() for s in self.used_classes],
                get_orig=False,
                train_indices=self.labeled_indices,
                data_dir_orig=self.general_path + "/datasets/BDD100K/bdd100k/",
            )
        generate_commands_and_create_dirs(
            folder_name.split("/")[1],
            self.dataset,
            labeled_portion=self.labeled_portion,
            num_labeled=self.num_labeled,
        )

    def mds(self, synthetic=False, generate=True):
        """Finds MDs in GT via prediction consistency.
        Args:
            synthetic (bool, optional): Whether to use synthetic approach. Defaults to False.
            generate (bool, optional): Whether to generate new GT labels. Defaults to True.
        """
        extra_correct_dets = [
            np.equal(np.max(giou, axis=0), self.md_max_inter)
            * np.greater_equal(ciou, self.consist_intersection)
            for ciou, giou in zip(self.ciou_perim, self.ious)
        ]

        if synthetic:
            # Drop gt manually
            total_gt = len(np.concatenate(self.gt_cls))
            dropped_gt = int(self.md_dropped_gt * total_gt)

            valid_drop_indices = []
            for i, gts in enumerate(self.gt_box):
                if len(gts) > 1:
                    valid_drop_indices.extend([(i, j) for j in range(len(gts) - 1)])

            # Ensure that we do not drop more than we have valid indices for
            dropped_gt = min(dropped_gt, len(valid_drop_indices))

            # Randomly select indices to drop
            np.random.shuffle(valid_drop_indices)
            selected_drops = valid_drop_indices[:dropped_gt]

            # Create new gt_box and pred_box after dropping detections
            new_gt_box = []
            for i, gts in enumerate(self.gt_box):
                new_gts = [
                    gts[j] for j in range(len(gts)) if (i, j) not in selected_drops
                ]
                new_gt_box.append(new_gts)

            # Recalculate IoUs, matched_det, and ious_gt
            post_drop_ious = [
                [calc_iou_np([gt], self.pred_box[i]) for gt in new_gt_box[i]]
                for i in range(len(new_gt_box))
            ]
            post_drop_extra_correct_dets = [
                np.less_equal(np.max(giou, axis=0), self.md_max_inter)
                * np.greater_equal(ciou, self.consist_intersection)
                for ciou, giou in zip(self.ciou_perim, post_drop_ious)
            ]
            dropped_gt_box = []
            for i, gts in enumerate(self.gt_box):
                dropped_gts = [
                    gts[j] for j in range(len(gts)) if (i, j) in selected_drops
                ]
                dropped_gt_box.append(dropped_gts)

            added_pred_box = [
                [
                    self.pred_box[j][i]
                    for i in range(len(self.pred_box[j]))
                    if post_drop_extra_correct_dets[j][i]
                ]
                for j in range(len(self.pred_box))
            ]

            drop_ious = [
                [
                    calc_iou_np([gt], added_pred_box[i])
                    for gt in dropped_gt_box[i]
                    if len(added_pred_box[i]) > 0
                ]
                for i in range(len(dropped_gt_box))
            ]

            post_drop_ious_gt = [
                np.max(iou, axis=-1) for iou in drop_ious if len(iou) > 0
            ]
            all_diou = np.concatenate(post_drop_ious_gt)

            print(
                np.sum(np.concatenate(extra_correct_dets)),
                np.sum(np.concatenate(post_drop_extra_correct_dets)),
                np.mean(all_diou[all_diou > 0]),
            )
            if generate:
                self.corrected_gt(
                    folder_name="/"
                    + self.added_name_glc
                    + "_synthetic_addedmissing_gt_V"
                    + str(self.v_number)
                    + "/",
                    add_missingboxes=True,
                    missing_gt_boxes=post_drop_extra_correct_dets,
                    drop_gt=selected_drops,
                )

                self.corrected_gt(
                    folder_name="/"
                    + self.added_name_glc
                    + "_synthetic_drop_gt_V"
                    + str(self.v_number)
                    + "/",
                    drop_gt=selected_drops,
                )
            else:
                return post_drop_extra_correct_dets, selected_drops

        else:
            if generate:
                self.corrected_gt(
                    folder_name="/"
                    + self.added_name_glc
                    + "_addedmissing_gt_V"
                    + str(self.v_number)
                    + "/",
                    add_missingboxes=True,
                    missing_gt_boxes=extra_correct_dets,
                )
            else:
                return extra_correct_dets

    def mistakes(self, synthetic=False, generate=True):
        """Finds mistakes in GT via intersection with predictions at low score.
        Args:
            synthetic (bool, optional): Whether to use synthetic approach. Defaults to False.
            generate (bool, optional): Whether to generate new GT labels. Defaults to True.
        """
        wrong_gt = [np.where(iou == 0)[0] for iou in self.ious_gt]
        print(
            "% mistakes GT",
            len(np.concatenate(wrong_gt)) / len(np.concatenate(self.gt_box)) * 100,
        )
        if synthetic:
            ## Add mistakes manually
            sel_ind = 511
            image_data = self.clean_perd_im_names[sel_ind]
            image = Image.open(
                self.gt_images_folder
                + "/"
                + image_data.split(".")[0]
                + "."
                + self.im_format
            )

            def check_intersection(box1, box2):
                """Check if two boxes intersect"""
                x1_min, y1_min, x1_max, y1_max = box1
                x2_min, y2_min, x2_max, y2_max = box2

                if (
                    x1_max < x2_min
                    or x2_max < x1_min
                    or y1_max < y2_min
                    or y2_max < y1_min
                ):
                    return False
                return True

            def generate_random_box(image_size, existing_boxes):
                """
                Generates a random bounding box within the given image size that does not intersect with any of the existing boxes.
                Parameters:
                  image_size (tuple): The size of the image in the format (width, height).
                  existing_boxes (list): A list of existing bounding boxes in the format [box1, box2, ...].
                Returns:
                  new_box (list): A new bounding box or None if a valid box is not found within the maximum number of attempts.
                """

                max_attempts = 100
                for _ in range(max_attempts):
                    x_min = np.random.uniform(0, image_size[0])
                    y_min = np.random.uniform(0, image_size[1])
                    width = np.random.uniform(
                        self.mistakes_lower_size, self.mistakes_upper_size
                    )
                    height = np.random.uniform(
                        self.mistakes_lower_size, self.mistakes_upper_size
                    )
                    x_max = min(x_min + width, image_size[0])
                    y_max = min(y_min + height, image_size[1])

                    new_box = [x_min, y_min, x_max, y_max]

                    if all(
                        not check_intersection(new_box, box) for box in existing_boxes
                    ):
                        return new_box
                return None  # If a valid box is not found in max_attempts

            def add_random_boxes(gt_box, image_size, num_random_boxes=1):
                """
                Add random boxes to the ground truth boxes.
                Args:
                    gt_box (list): List of ground truth boxes.
                    image_size (tuple): Size of the image.
                    num_random_boxes (int, optional): Number of random boxes to add. Defaults to 1.
                Returns:
                    list: List of ground truth boxes with added random boxes.
                """

                new_gt_box = []
                for gts in gt_box:
                    new_gts = []  # Make a copy of the ground truth boxes
                    for _ in range(num_random_boxes):
                        random_box = generate_random_box(image_size, gts)
                        if random_box:
                            new_gts.append(random_box)
                    new_gt_box.append(new_gts)
                return new_gt_box

            new_gt_box = add_random_boxes(
                self.gt_box, image.size, num_random_boxes=self.mistakes_per_image
            )
            print(
                np.sum(len(a) for a in new_gt_box) - np.sum(len(a) for a in self.gt_box)
            )

            ious_mistakes = [
                [calc_iou_np([gt], self.pred_box[i]) for gt in new_gt_box[i]]
                for i in range(len(new_gt_box))
            ]
            ious_gt_mistakes = [np.max(iou, axis=-1) for iou in ious_mistakes]
            wrong_gt_mistakes = [np.where(iou == 0)[0] for iou in ious_gt_mistakes]
            print(
                "% mistakes GT",
                len(np.concatenate(wrong_gt)) / len(np.concatenate(self.gt_box)) * 100,
            )
            print(len(np.concatenate(wrong_gt_mistakes)))
            if generate:
                self.corrected_gt(
                    folder_name="/"
                    + self.added_name_glc
                    + "_synthetic_addmistakes_gt_V"
                    + str(self.v_number)
                    + "/",
                    add_mistakes=new_gt_box,
                )
                self.corrected_gt(
                    folder_name="/"
                    + self.added_name_glc
                    + "_synthetic_addremovemistakes_gt_V"
                    + str(self.v_number)
                    + "/",
                    add_mistakes=new_gt_box,
                    remove_added_mistakes=wrong_gt_mistakes,
                )
            else:
                return wrong_gt_mistakes, new_gt_box
        else:
            if generate:
                self.corrected_gt(
                    folder_name="/"
                    + self.added_name_glc
                    + "_removemistakes_gt_V"
                    + str(self.v_number)
                    + "/",
                    remove_mistakes=True,
                    wrong_gt=wrong_gt,
                )
            else:
                return wrong_gt

    def noisy_boxes(self, synthetic=False, generate=True):
        """Finds noisy GT and replaces them with predictions based on their consistency.
        Args:
            synthetic (bool, optional): Whether to use synthetic approach. Defaults to False.
            generate (bool, optional): Whether to generate new GT labels. Defaults to True.
        """
        if synthetic:
            ## Add noise manually
            def add_deviation_to_boxes(gt_box):
                """
                Add deviation to the given ground truth boxes.

                Args:
                    gt_box (list): A list of lists representing the ground truth boxes.
                        Each inner list contains the coordinates of a box in the format [y1, x1, y2, x2].

                Returns:
                    list: The modified ground truth boxes with added deviation.

                """
                total_boxes = sum(len(box_group) for box_group in gt_box)
                num_boxes_to_modify = max(
                    1, int(self.correct_boxes_to_modify * total_boxes)
                )  # At least 20% of boxes

                # Flatten the list to work with individual boxes
                flat_boxes = [
                    (i, j, box)
                    for i, box_group in enumerate(gt_box)
                    for j, box in enumerate(box_group)
                ]
                flat_boxes = np.array(flat_boxes, dtype=object)

                # Randomly select indices to modify
                indices_to_modify = np.random.choice(
                    len(flat_boxes), num_boxes_to_modify, replace=False
                )

                for idx in indices_to_modify:
                    i, j, box = flat_boxes[idx]
                    y1, x1, y2, x2 = box
                    height = y2 - y1
                    width = x2 - x1

                    # Calculate 10% deviation
                    height_deviation = height * self.correct_boxes_width_height
                    width_deviation = width * self.correct_boxes_width_height

                    y_center = (y1 + y2) / 2
                    x_center = (x1 + x2) / 2

                    y_center_shift = (
                        y_center + np.random.choice([-1, 1]) * height_deviation / 2
                    )
                    x_center_shift = (
                        x_center + np.random.choice([-1, 1]) * width_deviation / 2
                    )

                    # Optionally change the size (either making the box larger or smaller on one side)
                    new_height = height + np.random.choice([-1, 1]) * height_deviation
                    new_width = width + np.random.choice([-1, 1]) * width_deviation

                    # Calculate the new coordinates based on the shifted center and possibly altered size
                    y1_new = y_center_shift - new_height / 2
                    x1_new = x_center_shift - new_width / 2
                    y2_new = y_center_shift + new_height / 2
                    x2_new = x_center_shift + new_width / 2

                    # Update the box in the original structure
                    gt_box[i][j] = [y1_new, x1_new, y2_new, x2_new]

                return gt_box

            modified_gt_box = add_deviation_to_boxes(copy.deepcopy(self.gt_box))

            mod_ious = [
                [calc_iou_np([gt], self.pred_box[i]) for gt in modified_gt_box[i]]
                for i in range(len(modified_gt_box))
            ]
            matched_det_mod = [np.argmax(iou, axis=-1) for iou in mod_ious]
            ious_gt_mod = [np.max(iou, axis=-1) for iou in mod_ious]

            better_boxes_mod = [
                matched_det_mod[i][
                    (
                        np.asarray(self.ciou_perim[i])[matched_det_mod[i]]
                        > self.consist_intersection
                    )
                    * (ious_gt_mod[i] <= self.correct_max_inter)
                    * (
                        np.asarray(self.score_perim[i])[matched_det_mod[i]]
                        >= self.correct_score
                    )
                ]
                for i in range(len(modified_gt_box))
            ]

            corrected_gt_boxes_mod = copy.deepcopy(modified_gt_box)
            for sel_ind in range(len(self.pred_box)):
                i = 0
                for box in self.pred_box[sel_ind]:
                    if i in better_boxes_mod[sel_ind]:
                        gt_ind = np.where(matched_det_mod[sel_ind] == i)[0][0]
                        corrected_gt_boxes_mod[sel_ind][gt_ind] = box
                    i += 1
            if generate:
                self.corrected_gt(
                    folder_name="/"
                    + self.added_name_glc
                    + "_synthetic_noisycorrected_gt_V"
                    + str(self.v_number)
                    + "/",
                    correct_boxes=True,
                    corrected_gt_boxes=corrected_gt_boxes_mod,
                )
                self.corrected_gt(
                    folder_name="/"
                    + self.added_name_glc
                    + "_synthetic_noisyv2_gt_V"
                    + str(self.v_number)
                    + "/",
                    correct_boxes=True,
                    corrected_gt_boxes=modified_gt_box,
                )
            else:
                return corrected_gt_boxes_mod, modified_gt_box
        else:
            better_boxes = [
                self.matched_det[i][
                    (
                        np.asarray(self.ciou_perim[i])[self.matched_det[i]]
                        > self.consist_intersection
                    )
                    * (self.ious_gt[i] <= self.correct_max_inter)
                    * (
                        np.asarray(self.score_perim[i])[self.matched_det[i]]
                        >= self.correct_score
                    )
                ]
                for i in range(len(self.gt_box))
            ]
            corrected_gt_boxes = copy.deepcopy(self.gt_box)
            for sel_ind in range(len(self.pred_box)):
                i = 0
                for box in self.pred_box[sel_ind]:
                    if i in better_boxes[sel_ind]:
                        gt_ind = np.where(self.matched_det[sel_ind] == i)[0][0]
                        corrected_gt_boxes[sel_ind][gt_ind] = box
                    i += 1
            if generate:
                self.corrected_gt(
                    folder_name="/"
                    + self.added_name_glc
                    + "_noisycorrected_gt_V"
                    + str(self.v_number)
                    + "/",
                    correct_boxes=True,
                    corrected_gt_boxes=corrected_gt_boxes,
                )
            else:
                return corrected_gt_boxes


AdvancedLabels(
    glc_method=["noisy"],  # ["mistakes", "md", "noisy"]
    synthetic=True,
    added_name_glc="glc",
    dataset="KITTI",
    model_name="EXP_KITTI_STAC_teacher15_V1",
    labeled_indices_path="num_labeled_15/V1/_train_init_V1.txt",
    added_name="num_labeled_15",
    labeled_portion=15,
    num_labeled=897,
    # #
    # model_name="EXP_KITTI_STAC_teacher10_V1",
    # labeled_indices_path="num_labeled_10/V1/_train_init_V1.txt",
    # model_name="EXP_KITTI_STAC_teacher10_V0",
    # labeled_indices_path="num_labeled_10/V0/_train_init_V0.txt",
    # added_name="num_labeled_10",
    # #
    # dataset="BDD100K",
    # model_name="EXP_BDD100K_STAC_teacher10_V1",
    # labeled_indices_path="num_labeled_10/V1/_train_init_V1.txt",
    # added_name="num_labeled_10",
    # labeled_portion=10,
    # num_labeled=6985,
    # #
    # model_name="EXP_BDD100K_STAC_teacher1_V1",
    # labeled_indices_path="num_labeled_1/V1/_train_init_V1.txt",
    # added_name="num_labeled_1",
    # #
    # iou_consist=0.90,
    # md_max_inter=0,
    md_dropped_gt=0.20,
    mistakes_per_image=5,
    # mistake_upper_size=100,
    # mistake_lower_size=10,
    correct_boxes_to_modify=0.20,
    correct_boxes_width_height=0.20,
    correct_max_inter=0.90,
    correct_score=0.0,
    v_number=1,
)
