# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================

import json
import os
import shutil
import sys

import ijson
import numpy as np

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ssl_utils.parent import Parent_SSL


class ThreeDproblem(Parent_SSL):
    def __init__(
        self,
        method,
        added_pseudo_name,
        labeled_portion=0,
        delta_s=0.9,
        v_number=0,
        *args,
        **kwargs,
    ):
        """
        Initializes 3D class.

        Args:
            method (str): The method to be used between [nomd, nofd, nomdfd, fixmd, highprec, nonoise].
            added_pseudo_name (str): The name to add to saving paths.
            labeled_portion (float, optional): The labeled portion. Defaults to 0.
            delta_s (float, optional): The delta_s score threshold value. Defaults to 0.9.
            v_number (int, optional): The model verion number. Defaults to 0.
            *args: Variable length argument list. Belongs to parent class.
            **kwargs: Arbitrary keyword arguments. Belongs to parent class.
        """
        super().__init__(*args, **kwargs)  # Call parent constructor
        self.added_pseudo_name = added_pseudo_name
        self.delta_s = delta_s
        self.labeled_portion = labeled_portion
        self.v_number = v_number
        if "nomd" in method:  # Select images without any MDs
            self.corrected_pseudo(
                folder_name="/" + self.added_pseudo_name + "_nomd_",
                remove_imgs_with_mds=True,
            )
        if "nofd" in method:  # Remove all FDs
            self.corrected_pseudo(
                folder_name="/" + self.added_pseudo_name + "_nofd_",
                remove_fds=True,
            )
        if "nomdfd" in method:  # Select images without any MDs and remove all FDs
            self.corrected_pseudo(
                folder_name="/" + self.added_pseudo_name + "_nofdnomd_",
                remove_imgs_with_mds=True,
                remove_fds=True,
            )
        if "fixmd" in method:  # Add missing MDs
            self.corrected_pseudo(
                folder_name="/" + self.added_pseudo_name + "_addmd_",
                add_mds=True,
            )
        if "highprec" in method:  # Select high precision detections only
            self.corrected_pseudo(
                folder_name="/" + self.added_pseudo_name + "_highprec_",
                high_precision=True,
            )
        if "nonoise" in method:  # Replace noisy detections with GT
            self.corrected_pseudo(
                folder_name="/" + self.added_pseudo_name + "_nonoise_",
                remove_noise=True,
            )

    def corrected_pseudo(
        self,
        folder_name="/gt/",
        remove_imgs_with_mds=False,
        remove_fds=False,
        remove_noise=False,
        add_mds=False,
        high_precision=False,
    ):
        """
        Generate corrected pseudo labels based on certain criteria.

        Args:
            folder_name (str, optional): Folder name for storing corrected pseudo labels. Defaults to "/gt/".
            remove_imgs_with_mds (bool, optional): Whether to remove images with missing detections. Defaults to False.
            remove_fds (bool, optional): Whether to remove false detections. Defaults to False.
            remove_noise (bool, optional): Whether to replace noisy detections with IoU below 0.75 with GT. Defaults to False.
            add_mds (bool, optional): Whether to add missing detections to the pseudo labels. Defaults to False.
            high_precision (bool, optional): Whether to use high quality predictions only based on IoU 0.9. Defaults to False.
        """
        corrected_pseudo_path = (
            self.dataset_path
            + "/pseudo_labels/"
            + self.added_name
            + folder_name
            + "score"
            + self.det_folder.split("score")[-1]
        )
        if os.path.exists(corrected_pseudo_path):
            shutil.rmtree(corrected_pseudo_path)
        os.makedirs(corrected_pseudo_path)
        sys.path.append(os.path.abspath(self.general_path))
        self.images_data = self.read_pred_folder()  # reset
        self.extract_pseudo_gt_data(new_dets=True)

        if self.dataset == "KITTI":
            self.labeled_portion = self.labeled_portion or 10
            for i in range(len(self.images_data)):
                pred_file_path = self.det_folder + "/" + self.images_data[i]
                gt_file_path = self.gt_labels_folder + "/" + self.images_data[i]
                with open(pred_file_path, "r") as file:
                    pred_lines = file.readlines()

                with open(gt_file_path, "r") as file:
                    lines = file.readlines()
                    gt_lines = []
                    for line in lines:
                        parts = line.strip().split(" ")
                        if len(parts) > 0 and parts[0] in self.used_classes:
                            gt_lines.append(line)

                if high_precision:
                    iou_thr = 0.9
                elif remove_noise:
                    iou_thr = 0.75
                else:
                    iou_thr = 0.5
                gt_selector = np.max(self.perim_ious[i], axis=-1) >= iou_thr
                selector = np.unique(
                    np.argmax(self.perim_ious[i], axis=-1)[gt_selector]
                )
                pred_new_lines = list(np.asarray(pred_lines)[selector])
                gt_new_lines = list(np.asarray(gt_lines)[~gt_selector])

                if remove_imgs_with_mds and self.n_missing_dets[i] > 0:
                    continue
                new_lines = pred_lines
                if remove_noise:
                    new_lines = np.asarray(new_lines, dtype="<U83")
                    gt_lines = np.asarray(gt_lines, dtype="<U83")
                    gt_selector = np.argmax(
                        np.asarray(self.perim_ious[i]).T[selector], axis=-1
                    )
                    new_lines[selector] = gt_lines[gt_selector]
                    new_lines = list(new_lines)
                if remove_fds or high_precision:
                    new_lines = pred_new_lines
                if (
                    add_mds
                ):  # Images originally missing will remain missing, we just fill in detections
                    new_lines += gt_new_lines
                file_path = corrected_pseudo_path + "/" + self.images_data[i]
                if len(new_lines) > 0:
                    with open(file_path, "w") as file:
                        file.writelines(new_lines)
            stac_command = f"PYTHONPATH=/{self.general_path}/src/ python -m SSL_stac --gpu 0 --dataset KITTI --portion_labeled {self.labeled_portion} --tau {self.delta_s} --selection_strategy {folder_name[1:]}score --version_num {self.v_number} --num_epochs 200"

        else:
            self.labeled_portion = self.labeled_portion or 1
            json_path = (
                self.det_folder
                if "json" in self.det_folder
                else self.det_folder + "/pseudo_labels.json"
            )
            with open(json_path, "rb") as json_f:
                pred_items = list(ijson.items(json_f, "item"))
            gt_im_names = [item["name"] for item in self.bdd_data]
            for i, item in enumerate(pred_items):  # self.bdd_data for GT labels
                gt_i = np.where(item["name"] == np.asarray(gt_im_names))[0][0]
                if high_precision:
                    iou_thr = 0.9
                elif remove_noise:
                    iou_thr = 0.75
                else:
                    iou_thr = 0.5
                gt_selector = np.max(self.perim_ious[i], axis=-1) >= iou_thr
                selector = np.unique(
                    np.argmax(self.perim_ious[i], axis=-1)[gt_selector]
                )
                if len(selector) > 0:
                    pred_array = np.asarray(item["labels"])
                    pred_new_items = list(pred_array[selector])
                    gt_lines = np.array(
                        [
                            g
                            for g in self.bdd_data[gt_i]["labels"]
                            if g["category"] in self.used_classes
                        ]
                    )
                    gt_new_items = list(gt_lines[~gt_selector])

                    if remove_imgs_with_mds and self.n_missing_dets[i] > 0:
                        del pred_items[i]["labels"]
                        continue
                    new_items = item["labels"]
                    if remove_noise:
                        new_items = np.asarray(new_items)
                        gt_selector = np.argmax(
                            np.asarray(self.perim_ious[i]).T[selector], axis=-1
                        )
                        new_items[selector] = gt_lines[gt_selector]
                        new_items = list(new_items)
                    if remove_fds or high_precision:
                        new_items = pred_new_items
                    if add_mds:
                        new_items += gt_new_items
                    pred_items[i]["labels"] = new_items
            from decimal import Decimal

            def convert_decimals(obj):
                """Convert decimals to floats"""
                if isinstance(obj, list):
                    return [convert_decimals(i) for i in obj]
                elif isinstance(obj, dict):
                    return {k: convert_decimals(v) for k, v in obj.items()}
                elif isinstance(obj, Decimal):
                    return float(obj)
                else:
                    return obj

            converted_pred_items = [item for item in pred_items if "labels" in item]
            converted_pred_items = convert_decimals(converted_pred_items)
            with open(corrected_pseudo_path + "/pseudo_labels.json", "w") as json_file:
                json.dump(converted_pred_items, json_file, indent=4)
            stac_command = f"PYTHONPATH=/{self.general_path}/src/ python -m SSL_stac --gpu 0 --dataset BDD100K --portion_labeled {self.labeled_portion} --tau {self.delta_s} --selection_strategy {folder_name[1:]}_score --version_num {self.v_number} --num_epochs 200"
        original_data = self.print_data
        orig_det_folder = self.det_folder
        corrected_pseudo_datapath = (
            self.dataset_path
            + "/pseudo_labels/"
            + self.added_name
            + folder_name
            + "data_score"
            + self.det_folder.split("score")[-1]
        )
        self.det_folder = corrected_pseudo_path
        self.images_data = self.read_pred_folder()
        self.extract_pseudo_gt_data(new_dets=True)
        new_data = self.print_data
        os.makedirs(corrected_pseudo_datapath, exist_ok=True)
        filename = corrected_pseudo_datapath + "/output.txt"
        with open(filename, "w") as file:
            file.write(f"original: {original_data} \n")
            file.write(f"new data: {new_data} \n")
        self.det_folder = orig_det_folder
        print(stac_command)


# Create an instance of the child class
ThreeDproblem(
    # dataset="KITTI",
    # det_folder="pseudo_labels/num_labeled_10/score_thr_04_V0/",
    # model_name="EXP_KITTI_STAC_teacher10_V0",
    # labeled_indices_path="num_labeled_10/V0/_train_init_V0.txt",
    # added_name="num_labeled_10",
    dataset="BDD100K",
    det_folder="pseudo_labels/num_labeled_1/score_thr_04_V1",
    model_name="EXP_BDD100K_STAC_teacher1_V1",
    labeled_indices_path="num_labeled_1/V1/_train_init_V1.txt",
    added_name="num_labeled_1",
    method=["nofd"],  # ["nomd", "nofd", "nomdfd", "fixmd", "highprec", "nonoise"]
    added_pseudo_name="3d_problem",
    labeled_portion=0,
    delta_s=0.4,
    v_number=1,
)
