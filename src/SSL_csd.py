# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================

import argparse
import os
import subprocess
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(7, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.BDD100K.bdd_tf_creator import bdd_csd_tfrecords
from datasets.KITTI.kitti_tf_creator import kitti_csd_tfrecords


class CSD:
    """A custom implementation of consistency-based SSL based on CSD:
    Jeong, Jisoo, et al. "Consistency-based semi-supervised learning for object detection."
    Advances in neural information processing systems 32 (2019).
    """

    def __init__(
        self,
        dataset="KITTI",
        ratio=3,
        added_name="_V",
        version_num=0,
        num_epochs=200,
        training_method="baseline",
        early_stopping=False,
        general_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        rand_seed=None,
    ):
        """Initializes CSD

        Args:
            dataset (str, optional): Dataset name. Defaults to "KITTI".
            ratio (int, optional): Ratio unlabeled to labeled data.
            added_name (str, optional): Add name for saving. Defaults to "_V".
            version_num (int, optional): Training version. Defaults to 0.
            num_epochs (int, optional): Number of epochs per AL iteration. Defaults to 200.
            training_method (str, optional): Training method, lossatt, baseline or else "" for MC Dropout+LA. Defaults to "baseline".
            early_stopping (bool, optional): Activates early stopping instead of full training. Defaults to False.
            general_path (str, optional): Path to wdir. Defaults to parent folder of src.
            rand_seed (_type_, optional): Random seed for all random operations. Defaults to None.
        """
        print(
            f"dataset: {dataset}, ratio: {ratio}, added_name: {added_name}, training_method: {training_method}"
            f"version_num: {version_num}, num_epochs: {num_epochs}, rand_seed: {rand_seed}"
        )
        self.ratio = ratio
        if rand_seed is not None:
            np.random.seed(rand_seed)
        self.dataset = dataset
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
        self.model_dir = self.general_path + "models/trained_models/CSD/"
        self.model_name = "efficientdet-d0"
        self.pretrained_ckpt = "efficientdet-d0"
        # Universal CSD config
        self.infer_results = (
            self.general_path + "/results/inference/"
        )  # Path to model inference results
        num_batches = len_train_ims // self.batch_size
        self.num_labeled = int(num_batches * (1 / (ratio + 1) * self.batch_size))
        print(
            f"# total images: {len_train_ims}, # labeled images: {self.num_labeled}, # unlabeled images {len_train_ims-self.num_labeled}"
        )
        self.version_num = version_num
        self.added_name = added_name + str(self.version_num)
        self.tf_active_path = (
            self.general_path
            + "datasets/"
            + self.dataset
            + "/tf_csd/ratio_"
            + str(ratio)
            + "/V"
            + str(self.version_num)
            + "/"
        )
        if not os.path.exists(self.tf_active_path):
            os.makedirs(self.tf_active_path)

    def train(self, num_examples_per_epoch, labeled_only=False):
        """Run model training

        Args:
            num_examples_per_epoch (int): Number of images in training TFRecord
            labeled_only (bool, optional): Train only on labeled data. Defaults to False.
        """
        print("Running Training")
        if labeled_only:
            model_directory = (
                self.model_dir
                + self.dataset
                + "_CSD_ratio_"
                + str(self.ratio)
                + "_labeledonly_"
                + self.added_name
            )
            num_examples_per_epoch = int(
                num_examples_per_epoch
                // self.batch_size
                * (1 / (self.ratio + 1) * self.batch_size)
            )  # Adjust to labeled only
        else:
            model_directory = (
                self.model_dir
                + self.dataset
                + "_CSD_ratio_"
                + str(self.ratio)
                + self.added_name
            )

        script_folder = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_folder)

        if not os.path.exists(model_directory):  # To save train log
            os.makedirs(model_directory)
        if labeled_only:
            command = (
                f"nohup python -m train_flags --train_file_pattern={self.train_tf_labeled} --val_file_pattern={self.val_tf} --model_name={self.model_name} --model_dir={model_directory} --batch_size={self.batch_size} --eval_samples={self.eval_samples} --num_epochs={self.num_epochs} --num_examples_per_epoch={num_examples_per_epoch} --pretrained_ckpt={self.pretrained_ckpt} --hparams={self.hparams} >> "
                + model_directory
                + "/train_log_"
                + str(self.ratio)
                + ".out"
            )
        else:
            command = (
                f"nohup python -m train_flags_SSL --train_file_pattern_labeled={self.train_tf_labeled} --train_file_pattern_unlabeled={self.train_tf_unlabeled} --val_file_pattern={self.val_tf} --ratio={self.ratio} --model_name={self.model_name} --model_dir={model_directory} --batch_size={self.batch_size} --eval_samples={self.eval_samples} --num_epochs={self.num_epochs} --num_examples_per_epoch={num_examples_per_epoch} --pretrained_ckpt={self.pretrained_ckpt} --hparams={self.hparams} >> "
                + model_directory
                + "/train_log_"
                + str(self.ratio)
                + ".out"
            )
        subprocess.run(command, shell=True)

    def get_tfrecord_generator(self):
        """Selects TFRecord generator based on dataset"""
        if self.dataset == "KITTI":
            return kitti_csd_tfrecords
        elif self.dataset == "BDD100K":
            return bdd_csd_tfrecords

    def run(self):
        """Runs CSD"""
        self.train_tf_labeled = (
            self.tf_active_path + "_train_labeled" + self.added_name + ".tfrecord"
        )
        self.train_tf_unlabeled = (
            self.tf_active_path + "_train_unlabeled" + self.added_name + ".tfrecord"
        )
        if not os.path.exists(self.train_tf_labeled):
            self.get_tfrecord_generator()(
                data_dir=self.data_dir,
                output_path=self.tf_active_path,
                classes_to_use=self.select_classes,
                label_map_path=self.label_path,
                num_labeled=self.num_labeled,
                train_indices=self.available_indices,
                saving_name=self.added_name,
            )
        self.train(len(self.available_indices))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initializes CSD")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--ratio", type=int, help="Ratio of labeled data")
    parser.add_argument("--added_name", type=str, help="Add name for saving")
    parser.add_argument("--version_num", type=int, help="Training version")
    parser.add_argument(
        "--num_epochs", type=int, help="Number of epochs per AL iteration"
    )
    parser.add_argument("--training_method", type=str, help="Training method")
    parser.add_argument("--general_path", type=str, help="Path to working directory")
    parser.add_argument(
        "--early_stopping",
        type=bool,
        help="Activates early stopping instead of full training",
    )
    parser.add_argument(
        "--rand_seed", type=int, help="Random seed for all random operations"
    )

    args = parser.parse_args()

    if None in [
        args.dataset,
        args.ratio,
        args.added_name,
        args.version_num,
        args.num_epochs,
        args.training_method,
    ]:
        select_gpu = "0"
        print("selected GPU: ", select_gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = select_gpu
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        CSD().run()
    else:
        CSD(
            dataset=args.dataset,
            ratio=args.ratio,
            added_name=args.added_name,
            version_num=args.version_num,
            num_epochs=args.num_epochs,
            training_method=args.training_method,
            early_stopping=args.early_stopping,
            general_path=args.general_path,
            rand_seed=args.rand_seed,
        ).run()
