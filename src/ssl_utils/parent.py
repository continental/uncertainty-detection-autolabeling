# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================

import ast  # isort:skip
import glob
import json
import os
import shutil
import sys

import ijson
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib.patches import Rectangle
from PIL import Image, ImageEnhance, ImageFilter
from skimage import io

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
P_PATH = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(P_PATH)
from aug import autoaugment
from datasets.BDD100K.bdd_tf_creator import bdd_custom_to_tfrecords
from datasets.KITTI.kitti_tf_creator import kitti_custom_to_tfrecords
from utils_box import calc_iou_np, relativize_uncert


def generate_commands_and_create_dirs(
    added_name,
    dataset,
    save_folder="pseudo_labels",
    labeled_portion=0,
    num_labeled=0,
    version_num="",
):
    """
    Generate training commands and create directories for training.

    Args:
        added_name (str): Name to add to saving paths.
        dataset (str): The dataset name.
        save_folder (str, optional): The folder to save the labels. Defaults to "pseudo_labels".
        labeled_portion (int, optional): The portion of labeled data. Defaults to 0.
        num_labeled (int, optional): The number of labeled data. Defaults to 0.
        version_num (str, optional): The model version number. Defaults to "".
    """
    if dataset == "KITTI":
        num_labeled = num_labeled or 598
        labeled_portion = labeled_portion or 10
        # Define the base paths and parameters
        base_train_path_kitti = (
            P_PATH
            + "/datasets/KITTI/"
            + save_folder
            + "/num_labeled_"
            + str(labeled_portion)
            + "/"
        )
        base_val_path_kitti = P_PATH + "/datasets/KITTI/tf/"
        base_model_dir_kitti = (
            P_PATH
            + "/models/trained_models/STAC/KITTI_STAC_teacher"
            + str(labeled_portion)
            + "_"
        )
        base_log_path_kitti = (
            P_PATH
            + "/models/trained_models/STAC/KITTI_STAC_teacher"
            + str(labeled_portion)
            + "_"
        )
        base_config_path_kitti = P_PATH + "/configs/train/allclasses_orig.yaml"
        # Create directories if they don't exist
        model_dir_kitti = base_model_dir_kitti + added_name + version_num

        os.makedirs(model_dir_kitti, exist_ok=True)

        # Generate commands
        command_kitti = (
            f"nohup python -m train_flags "
            f"--cuda_visible_devices=0 "
            f"--train_file_pattern={base_train_path_kitti}{added_name}/_train.tfrecord "
            f"--val_file_pattern={base_val_path_kitti}_val.tfrecord "
            f"--model_name=efficientdet-d0 "
            f"--model_dir={model_dir_kitti} "
            f"--batch_size=8 "
            f"--eval_samples=1496 "
            f"--num_epochs=200 "
            f"--num_examples_per_epoch={num_labeled} "
            f"--pretrained_ckpt=efficientdet-d0 "
            f"--hparams={base_config_path_kitti} "
            f">> {base_log_path_kitti}{added_name}{version_num}/train_log.out"
        )

        print(command_kitti)
    else:
        num_labeled = num_labeled or 698
        labeled_portion = labeled_portion or 1

        base_train_path_bdd = (
            P_PATH
            + "/datasets/BDD100K/"
            + save_folder
            + "/num_labeled_"
            + str(labeled_portion)
            + "/"
        )
        base_val_path_bdd = P_PATH + "/datasets/BDD100K/tf/"
        base_model_dir_bdd = (
            P_PATH
            + "/models/trained_models/STAC/BDD100K_STAC_teacher"
            + str(labeled_portion)
            + "_"
        )
        base_log_path_bdd = (
            P_PATH
            + "/models/trained_models/STAC/BDD100K_STAC_teacher"
            + str(labeled_portion)
            + "_"
        )
        base_config_path_bdd = P_PATH + "/configs/train/allclasses_orig_BDD.yaml"

        model_dir_bdd = base_model_dir_bdd + added_name + version_num
        os.makedirs(model_dir_bdd, exist_ok=True)
        command_bdd = (
            f"nohup python -m train_flags "
            f"--cuda_visible_devices=0 "
            f"--train_file_pattern={base_train_path_bdd}{added_name}/_train100k.tfrecord "
            f"--val_file_pattern={base_val_path_bdd}_val100k.tfrecord "
            f"--model_name=efficientdet-d0 "
            f"--model_dir={model_dir_bdd} "
            f"--batch_size=8 "
            f"--eval_samples=10000 "
            f"--num_epochs=200 "
            f"--num_examples_per_epoch={num_labeled} "
            f"--pretrained_ckpt=efficientdet-d0 "
            f"--hparams={base_config_path_bdd} "
            f">> {base_log_path_bdd}{added_name}{version_num}/train_log.out"
        )

        print(command_bdd)


class Parent_SSL:
    def __init__(
        self,
        dataset="KITTI",
        det_folder="pseudo_labels/num_labeled_10/score_thr_04_V0/",
        model_name="EXP_KITTI_STAC_teacher10_V0",
        labeled_indices_path="num_labeled_10/V0/_train_init_V0.txt",
        added_name="num_labeled_10",
        general_path=os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ),
    ):
        """
        Initializes the Parent class.

        Args:
            dataset (str, optional): The dataset name. Defaults to "KITTI".
            det_folder (str, optional): The detection folder path. Defaults to "pseudo_labels/num_labeled_10/score_thr_04_V0/".
            model_name (str, optional): The model name. Defaults to "EXP_KITTI_STAC_teacher10_V0".
            labeled_indices_path (str, optional): The labeled indices path. Defaults to "num_labeled_10/V0/_train_init_V0.txt".
            added_name (str, optional): The added name for saving paths. Defaults to "num_labeled_10".
            general_path (str, optional): The general path. Defaults to the parent directory of the current file.

        Raises:
            FileNotFoundError: If the dataset is unknown.

        """
        self.model_name = model_name
        self.labeled_indices_path = labeled_indices_path
        self.general_path = general_path
        self.dataset = dataset
        self.inference_path = (
            general_path + f"/results/inference/SSAL/{model_name}/prediction_data.txt"
        )
        self.dataset_path = general_path + f"/datasets/{dataset}/"
        self.det_folder = self.dataset_path + det_folder
        if dataset == "KITTI":
            self.gt_labels_folder = self.dataset_path + "/training/label_2"
            self.gt_images_folder = self.dataset_path + "/training/image_2"
            self.used_classes = [
                "Car",
                "Van",
                "Truck",
                "Pedestrian",
                "Person_sitting",
                "Cyclist",
                "Tram",
            ]
            self.im_format = "png"

            self.val_indices = []
            with open(self.dataset_path + "/vaL_index_list.txt", "r") as file:
                for line in file:
                    self.val_indices.append(int(line.strip()))

            self.labeled_imnames_all = sorted(
                glob.glob(self.gt_labels_folder + "/*.txt")
            )
            with open(
                self.dataset_path + "/tf_stac/" + labeled_indices_path, "r"
            ) as text_file:
                indices_string = text_file.read()
            self.labeled_indices = list(map(int, indices_string.split()))
            self.labeled_imnames = [
                self.labeled_imnames_all[i] for i in self.labeled_indices
            ]
        elif dataset == "BDD100K":
            self.gt_labels_folder = (
                self.dataset_path + "bdd100k/labels/bdd100k_labels_images_train.json"
            )
            self.gt_images_folder = self.dataset_path + "bdd100k/images/100k/train"
            self.used_classes = [
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
            self.im_format = "jpg"
            self.val_indices = []
            self.labeled_imnames_all = []
            with open(
                self.dataset_path + "bdd100k/labels/remaining_images.txt",
                "r",
            ) as file:
                for line in file:
                    self.labeled_imnames_all.append(str(line.strip()))

            with open(
                self.dataset_path + "/tf_stac/" + labeled_indices_path, "r"
            ) as text_file:
                indices_string = text_file.read()
            self.labeled_indices = list(map(int, indices_string.split()))
            self.labeled_imnames = [
                self.labeled_imnames_all[i] for i in self.labeled_indices
            ]

        else:
            print("Dataset Unknown!")
        self.gt_iou_thr = 0.5
        self.batch_size = 8
        self.added_name = added_name

    @staticmethod
    def apply_manual_augmentation(image, boxes):
        """Apply a random augmentation to the image and boxes."""

        def augment_horizontal_flip(image, boxes):
            """Randomly flip the image and boxes horizontally."""
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            width = image.width
            for box in boxes:
                box[0], box[2] = width - box[2], width - box[0]
            return image, boxes

        def augment_brightness(image, boxes, brightness_range=(0.7, 1.3)):
            """Randomly adjust the brightness of the image."""
            enhancer = ImageEnhance.Brightness(image)
            factor = np.random.uniform(brightness_range[0], brightness_range[1])
            image = enhancer.enhance(factor)
            return image, boxes

        def augment_contrast(image, boxes, contrast_range=(0.7, 1.3)):
            """Randomly adjust the contrast of the image."""
            enhancer = ImageEnhance.Contrast(image)
            factor = np.random.uniform(contrast_range[0], contrast_range[1])
            image = enhancer.enhance(factor)
            return image, boxes

        def augment_blur(image, boxes, blur_radius=(1, 3)):
            """Randomly blur the image."""
            image = image.filter(
                ImageFilter.GaussianBlur(radius=np.random.uniform(*blur_radius))
            )
            return image, boxes

        def augment_noise(image, boxes, noise_level=(5, 20)):
            """Add random noise to the image."""
            np_image = np.array(image)
            noise = np.random.randint(
                noise_level[0], noise_level[1], np_image.shape, dtype="uint8"
            )
            np_image = np_image + noise
            np_image = np.clip(
                np_image, 0, 255
            )  # Ensure values stay within valid range
            image = Image.fromarray(np_image.astype("uint8"))
            return image, boxes

        augmentations = [
            augment_horizontal_flip,
            augment_brightness,
            augment_contrast,
            augment_blur,
            augment_noise,
        ]

        augmentation = np.random.choice(augmentations)
        return augmentation(image, boxes)

    def crop_collage(
        self,
        list_classes,
        list_boxes,
        target_class=["Person_sitting"],
        plot_final=False,
        save_path="",
        gt=False,
        scale=False,
        full_im=False,
        manual_augmentations=False,
        rand_augment_wholeimage=False,
        tfcombine_gtcollage=True,
        filter_ims=None,
        low_scale=0.5,
        high_scale=1.0,
        version_num=0,
    ):
        """
        Collage cropped images from input images based on specified parameters.

        Args:
            extra_classes (list): List of classes for cropping.
            extra_boxes (list): List of bounding boxes for cropping.
            target_class (list, optional): List of target classes for cropping. Defaults to ["Person_sitting"].
            plot_final (bool, optional): Whether to plot last three collages. Defaults to False.
            save_path (str, optional): Path to save the cropped images. Defaults to "".
            gt (bool, optional): Whether to use ground truth labels for cropping. Defaults to False.
            scale (bool, optional): Whether to scale the cropped images. Defaults to False.
            full_im (bool, optional): Whether to use the full image for resampling. Defaults to False.
            manual_augmentations (bool, optional): Whether to apply manual augmentations on the crops. Defaults to False.
            rand_augment_wholeimage (bool, optional): Whether to apply randaug to the whole image. Defaults to False.
            tfcombine_gtcollage (bool, optional): Whether to combine ground truth collage with original images when generating the TFrecord. Defaults to True.
            filter_ims (list, optional): List of indices to filter images. Defaults to None.
            low_scale (float, optional): Lower scale factor for crop box padding. Defaults to 0.5.
            high_scale (float, optional): Higher scale factor for crop box padding. Defaults to 1.0.
            version_num (int, optional): Model version number. Defaults to 0.
        """
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        if gt:
            data_ims = [i.split("/")[-1] for i in self.labeled_imnames]
            collect_labeled_classes = []
            collect_labeled_boxes = []
            for im_name in data_ims:
                gt_objects = self.read_annotations()(
                    os.path.join(self.gt_labels_folder, im_name), self.used_classes
                )
                collect_labeled_classes.append(
                    np.asarray([gt["class"] for gt in gt_objects])
                )
                collect_labeled_boxes.append(
                    np.asarray([gt["bbox"] for gt in gt_objects])
                )
            list_classes = collect_labeled_classes
            list_boxes = collect_labeled_boxes
            print("Using GT for Collage")
        else:
            data_ims = self.images_data
            if filter_ims is not None:
                data_ims = [data_ims[i] for i, sm in enumerate(filter_ims) if sm]
            print("Using Pseudo-Labels for Collage")
        np.random.seed(42)
        cropped_images = []
        crop_boxes = []
        crop_classes = []

        for index, image_data in enumerate(data_ims):
            if self.dataset == "KITTI":
                im_format = "png"
            else:
                im_format = "jpg"
            image = Image.open(
                self.gt_images_folder + "/" + image_data.split(".")[0] + "." + im_format
            )
            classes = np.array(list_classes[index])
            boxes = np.array(list_boxes[index])

            if len(boxes) > 0:
                width = int(np.array(image).shape[1])
                height = int(np.array(image).shape[0])
                if rand_augment_wholeimage:
                    if self.dataset == "KITTI":
                        boxes = np.stack(
                            [boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]],
                            axis=-1,
                        )  # since we need ymin xmin ymax xmax
                    image = tf.convert_to_tensor(np.array(image), dtype=tf.uint8)
                    boxes[:, 0] /= height
                    boxes[:, 1] /= width
                    boxes[:, 2] /= height
                    boxes[:, 3] /= width
                    boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)

                    image, boxes = autoaugment.distort_image_with_randaugment(
                        image, boxes, num_layers=1, magnitude=15
                    )

                    # image, boxes = autoaugment.distort_image_with_autoaugment(
                    #     image, boxes, "v0"
                    # )
                    image = image.numpy().astype(np.uint8)
                    image = Image.fromarray(image)
                    boxes = boxes.numpy().astype(float)

                    boxes[:, 0] *= height
                    boxes[:, 1] *= width
                    boxes[:, 2] *= height
                    boxes[:, 3] *= width
                if self.dataset != "KITTI" or rand_augment_wholeimage:
                    boxes = np.stack(
                        [boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]],
                        axis=-1,
                    )  # since we have xmin ymin xmax ymax

                for cls, box in zip(classes, boxes):
                    if cls in target_class:
                        padding = np.random.uniform(
                            low_scale, high_scale
                        )  # Random padding factor
                        wtransform = (box[2] - box[0]) * padding
                        htransform = (box[3] - box[1]) * padding
                        new_box = [
                            max(0, box[0] - wtransform),
                            max(0, box[1] - htransform),
                            min(width, box[2] + wtransform),
                            min(height, box[3] + htransform),
                        ]

                        # Check for other boxes within or intersecting the cropped area
                        overlapping_boxes = []
                        overlapping_classes = []
                        for other_cls, other_box in zip(classes, boxes):
                            if (
                                (
                                    other_box[0] < new_box[2]
                                    and other_box[0] > new_box[0]
                                )
                                or (
                                    other_box[2] < new_box[2]
                                    and other_box[2] > new_box[0]
                                )
                            ) and (
                                (
                                    other_box[1] < new_box[3]
                                    and other_box[1] > new_box[1]
                                )
                                or (
                                    other_box[3] < new_box[3]
                                    and other_box[3] > new_box[1]
                                )
                            ):
                                overlapping_classes.append(other_cls)
                                adjusted_box = [
                                    max(other_box[0], new_box[0]) - new_box[0],
                                    max(other_box[1], new_box[1]) - new_box[1],
                                    min(new_box[2], other_box[2]) - new_box[0],
                                    min(new_box[3], other_box[3]) - new_box[1],
                                ]
                                if (adjusted_box[2] - adjusted_box[0]) > 2 and (
                                    adjusted_box[3] - adjusted_box[1]
                                ) > 2:
                                    overlapping_boxes.append(adjusted_box)
                        if len(overlapping_boxes) < 100:
                            cropped_image = image.crop(new_box)
                            cropped_images.append(cropped_image)
                            crop_boxes.append(overlapping_boxes)
                            crop_classes.append(overlapping_classes)
                            # fig, ax = plt.subplots()
                            # ax.imshow(image)
                            # for box in boxes:
                            #     rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
                            #     ax.add_patch(rect)
                            # plt.show()
        if os.path.exists(save_path + "bdd100k_labels_images_train.json"):
            os.remove(save_path + "bdd100k_labels_images_train.json")
        max_boxes_per_collage = 100
        json_data = []
        first_entry = True
        i = 0

        # Shuffle the cropped images
        indices = np.arange(len(cropped_images))
        np.random.shuffle(indices)
        cropped_images = [cropped_images[id] for id in indices]
        crop_boxes = [crop_boxes[id] for id in indices]
        crop_classes = [crop_classes[id] for id in indices]
        while cropped_images:
            collage_width = image.width
            collage_height = image.height
            collage = Image.new("RGB", (collage_width, collage_height))
            x_offset = 0
            labels = []
            box_count = 0

            while (
                x_offset < collage_width
                and cropped_images
                and box_count < max_boxes_per_collage
            ):
                img = cropped_images.pop(0)
                original_boxes = crop_boxes.pop(0)
                original_classes = crop_classes.pop(0)
                original_width, original_height = img.width, img.height

                if len(cropped_images) == 0:
                    new_width = collage_width - x_offset
                else:
                    new_width = int(img.width * collage_height / img.height)

                img = img.resize((new_width, collage_height), Image.LANCZOS)
                if x_offset + img.width > collage_width:
                    img = img.resize(
                        (collage_width - x_offset, collage_height), Image.LANCZOS
                    )

                if manual_augmentations:
                    img, original_boxes = self.apply_manual_augmentation(
                        img, original_boxes
                    )

                if scale:
                    quadrant_sizes = [
                        (np.ceil(img.width * 0.75), np.ceil(img.height * 0.75)),
                        (np.ceil(img.width * 0.25), np.ceil(img.height * 0.75)),
                        (np.ceil(img.width * 0.75), np.ceil(img.height * 0.25)),
                        (np.ceil(img.width * 0.25), np.ceil(img.height * 0.25)),
                    ]
                    quadrant_sizes = np.asarray(quadrant_sizes).astype(int)

                    collage_part = Image.new("RGB", (img.width, img.height))

                    # Resize and place each quadrant
                    y_offset_part = 0
                    x_offset_part = 0
                    positions = []
                    for j in range(4):
                        resized_img = img.resize(quadrant_sizes[j], Image.LANCZOS)
                        if j == 2:  # Reset for second row
                            y_offset_part = quadrant_sizes[0][1]
                            x_offset_part = 0
                        collage_part.paste(resized_img, (x_offset_part, y_offset_part))
                        positions.append((x_offset + x_offset_part, y_offset_part))
                        x_offset_part += resized_img.width  # Move x offset to the right
                        if j == 1:  # Reset x_offset after the first row
                            x_offset_part = 0

                    for idx, pos in enumerate(positions):
                        scale_x = quadrant_sizes[idx][0] / original_width
                        scale_y = quadrant_sizes[idx][1] / original_height
                        for cls, original_box in zip(original_classes, original_boxes):
                            new_x1 = pos[0] + scale_x * original_box[0]
                            new_y1 = pos[1] + scale_y * original_box[1]
                            new_x2 = pos[0] + scale_x * original_box[2]
                            new_y2 = pos[1] + scale_y * original_box[3]
                            new_box = [new_x1, new_y1, new_x2, new_y2]
                            labels.append([cls, new_box])
                            box_count += 1
                            if box_count >= max_boxes_per_collage:
                                break
                    collage.paste(collage_part, (x_offset, 0))
                else:
                    collage.paste(img, (x_offset, 0))

                    scale_x = img.width / original_width
                    scale_y = img.height / original_height

                    for cls, original_box in zip(original_classes, original_boxes):
                        new_x1 = x_offset + scale_x * original_box[0]
                        new_y1 = scale_y * original_box[1]
                        new_x2 = x_offset + scale_x * original_box[2]
                        new_y2 = scale_y * original_box[3]
                        new_box = [new_x1, new_y1, new_x2, new_y2]
                        labels.append([cls, new_box])
                        box_count += 1

                        if box_count >= max_boxes_per_collage:
                            break

                x_offset += img.width

            # Save the collage
            if self.dataset == "KITTI":
                collage.save(save_path + f"{10000+i:06}.png")

                with open(save_path + f"{10000+i:06}.txt", "w") as file:
                    for clsbox in labels:
                        cls, label = clsbox
                        file.write(
                            f"{cls} 0.0 0 0 {label[0]} {label[1]} {label[2]} {label[3]} 0 0 0 0 0 0 0\n"
                        )
            else:
                collage.save(save_path + f"collage_{i}.jpg")
                data = []
                for d_i, clsbox in enumerate(labels):
                    cls, bbox = clsbox
                    label_data = {
                        "id": str(i) + str(d_i),
                        "attributes": {
                            "occluded": False,
                            "truncated": False,
                        },
                        "category": cls,
                        "box2d": {
                            "x1": bbox[0],
                            "x2": bbox[2],
                            "y1": bbox[1],
                            "y2": bbox[3],
                        },
                    }
                    data.append(label_data)

                json_data.append(
                    {
                        "name": f"collage_{i}.jpg",
                        "attributes": {
                            "weather": "clear",
                            "timeofday": "daytime",
                            "scene": "city street",
                        },
                        "timestamp": 10000,
                        "labels": data,
                    }
                )
                if not first_entry:
                    json_data.append(",\n")
                first_entry = False
            i += 1

        val_indices = []
        if self.dataset == "KITTI":
            with open(
                self.general_path + "/datasets/KITTI/vaL_index_list.txt", "r"
            ) as file:
                for line in file:
                    val_indices.append(int(line.strip()))

            kitti_custom_to_tfrecords(
                save_path,
                save_path,
                self.general_path + "/datasets/KITTI/kitti.pbtxt",
                [s.lower() for s in self.used_classes],
                get_orig=tfcombine_gtcollage,
                data_dir_orig=self.general_path + "/datasets/KITTI/training",
                validation_indices=val_indices,
                train_indices=self.labeled_indices,
            )
        else:
            with open(save_path + "bdd100k_labels_images_train.json", "a") as file:
                json.dump(json_data, file, indent=4)

            bdd_custom_to_tfrecords(
                save_path,
                save_path,
                self.general_path + "/datasets/BDD100K/bdd.pbtxt",
                [s.lower() for s in self.used_classes],
                get_orig=tfcombine_gtcollage,
                train_indices=self.labeled_indices,
                data_dir_orig=self.general_path + "/datasets/BDD100K/bdd100k/",
            )

        generate_commands_and_create_dirs(
            save_path.split("/")[-2],
            self.dataset,
            save_folder="collage_crops",
            labeled_portion=self.added_name.split("_")[-1],
            num_labeled=len(self.labeled_indices) + i,
            version_num="_V" + str(version_num),
        )

        if full_im:
            save_path = save_path[:-1] + "_full_image/"
            os.makedirs(save_path, exist_ok=True)

            cropped_images = []
            crop_boxes = []
            crop_classes = []

            for index, image_data in enumerate(data_ims):
                if self.dataset == "KITTI":
                    im_format = "png"
                else:
                    im_format = "jpg"
                image = Image.open(
                    self.gt_images_folder
                    + "/"
                    + image_data.split(".")[0]
                    + "."
                    + im_format
                )
                classes = np.array(list_classes[index])
                boxes = np.array(list_boxes[index])
                if len(boxes) > 0:
                    for cls, box in zip(classes, boxes):
                        if cls in target_class:
                            if manual_augmentations:
                                image = tf.convert_to_tensor(
                                    np.array(image), dtype=tf.uint8
                                )
                                width = int(image.shape[1])
                                height = int(image.shape[0])
                                boxes[:, 0] /= height
                                boxes[:, 1] /= width
                                boxes[:, 2] /= height
                                boxes[:, 3] /= width
                                boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)

                                image, boxes = (
                                    autoaugment.distort_image_with_randaugment(
                                        image, boxes, num_layers=1, magnitude=15
                                    )
                                )

                                # image, boxes = autoaugment.distort_image_with_autoaugment(
                                #     image, boxes, "v0"
                                # )
                                image = image.numpy().astype(np.uint8)
                                image = Image.fromarray(image)
                                boxes = boxes.numpy().astype(float)

                                boxes[:, 0] *= height
                                boxes[:, 1] *= width
                                boxes[:, 2] *= height
                                boxes[:, 3] *= width
                            if self.dataset != "KITTI":
                                boxes = np.stack(
                                    [
                                        boxes[:, 1],
                                        boxes[:, 0],
                                        boxes[:, 3],
                                        boxes[:, 2],
                                    ],
                                    axis=-1,
                                )
                            cropped_images.append(image)
                            crop_boxes.append(boxes)
                            crop_classes.append(classes)
                            break

            if os.path.exists(save_path + "bdd100k_labels_images_train.json"):
                os.remove(save_path + "bdd100k_labels_images_train.json")
            json_data = []
            first_entry = True
            i = 0
            while cropped_images:
                img = cropped_images.pop(0)
                original_boxes = crop_boxes.pop(0)
                original_classes = crop_classes.pop(0)

                if self.dataset == "KITTI":
                    img.save(save_path + f"{10000+i:06}.png")

                    with open(save_path + f"{10000+i:06}.txt", "w") as file:
                        for d_i in range(len(original_classes)):
                            cls = original_classes[d_i]
                            bbox = original_boxes[d_i]
                            file.write(
                                f"{cls} 0.0 0 0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} 0 0 0 0 0 0 0\n"
                            )
                else:
                    img.save(save_path + f"collage_{i}.jpg")
                    data = []
                    for d_i in range(len(original_classes)):
                        cls = original_classes[d_i]
                        bbox = original_boxes[d_i]
                        label_data = {
                            "id": str(i) + str(d_i),
                            "attributes": {
                                "occluded": False,
                                "truncated": False,
                            },
                            "category": cls,
                            "box2d": {
                                "x1": bbox[0],
                                "y1": bbox[1],
                                "x2": bbox[2],
                                "y2": bbox[3],
                            },
                        }
                        data.append(label_data)

                    json_data.append(
                        {
                            "name": f"collage_{i}.jpg",
                            "attributes": {
                                "weather": "clear",
                                "timeofday": "daytime",
                                "scene": "city street",
                            },
                            "timestamp": 10000,
                            "labels": data,
                        }
                    )
                    if not first_entry:
                        json_data.append(",\n")
                    first_entry = False
                i += 1

            val_indices = []
            if self.dataset == "KITTI":
                with open(
                    self.general_path + "/datasets/KITTI/vaL_index_list.txt", "r"
                ) as file:
                    for line in file:
                        val_indices.append(int(line.strip()))

                kitti_custom_to_tfrecords(
                    save_path,
                    save_path,
                    self.general_path + "/datasets/KITTI/kitti.pbtxt",
                    [s.lower() for s in self.used_classes],
                    get_orig=True,
                    data_dir_orig=self.general_path + "/datasets/KITTI/training",
                    validation_indices=val_indices,
                    train_indices=self.labeled_indices,
                )
            else:
                with open(save_path + "bdd100k_labels_images_train.json", "a") as file:
                    json.dump(json_data, file, indent=4)

                bdd_custom_to_tfrecords(
                    save_path,
                    save_path,
                    self.general_path + "/datasets/BDD100K/bdd.pbtxt",
                    [s.lower() for s in self.used_classes],
                    get_orig=True,
                    train_indices=self.labeled_indices,
                    data_dir_orig=self.general_path + "/datasets/BDD100K/bdd100k/",
                )

            generate_commands_and_create_dirs(
                save_path.split("/")[-2],
                self.dataset,
                save_folder="collage_crops",
                labeled_portion=self.added_name.split("_")[-1],
                num_labeled=len(self.labeled_indices) + i,
                version_num="_V" + str(version_num),
            )
        if plot_final:
            label_files = [f for f in os.listdir(save_path) if f.endswith(".txt")][:3]

            # Loop through each label file and corresponding image
            for label_file in label_files:
                image_file = label_file.replace(".txt", ".png")
                image_path = os.path.join(save_path, image_file)
                label_path = os.path.join(save_path, label_file)

                # Load the image
                img = Image.open(image_path)
                fig, ax = plt.subplots(1)
                ax.imshow(img)

                # Read the label file and plot each bounding box
                with open(label_path, "r") as file:
                    for line in file:
                        parts = line.strip().split()
                        x1, y1, x2, y2 = map(float, parts[4:8])
                        rect = patches.Rectangle(
                            (x1, y1),
                            x2 - x1,
                            y2 - y1,
                            linewidth=1,
                            edgecolor="r",
                            facecolor="none",
                        )
                        ax.add_patch(rect)
                plt.show()

    def plot_kitti_example(self, im_name, det_path, label_filter=None):
        """
        Plots an example image from the KITTI dataset with bounding box annotations.

        Args:
            im_name (str): The name of the image file.
            det_path (str): The path to the folder containing the detection files.
            label_filter (list, optional): A list of labels to filter the annotations. Defaults to None.
        """
        img_path = self.gt_images_folder + f"/{im_name}." + self.im_format
        colors = sns.color_palette("Paired", 9 * 2)
        img = np.array(io.imread(img_path), dtype=np.int32)
        with open(det_path + f"/{im_name}." + "txt", "r") as f:
            labels = f.readlines()
        if label_filter is not None:
            labels = np.asarray(labels)[label_filter]
        fig = plt.figure()
        plt.imshow(img)
        for line in labels:
            line = line.split()
            label, _, _, _, x1, y1, x2, y2, _, _, _, _, _, _, _ = line
            x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
            if label in self.used_classes:
                plt.gca().add_patch(
                    Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        edgecolor=colors[self.used_classes.index(label)],
                        facecolor="none",
                    )
                )
                plt.text(
                    x1 + 3,
                    y1 + 3,
                    label,
                    bbox=dict(
                        facecolor=colors[self.used_classes.index(label)], alpha=0.5
                    ),
                    fontsize=7,
                    color="k",
                )
        plt.tight_layout()
        plt.show()

    def plot_bdd_example(self, im_index, det_path=None):
        """
        Plot an example image with bounding box annotations from the BDD dataset.

        Args:
            im_index (int): The index of the image to plot.
            det_path (str, optional): The path to the detection labels file. If not provided, the default path will be used.
        """
        if det_path is None:
            det_path = self.gt_labels_folder
        colors = sns.color_palette("Paired", 9 * 2)
        with open(det_path, "rb") as json_f:
            item = list(ijson.items(json_f, "item"))[im_index]
            img_name = item["name"]
            img_path = os.path.join(self.gt_images_folder, img_name)
            if os.path.isfile(img_path):
                image = np.array(io.imread(img_path), dtype=np.int32)
                width = int(image.shape[1])
                height = int(image.shape[0])

                xmins = []
                ymins = []
                xmaxs = []
                ymaxs = []
                classes = []
                labels = []
                occluded = []
                truncated = []

                current_labels = item["labels"]
                for label in current_labels:
                    category = label["category"]
                    if category in self.used_classes:
                        labels.append(self.used_classes.index(category))
                        classes.append(category.encode("utf8"))
                        attributes = label["attributes"]
                        occluded.append(int(attributes["occluded"] == "true"))
                        truncated.append(int(attributes["truncated"] == "true"))
                        box2d = label["box2d"]
                        xmins.append(float(box2d["x1"]) / width)
                        ymins.append(float(box2d["y1"]) / height)
                        xmaxs.append(float(box2d["x2"]) / width)
                        ymaxs.append(float(box2d["y2"]) / height)
                fig = plt.figure()
                plt.imshow(image)
                for i in range(len(xmins)):
                    x1, y1, x2, y2 = map(
                        float,
                        [
                            xmins[i] * float(width),
                            ymins[i] * float(height),
                            xmaxs[i] * float(width),
                            ymaxs[i] * float(height),
                        ],
                    )
                    plt.gca().add_patch(
                        Rectangle(
                            (x1, y1),
                            x2 - x1,
                            y2 - y1,
                            linewidth=2,
                            edgecolor=colors[labels[i]],
                            facecolor="none",
                        )
                    )
                    plt.text(
                        x1 + 3,
                        y1 + 3,
                        str(classes[i])[2:-1],
                        bbox=dict(facecolor=colors[labels[i]], alpha=0.5),
                        fontsize=5,
                        color="k",
                    )

                plt.tight_layout()
                plt.show()

    @staticmethod
    def read_predictions(inference_path, selection_strategy, predictions=False):
        """
        Reads predictions from a file and processes them based on the selection strategy.

        Args:
            inference_path (str): The path to the file containing the predictions.
            selection_strategy (str): The selection strategy to be applied to the predictions.
            predictions (bool, optional): Whether to include the prediction box and class in the saved data. Defaults to False.

        Returns:
            tuple: A tuple containing the processed predictions and additional information.
                - The unique image names (numpy.ndarray)
                - The per-image scores (list)
                - The per-image classes (list) (optional)
                - The per-image bounding boxes (list) (optional)
                - The per-image consistency IoU (list) (optional)
                - The per-image consistency classes (list) (optional)
        """
        f = open(inference_path, "r")
        dets = f.readlines()
        detections = [ast.literal_eval(d.replace("inf", "2e308")) for d in dets]
        act_consistency = "cons_iou" in detections[0] in detections
        if act_consistency:
            pred_ccls = []
            pred_ciou = []
        pred_imgs_names = []
        pred_imgs_names.append(detections[0]["image_name"])
        img_name = detections[0]["image_name"]
        i = 0
        # Get per image score
        if predictions:
            per_image_classes = []
            per_image_boxes = []
        per_image_score = []
        while i < len(detections):
            curr_img_name = detections[i]["image_name"]
            if "calib" in selection_strategy:
                box_calib_mode = "iso_perclscoo_"
                class_calib_mode = "iso_percls_"
                if "box" in selection_strategy:
                    add_mode = box_calib_mode
                else:
                    add_mode = class_calib_mode
            else:
                if "box" in selection_strategy or "class" in selection_strategy:
                    add_mode = "uncalib_"
                else:
                    add_mode = ""
            if predictions:
                temp_box = []
                temp_class = []
            if "alluncert" in selection_strategy:
                temp_ssl_score = [[], [], []]
            elif "epuncert" in selection_strategy or "ental" in selection_strategy:
                temp_ssl_score = [[], []]
            else:
                temp_ssl_score = []
            if act_consistency:
                temp_pred_ccls = []
                temp_pred_ciou = []

            # Iterate over detections for current image
            while curr_img_name == img_name:
                if "alluncert" in selection_strategy:
                    if "calib" in selection_strategy:
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

                elif "epuncert" in selection_strategy:
                    if "calib" in selection_strategy:
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

                elif "ental" in selection_strategy:
                    if "calib" in selection_strategy:
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

                else:
                    try_last = add_mode + selection_strategy.split("_")[-1]
                    try_full = add_mode + selection_strategy
                    if try_last in detections[i] or try_full in detections[i]:
                        if try_full in detections[i]:
                            sel_key = try_full
                        else:
                            sel_key = try_last

                        sel_metric = detections[i][sel_key]
                        if "box" in selection_strategy and "norm" in selection_strategy:
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
                if predictions:
                    temp_box.append(detections[i]["bbox"])
                    temp_class.append(detections[i]["class"])
                if act_consistency:
                    temp_pred_ccls.append(detections[i]["cons_cls"])
                    temp_pred_ciou.append(detections[i]["cons_iou"])
                i += 1  # Check if last detection
                if i == len(detections):
                    break
                curr_img_name = detections[i]["image_name"]
                pred_imgs_names.append(detections[i]["image_name"])

            # Concatenate scores per image
            per_image_score.append(temp_ssl_score)

            if predictions:
                per_image_boxes.append(temp_box)
                per_image_classes.append(temp_class)
            if act_consistency:
                pred_ciou.append(temp_pred_ciou)
                pred_ccls.append(temp_pred_ccls)

            img_name = curr_img_name
        output = [
            np.unique(pred_imgs_names),
            per_image_score,
        ]
        if predictions:
            output.append(per_image_classes)
            output.append(per_image_boxes)
        if act_consistency:
            output.append(pred_ciou)
            output.append(pred_ccls)

        return tuple(output)

    @staticmethod
    def _read_kitti_annotations(file_path, used_classes):
        """
        Read the annotations from a KITTI file and return a list of objects.

        Args:
            file_path (str): The path to the KITTI file.
            used_classes (list): A list of classes to filter the annotations.

        Returns:
            list: A list of objects containing the class and bounding box coordinates.

        """
        objects = []
        with open(file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(" ")
                if len(parts) > 0 and parts[0] in used_classes:
                    obj = {
                        "class": parts[0],
                        "bbox": [
                            float(parts[4]),
                            float(parts[5]),
                            float(parts[6]),
                            float(parts[7]),
                        ],
                    }
                    objects.append(obj)
        return objects

    def _read_bdd100k_annotations(
        self, file_path, used_classes, gt=True, given_data=None
    ):
        """
        Read BDD100K annotations from a JSON file.

        Args:
            file_path (str): The path to the JSON file.
            used_classes (list): A list of classes to filter the annotations.
            gt (bool, optional): Whether to read ground truth annotations. Defaults to True.
            given_data (list, optional): A pre-loaded list of BDD100K annotations. Defaults to None.

        Returns:
            list: A list of dictionaries containing the filtered annotations.
                Each dictionary contains the class name and the bounding box coordinates.
        """
        im_name = file_path.split("/")[-1]
        objects = []
        if gt:
            json_path = file_path.split("json")[0] + "json"
            if getattr(self, "bdd_data", None) is None:
                with open(json_path, "r") as file:
                    self.bdd_data = json.load(file)  # Load the entire JSON data
            bdd_data = self.bdd_data
        else:
            bdd_data = given_data

        im_names = np.asarray([item["name"] for item in bdd_data])
        im_ind = np.where(im_names == im_name)[0][0]
        im_dets = bdd_data[im_ind]["labels"]
        for obj in im_dets:
            if obj["category"] in used_classes:
                # Assuming the bbox is in 'bbox' and formatted as [x1, y1, x2, y2]
                bbox = obj["box2d"]
                objects.append(
                    {
                        "class": obj["category"],
                        "bbox": [
                            float(bbox["y1"]),
                            float(bbox["x1"]),
                            float(bbox["y2"]),
                            float(bbox["x2"]),
                        ],
                    }
                )
        return objects

    def read_annotations(self):
        """Reads and returns the annotations based on the selected dataset."""
        if self.dataset == "KITTI":
            return self._read_kitti_annotations
        elif self.dataset == "BDD100K":
            return self._read_bdd100k_annotations

    def _get_cls_dist(self, val=False):
        """
        Calculate the class distribution of labeled images.

        Args:
            val (bool): If True, calculate the class distribution for validation images.
                        If False, calculate the class distribution for training images.

        Returns:
            list: A list containing the count of each class in the images.
        """

        im_names = (
            [self.labeled_imnames_all[i] for i in self.val_indices]
            if val
            else self.labeled_imnames
        )
        self.collect_labeled_classes = []
        for im_name in im_names:
            gt_objects = self.read_annotations()(
                os.path.join(self.gt_labels_folder, im_name), self.used_classes
            )
            gt_classes = np.asarray([gt["class"] for gt in gt_objects])
            self.collect_labeled_classes.append(gt_classes)
        all_labeled_classes = np.concatenate(self.collect_labeled_classes)
        cls_dist = [np.sum(all_labeled_classes == c) for c in self.used_classes]
        return cls_dist

    def _plot_cls_score(self, perc_score, cls_dist):
        """
        Plot the class score and number of detections for each class.

        Args:
            perc_score (list): List of scores per class.
            cls_dist (list): List of number of detections per class.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plotting the scatter plot
        axes[0].scatter(self.used_classes, perc_score, color="blue")
        axes[0].set_ylabel("Score per Class")
        axes[0].set_xticks(self.used_classes)
        axes[0].set_xticklabels(self.used_classes, rotation=45, ha="right")

        # Plotting the bar chart
        bars = axes[1].bar(self.used_classes, cls_dist, color="green")
        axes[1].set_ylabel("Number of Detections")
        axes[1].set_xticklabels(self.used_classes, rotation=45, ha="right")

        # Adding the text labels above the bars
        for bar in bars:
            yval = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                int(yval),
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.show()

    def _weight_generator(self):
        """Returns the weight generator based on the dataset."""
        if self.dataset == "KITTI":
            return self._weight_generator_kitti
        elif self.dataset == "BDD100K":
            return self._weight_generator_bdd

    def _weight_generator_bdd(self, perdet_score, out_path="", gt=True):
        """
        Include a score per detection in the pseudo labels / GT of the BDD dataset.

        Args:
            perdet_score (dict): Dictionary containing per detection scores.
            out_path (str, optional): Output path for saving labels with score. Defaults to "".
            gt (bool, optional): Flag indicating whether to use ground truth labels. Defaults to True.
        """

        if getattr(self, "bdd_data", None) is None:
            with open(self.gt_labels_folder, "r") as file:
                self.bdd_data = json.load(file)  # Load the entire JSON data
        for i in range(len(self.bdd_data)):
            if self.bdd_data[i]["name"] in self.labeled_imnames_all:
                for j in range(len(self.bdd_data[i]["labels"])):
                    cat = self.bdd_data[i]["labels"][j]["category"]
                    if cat in self.used_classes:
                        self.bdd_data[i]["labels"][j]["pseudo_score"] = np.round(
                            perdet_score[cat], 2
                        )
        with open(
            self.dataset_path + "/pseudo_labels/" + out_path + "pseudo_labels.json", "w"
        ) as json_file:
            json.dump(self.bdd_data, json_file, indent=4)

    def _weight_generator_kitti(
        self,
        perdet_score,
        out_path="",
        gt=False,
    ):
        """
        Include a score per detection in the pseudo labels / GT of the KITTI dataset.

        Args:
            perdet_score (dict): Dictionary containing per detection scores.
            out_path (str, optional): Output path for saving labels with score. Defaults to "".
            gt (bool, optional): Flag indicating whether to use ground truth labels. Defaults to False.

        """
        if gt:
            source_folder = self.gt_labels_folder
        else:
            source_folder = self.det_folder
        pseudo_save_path = self.dataset_path + "/pseudo_labels/" + out_path
        pred_im_names = glob.glob(source_folder + "/*.txt")
        pred_im_names = [p.split("/")[-1] for p in pred_im_names]
        for i in range(len(pred_im_names)):
            pred_file_path = source_folder + "/" + pred_im_names[i]
            with open(pred_file_path, "r") as file:
                pred_lines = file.readlines()
            pred_new_lines = []
            for line in pred_lines:
                parts = line.split()
                if gt:
                    class_type = parts[0]
                    if class_type in self.used_classes:
                        score = perdet_score[class_type]  # Fetch score
                        parts[-1] = str(score) + "\n"
                        updated_line = " ".join(parts)
                        pred_new_lines.append(updated_line)
                else:
                    class_type = parts[0]
                    score = perdet_score[class_type]  # Fetch score
                    parts[-1] = str(np.round(score, 2)) + "\n"
                    updated_line = " ".join(parts)
                    pred_new_lines.append(updated_line)

            if len(pred_new_lines) > 0:
                file_path = pseudo_save_path + "/" + pred_im_names[i]
                with open(file_path, "w") as file:
                    file.writelines(pred_new_lines)

    def _curriculum_generator(self, perdet_score, out_path):
        """
        RCF: Generates a curriculum for training based on class distribution,
        splitting the dataset into rare and common examples based on the occuring classes.

        Args:
            perdet_score (list): List of per-detection scores.
            out_path (str): Output path for saving the split indices.
        """
        perim_score = []
        for perim_cls in self.collect_labeled_classes:
            perim_score.append(np.mean([perdet_score[s] for s in perim_cls]))

        unlabeled_start = len(self.labeled_imnames) // self.batch_size
        sort_names = np.argsort(perim_score)
        common_im_names = [
            self.labeled_imnames[i].split("/")[-1]
            for i in sort_names[: len(self.labeled_imnames) - unlabeled_start]
        ]
        rare_im_names = [
            self.labeled_imnames[i].split("/")[-1]
            for i in sort_names[-unlabeled_start:]
        ]
        file_path = self.dataset_path + "/pseudo_labels/" + out_path
        with open(file_path, "w") as file:
            for name in common_im_names:
                file.write(f"{name}\n")
            file.write("---\n")  # Marker to separate the lists
            for name in rare_im_names:
                file.write(f"{name}\n")

    def weight_images_cls_dist(
        self,
        added_name="",
        rcf=False,
        lowest_weight=1,
        highest_weight=10,
        version_num=0,
        delta_s=0.4,
        visualize=False,
    ):
        """
        Weight the images based on class distribution.

        Args:
            extend_name (str, optional): Additional name for saving paths. Defaults to "".
            rcf (bool, optional): Whether to generate RCF curriculum. Defaults to False.
            lowest_weight (int, optional): Lowest class weight value. Defaults to 1.
            highest_weight (int, optional): Highest class weight value. Defaults to 10.
            version_num (int, optional): Model version number. Defaults to 0.
            delta_s (float, optional): Detection score threshold. Defaults to 0.4.
            visualize (bool, optional): Whether to visualize the class scores. Defaults to False.
        """

        def _scale_vals(occ_perc, lowest_weight=1, highest_weight=10):
            """Scale the per class score based on the lowest_weight and highest_weight."""
            scaled_occ_perc = lowest_weight + (
                (occ_perc - min(occ_perc)) * (highest_weight - lowest_weight)
            ) / (max(occ_perc) - min(occ_perc))
            return scaled_occ_perc, highest_weight

        cls_dist = self._get_cls_dist()
        mask = [x > 1 for x in cls_dist]
        inv_ratio_log = [1 / np.log(x) for x in np.asarray(cls_dist)[mask]]
        scaled_occ_perc, hw = _scale_vals(
            inv_ratio_log, lowest_weight=lowest_weight, highest_weight=highest_weight
        )
        scaled_occ_perc = list(scaled_occ_perc)
        for i in np.where(np.invert(mask))[0]:
            scaled_occ_perc.insert(i, hw)
        perdet_score = {
            uc: o for uc, o in zip(self.used_classes, np.round(scaled_occ_perc, 5))
        }
        out_path = self.added_name + "/" + added_name + "_cblog" + str(hw) + "_imscore/"
        selection_strategy_cb = out_path.split("/")[1]
        portion_labeled = self.added_name.split("_")[-1]
        if not os.path.exists(self.dataset_path + "/pseudo_labels/" + out_path):
            os.makedirs(self.dataset_path + "/pseudo_labels/" + out_path)
            self._weight_generator()(perdet_score, out_path=out_path, gt=True)
        stac_command = f"PYTHONPATH=/{self.general_path}/src/ python -m SSL_stac --gpu 0 --dataset {self.dataset} --portion_labeled {portion_labeled} --tau {delta_s} --selection_strategy score --teacher_strategy {selection_strategy_cb} --clw {highest_weight} --version_num {version_num} --num_epochs 200"
        print(stac_command)

        if rcf:
            out_path = (
                self.added_name
                + "/"
                + added_name
                + "_curriculum_learning"
                + str(hw)
                + ".txt"
            )
            if not os.path.exists(self.dataset_path + "/pseudo_labels/" + out_path):
                self._curriculum_generator(perdet_score, out_path=out_path)
            selection_strategy = out_path.split("/")[1].split(".")[0]
            stac_command = f"PYTHONPATH=/{self.general_path}/src/ python -m SSL_stac --gpu 0 --dataset {self.dataset} --portion_labeled {portion_labeled} --tau {delta_s} --selection_strategy score --teacher_strategy {selection_strategy} --clw {highest_weight} --version_num {version_num} --num_epochs 200\n"
            stac_command += f"PYTHONPATH=/{self.general_path}/src/ python -m SSL_stac --gpu 0 --dataset {self.dataset} --portion_labeled {portion_labeled} --tau {delta_s} --selection_strategy score --teacher_strategy {selection_strategy}_aug --clw {highest_weight} --version_num {version_num} --num_epochs 200\n"

            selection_strategy = selection_strategy_cb + "_" + selection_strategy
            stac_command += f"PYTHONPATH=/{self.general_path}/src/ python -m SSL_stac --gpu 0 --dataset {self.dataset} --portion_labeled {portion_labeled} --tau {delta_s} --selection_strategy score --teacher_strategy {selection_strategy} --clw {highest_weight} --version_num {version_num} --num_epochs 200\n"
            stac_command += f"PYTHONPATH=/{self.general_path}/src/ python -m SSL_stac --gpu 0 --dataset {self.dataset} --portion_labeled {portion_labeled} --tau {delta_s} --selection_strategy score --teacher_strategy {selection_strategy}_aug --clw {highest_weight} --version_num {version_num} --num_epochs 200\n"
            print(stac_command)

        if visualize:
            self._plot_cls_score(scaled_occ_perc, cls_dist)

    def read_pred_folder(self):
        """Reads the prediction folder and returns a list of files with the extensions '.txt' or '.json'."""
        return [
            file
            for file in os.listdir(self.det_folder)
            if file.endswith(".txt") or file.endswith(".json")
        ]

    def _percls_analysis(self):
        """Perform per-class analysis on the allocated detections."""

        gt_classes = np.concatenate(self.allocated_dets["gt"]["class"])
        det_classes = np.concatenate(self.allocated_dets["pseudo"]["class"])
        ious = np.concatenate(
            [
                calc_iou_np(gt_boxes, pseudo_boxes)
                for gt_boxes, pseudo_boxes in zip(
                    self.allocated_dets["gt"]["box"],
                    self.allocated_dets["pseudo"]["box"],
                )
                if len(gt_boxes) > 0
            ]
        )
        gt_cls_dist = []
        det_cls_dist = []
        perc_miou = []
        perc_acc = []
        for cls in self.used_classes:
            perc_acc.append(
                np.round(
                    np.mean(
                        gt_classes[gt_classes == cls] == det_classes[gt_classes == cls]
                    ),
                    2,
                )
            )
            perc_miou.append(np.mean(ious[gt_classes == cls]))
            gt_cls_dist.append(np.sum(gt_classes == cls))
            det_cls_dist.append(np.sum(det_classes == cls))

        rounded_iou = np.round(perc_miou, 2)
        iou_mapping = dict(zip(self.used_classes, rounded_iou))
        self.print_data += f"mIou: {iou_mapping}\n"
        acc_mapping = dict(zip(self.used_classes, perc_acc))
        self.print_data += f"Acc: {acc_mapping}\n"

        matched_dets_cls_dist = {
            uc: np.sum(det_classes == uc) for uc in self.used_classes
        }
        self.print_data += f"Matched Dets: {matched_dets_cls_dist}\n"
        matched_gts_cls_dist = {
            uc: np.sum(gt_classes == uc) for uc in self.used_classes
        }
        self.print_data += f"Matched GT: {matched_gts_cls_dist}\n"

        extra_dets = [
            [i for i in range(len(pb)) if pb[i] not in ab]
            for pb, ab in zip(
                self.collect_pseudo_boxes, self.allocated_dets["pseudo"]["box"]
            )
        ]
        ed_cls_dist = np.concatenate(
            [
                self.collect_pseudo_classes[i][extra_dets[i]]
                for i in range(len(extra_dets))
            ]
        )
        ed_cls_dist = {uc: np.sum(ed_cls_dist == uc) for uc in self.used_classes}
        self.print_data += f"No Match Dets: {ed_cls_dist}\n"

        nomatch_gts = [
            [
                i
                for i in range(len(pb))
                if not np.any(np.all(np.asarray(pb[i]) == np.asarray(ab), axis=-1))
            ]
            for pb, ab in zip(self.collect_gt_boxes, self.allocated_dets["gt"]["box"])
        ]
        mp_classes_dist = np.concatenate(
            [
                self.collect_gt_classes[i][nomatch_gts[i]]
                for i in range(len(nomatch_gts))
            ]
        )
        mp_classes_dist = {
            uc: np.sum(mp_classes_dist == uc) for uc in self.used_classes
        }
        self.print_data += f"No Match GT: {mp_classes_dist}\n"

    def extract_pseudo_gt_data(self, new_dets=False):
        """
        Comapre predictions to ground truth data.

        Args:
            new_dets (bool, optional): Flag indicating whether new detections are used or the list of BDD detections can be used. Defaults to False.

        """
        self.collect_pseudo_boxes = []
        self.collect_gt_boxes = []
        self.collect_gt_classes = []
        self.collect_pseudo_classes = []

        self.n_gts_perim = []
        self.n_pred_perim = []
        self.n_gt_matches = []
        self.matched_preds = []
        self.nomatch_preds = []
        self.n_extra_detections = []

        self.perim_ious = []
        mious = []
        macc = []

        self.allocated_dets = {
            "gt": {"class": [], "box": []},
            "pseudo": {"class": [], "box": []},
        }
        heatmap_md = np.zeros((1000, 2000))
        heatmap_fd = np.zeros((1000, 2000))
        all_gt_objects = []
        all_det_objects = []
        if self.dataset == "KITTI":
            for image_data in self.images_data:
                all_gt_objects.append(
                    self.read_annotations()(
                        os.path.join(self.gt_labels_folder, image_data),
                        self.used_classes,
                    )
                )
                all_det_objects.append(
                    self.read_annotations()(
                        os.path.join(self.det_folder, image_data), self.used_classes
                    )
                )
        else:
            if (getattr(self, "bdd_pseudo_data", None) is None) or new_dets:
                with open(
                    os.path.join(self.det_folder, self.images_data[0]), "r"
                ) as file:
                    self.bdd_pseudo_data = json.load(file)  # Load the entire JSON data

            self.images_data = [nn["name"] for nn in self.bdd_pseudo_data]
            for im_name in self.images_data:
                all_gt_objects.append(
                    self.read_annotations()(
                        os.path.join(self.gt_labels_folder, im_name), self.used_classes
                    )
                )
                all_det_objects.append(
                    self.read_annotations()(
                        os.path.join(self.det_folder, im_name),
                        self.used_classes,
                        gt=False,
                        given_data=self.bdd_pseudo_data,
                    )
                )

        for image_data in range(len(all_gt_objects)):
            gt_objects = all_gt_objects[image_data]
            det_objects = all_det_objects[image_data]

            self.n_gts_perim.append(len(gt_objects))
            self.n_pred_perim.append(len(det_objects))

            gt_boxes = [gt["bbox"] for gt in gt_objects]
            self.collect_gt_boxes.append(gt_boxes)
            det_boxes = [det["bbox"] for det in det_objects]
            self.collect_pseudo_boxes.append(det_boxes)
            gt_classes = np.asarray([gt["class"] for gt in gt_objects])
            self.collect_gt_classes.append(gt_classes)
            det_classes = np.asarray([det["class"] for det in det_objects])
            self.collect_pseudo_classes.append(det_classes)

            ious = [calc_iou_np([gt], det_boxes) for gt in gt_boxes]
            self.perim_ious.append(ious)
            ious = np.asarray(ious)

            matched_pred = np.unique(
                np.argmax(ious, axis=-1)[np.max(ious, axis=-1) >= self.gt_iou_thr]
            )  # A detection might match multiple GTs, find the one it matches most
            self.matched_preds.append(matched_pred)
            self.nomatch_preds.append(
                np.setdiff1d(np.arange(len(det_objects)), matched_pred)
            )  # Extra detections that are not allocated to a GT

            matched_gt = []
            for i in matched_pred:
                idx = np.argmax(
                    ious[:, i]
                )  # Ensure the index hasn't been selected already
                while idx in matched_gt:
                    ious[idx, i] = -1  # Invalidate this choice
                    idx = np.argmax(ious[:, i])
                matched_gt.append(idx)

            macc.append(
                [
                    gt_classes[matched_gt[i]]
                    == det_classes[
                        matched_pred[i]
                    ]  # check if the class of the found gt and the matched pred match
                    for i in range(len(matched_pred))
                ]
            )
            mious.append(
                [np.max(ious[i]) for i in matched_gt]
            )  # Calculate the iou per match det

            n_matches = len(matched_pred)
            self.n_gt_matches.append(
                n_matches
            )  # Find unique detections which cover GT, one pred cannot cover multiple GTs
            self.n_extra_detections.append(len(det_objects) - n_matches)

            self.allocated_dets["gt"]["class"].append(
                [gt_classes[matched_gt[i]] for i in range(len(matched_gt))]
            )
            self.allocated_dets["gt"]["box"].append(
                [gt_boxes[matched_gt[i]] for i in range(len(matched_gt))]
            )
            self.allocated_dets["pseudo"]["class"].append(
                [det_classes[i] for i in matched_pred]
            )
            self.allocated_dets["pseudo"]["box"].append(
                [det_boxes[i] for i in matched_pred]
            )

            for xmin, ymin, xmax, ymax in [
                gt_boxes[i]
                for i in [j for j in range(len(gt_boxes)) if j not in matched_gt]
            ]:
                heatmap_md[int(ymin) : int(ymax), int(xmin) : int(xmax)] += 1
            for xmin, ymin, xmax, ymax in [
                det_boxes[i]
                for i in [j for j in range(len(det_boxes)) if j not in matched_pred]
            ]:
                heatmap_fd[int(ymin) : int(ymax), int(xmin) : int(xmax)] += 1

        self.n_missing_dets = (
            np.asarray(self.n_gts_perim) - np.asarray(self.n_gt_matches)
        ) / np.asarray(self.n_gts_perim)
        matched_gts_percent = np.round(
            np.sum(self.n_gt_matches) / np.sum(self.n_gts_perim) * 100, 2
        )
        extra_detections_percent = np.round(
            np.sum(np.sum(self.n_extra_detections)) / np.sum(self.n_pred_perim) * 100, 2
        )
        m_macc = np.round(
            np.sum(np.concatenate(macc)) / len(np.concatenate(macc)) * 100, 2
        )
        m_miou = np.round(np.mean(np.concatenate(mious)) * 100, 2)
        self.print_data = f"number of gts: {np.sum(self.n_gts_perim)}\nnumber of preds: {np.sum(self.n_pred_perim)}\nfound gts: {np.sum(self.n_gt_matches)}, {matched_gts_percent}(%), missing {np.round(100-matched_gts_percent,2)}(%)\nextra possibly false preds: {np.sum(self.n_extra_detections)}, {extra_detections_percent}(%)\nmAcc on found dets: {m_macc}%\nmIoU on found dets: {m_miou}%"
        self.print_data += "\n"
        self._percls_analysis()
        print(self.print_data)
