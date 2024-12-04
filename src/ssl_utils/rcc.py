# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================

import os
import sys

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ssl_utils.parent import Parent_SSL


class RCC(Parent_SSL):
    """Generates image collage by cropping and augmenting the dataset based on targeted classes"""

    def __init__(
        self,
        scale=True,
        manual_augmentations=True,
        cb_weight=False,
        full_im=False,
        rand_augment_wholeimage=False,
        low_scale=0.5,
        high_scale=1.0,
        added_rcc_name="rcc",
        version_num=0,
        *args,
        **kwargs,
    ):
        """
        Initializes the RCC class.

        Args:
            scale (bool, optional): Whether to scale the images in the collage into 4 scales. Defaults to True.
            manual_augmentations (bool, optional): Whether to apply manually defined augmentations. Defaults to True.
            cb_weight (bool, optional): Whether to generate the pseudo labels with weights based on class distribution. Defaults to False.
            full_im (bool, optional): Whether to use the full image for resampling. Defaults to False.
            rand_augment_wholeimage (bool, optional): Whether to apply randaug to the whole image before cropping. Defaults to False.
            low_scale (float, optional): The lower scale limit. Defaults to 0.5.
            high_scale (float, optional): The upper scale limit. Defaults to 1.0.
            added_rcc_name (str, optional): The name added to saving paths. Defaults to "rcc".
            version_num (int, optional): The model version number. Defaults to 0.
            *args: Variable length argument list. Belongs to parent class.
            **kwargs: Arbitrary keyword arguments. Belongs to parent class.
        """
        super().__init__(*args, **kwargs)  # Call parent constructor

        if self.dataset == "KITTI":
            target_class = ["Person_sitting", "Tram"]
        else:
            target_class = ["train", "rider", "motorcycle", "bicycle"]
        self.crop_collage(
            [],
            [],
            target_class=target_class,
            save_path=self.general_path
            + "/datasets/"
            + self.dataset
            + "/collage_crops/"
            + self.added_name
            + "/"
            + added_rcc_name
            + "/",
            gt=True,
            scale=scale,
            manual_augmentations=manual_augmentations,
            full_im=full_im,
            rand_augment_wholeimage=rand_augment_wholeimage,
            low_scale=low_scale,
            high_scale=high_scale,
            version_num=version_num,
        )

        if cb_weight:
            self.weight_images_cls_dist(added_name=added_rcc_name)


# Create an instance of the child class
RCC(
    dataset="KITTI",
    labeled_indices_path="num_labeled_10/V0/_train_init_V0.txt",
    added_name="num_labeled_10",
    # dataset="BDD100K",
    # labeled_indices_path="num_labeled_1/V1/_train_init_V1.txt",
    # added_name="num_labeled_1",
    scale=False,
    manual_augmentations=False,
    cb_weight=False,
    full_im=False,
    rand_augment_wholeimage=False,
    low_scale=0.5,
    high_scale=1.0,
    added_rcc_name="rcc",
    version_num=1,
)
