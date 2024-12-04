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


class RCF(Parent_SSL):
    """Generates training strategy by splitting dataset into hard and normal examples based on class distribution"""

    def __init__(
        self,
        added_rcc_name="rcf",
        version_num=0,
        lowest_weight=1,
        highest_weight=10,
        delta_s=0.4,
        *args,
        **kwargs,
    ):
        """
        Initializes the RCF class.
        Parameters:
          added_rcc_name (str): The name added to saving paths.
          version_num (int): The version number of the model.
          lowest_weight (int): The lowest value for class weight.
          highest_weight (int): The highest value for class weight.
          delta_s (float): The delta_s score threshold value.
          *args: Variable length argument list. Belongs to parent class.
          **kwargs: Arbitrary keyword arguments. Belongs to parent class.
        """

        super().__init__(*args, **kwargs)  # Call parent constructor
        self.weight_images_cls_dist(
            added_name=added_rcc_name,
            rcf=True,
            lowest_weight=lowest_weight,
            highest_weight=highest_weight,
            version_num=version_num,
            delta_s=delta_s,
        )


# Create an instance of the child class
RCF(
    dataset="KITTI",
    labeled_indices_path="num_labeled_10/V0/_train_init_V0.txt",
    added_name="num_labeled_10",
    # dataset="BDD100K",
    # labeled_indices_path="num_labeled_1/V0/_train_init_V0.txt",
    # added_name="num_labeled_1",
    added_rcc_name="rcf",
    version_num=0,
    lowest_weight=1,
    highest_weight=30,
    delta_s=0.4,
)
