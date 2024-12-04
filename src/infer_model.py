# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Inference with a model """


import glob
import math
import os
import shutil
from shutil import copyfile

import cv2
import numpy as np
import tensorflow as tf
from add_corruption import add_weather, apply_corruption
from PIL import Image
from uncertainty_analysis import MainUncertViz
from utils_box import CalibrateBoxUncert, calc_iou_np, relativize_uncert
from utils_class import CalibrateClass, stable_softmax
from utils_extra import add_array_dict, save_uncert


class InferImages:
    """Perform inference on images"""

    def __init__(
        self,
        infer_data,
        driver,
        model_params,
        model_name,
        mode="inference",
        added_name="",
        general_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ):
        """Constructs all the necessary attributes for the inference

        Args:
            infer_data (dict): Inference config file content
            driver (object): Driver to serve the images to the model
            model_params (dict): Model parameters
            model_name (str): Path containing model name
            mode (string): Activates the difference inference modes such as Auto-Labeling or SSAL
            added_name (string): Additional name for the SSAL mode for saving the results
            general_path (str): Path to working space
        """
        self.ssl_score = True if "SSL" in infer_data["saved_model_dir"] else False
        self.consistency_ssl = model_params["consistency_ssl"]
        self.general_path = general_path
        self.saving_path = general_path + "/results/inference/"
        self.model_name = model_name.split("/")[-1]
        self.infer = True
        self.auto_labeling = False
        self.ssal = False
        if mode == "auto-label":
            self.auto_labeling = True
            self.infer = False
            self.count_auto = 0
            self.count_skip = 0
            activate_calib = (
                "calib/"
                if (
                    model_params["calibrate_regression"]
                    and model_params["calibrate_classification"]
                )
                else "orig/"
            )
            opt_params_path = (
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                + "/results/validation/"
                + self.model_name
                + "/thresholding/"
                + activate_calib
            )
            fpr_tpr = model_params["thr_fpr_tpr"]
            fix_cd = "cd" if model_params["thr_cd"] else "fd"
            ious_thrs = model_params["thr_iou_thrs"]
            if os.path.exists(
                opt_params_path
                + "/optimal_params_"
                + fix_cd
                + "_"
                + str(fpr_tpr)
                + "_iou_"
                + str(np.min(ious_thrs))
                + "_"
                + str(np.max(ious_thrs))
                + ".txt"
            ):
                with open(
                    opt_params_path
                    + "/optimal_params_"
                    + fix_cd
                    + "_"
                    + str(fpr_tpr)
                    + "_iou_"
                    + str(np.min(ious_thrs))
                    + "_"
                    + str(np.max(ious_thrs))
                    + ".txt",
                    "r",
                ) as file:
                    self.opt_params = [
                        float(x.strip("[]")) for x in file.read().split()
                    ]
                with open(
                    opt_params_path
                    + "/optimal_thrs_"
                    + fix_cd
                    + "_"
                    + str(fpr_tpr)
                    + "_iou_"
                    + str(np.min(ious_thrs))
                    + "_"
                    + str(np.max(ious_thrs))
                    + ".txt",
                    "r",
                ) as file:
                    self.opt_thrs = [float(x.strip("[]")) for x in file.read().split()]
            else:
                if os.path.exists(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    + "/results/validation/"
                    + self.model_name
                ):
                    if "calib" in activate_calib:
                        MainUncertViz(self.model_name)
                    else:
                        MainUncertViz(self.model_name, calib=False)
                    with open(
                        opt_params_path
                        + "/optimal_params_"
                        + fix_cd
                        + "_"
                        + str(fpr_tpr)
                        + "_iou_"
                        + str(np.min(ious_thrs))
                        + "_"
                        + str(np.max(ious_thrs))
                        + ".txt",
                        "r",
                    ) as file:
                        self.opt_params = [
                            float(x.strip("[]")) for x in file.read().split()
                        ]
                    with open(
                        opt_params_path
                        + "/optimal_thrs_"
                        + fix_cd
                        + "_"
                        + str(fpr_tpr)
                        + "_iou_"
                        + str(np.min(ious_thrs))
                        + "_"
                        + str(np.max(ious_thrs))
                        + ".txt",
                        "r",
                    ) as file:
                        self.opt_thrs = [
                            float(x.strip("[]")) for x in file.read().split()
                        ]
                else:
                    print(
                        "Run validation first, to determine optimal threshold based on it!"
                    )
            self.save_dir = (
                self.saving_path
                + "auto_labeling/"
                + self.model_name
                + "/"
                + fix_cd
                + "/"
                + str(fpr_tpr)
                + "/"
                + "iou_"
                + str(np.min(ious_thrs))
                + "_"
                + str(np.max(ious_thrs))
                + "/"
            )
            if not os.path.exists(self.save_dir + "/labeled"):
                os.makedirs(self.save_dir + "/labeled")
                os.makedirs(self.save_dir + "/examine")
        elif mode == "SSAL":
            self.ssal = True
            self.infer = False
            if added_name != "":
                added_name += "_"
            self.save_dir = self.saving_path + "/SSAL/" + added_name + self.model_name
            if os.path.exists(self.save_dir):
                shutil.rmtree(self.save_dir)
        else:
            self.save_dir = self.saving_path + "images/" + self.model_name
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.model_params = model_params
        self.driver = driver
        if mode == "SSAL":
            imnames = self._get_infer_images(infer_data, indices=True)
        else:
            imnames = self._get_infer_images(infer_data)
        self._read_infer_images(imnames)
        self.len_images = len(imnames)
        # Create folder for saving max uncertainty per image
        if (
            self.model_params["loss_attenuation"] or self.model_params["mc_dropout"]
        ) and mode == "inference":
            self.uncert_path = self.saving_path + "uncertainty/" + self.model_name
            if os.path.isdir(
                self.uncert_path
            ):  # Necessary so that the results are not added to each other and mix up happens
                shutil.rmtree(self.uncert_path)
            os.makedirs(self.uncert_path)

    @staticmethod
    def _get_infer_images(infer_data, indices=False):
        """Extract the image names based on the inference config file

        Args:
            infer_data (dict): Inference config file content

        Returns:
            The image names for inference
        """
        input_images_names = []
        imnames = sorted(glob.glob(infer_data["infer_folder"] + "*"))
        if indices:
            if "infer_indices" in infer_data:
                with open(infer_data["infer_indices"], "r") as file:
                    for line in file:
                        input_images_names.append(imnames[int(line.strip())])
            else:
                input_images_names = [
                    im for im in imnames if "jpg" in im or "png" in im
                ]
        else:
            for i in range(
                np.min([infer_data["infer_first_frame"], len(imnames) - 1]),
                np.min([infer_data["infer_last_frame"], len(imnames) - 1]),
            ):
                input_images_names.append(imnames[i])
        return input_images_names

    def _augment_inference_image(self, image_name, image):
        """Augment inference image

        Args:
            image_name (str): Image name
            image (tensor): Image for inference
        """
        modes = self.model_params["infer_augment"]
        aug_names = []
        aug_imgs = []
        if "heq" in modes:
            img_yuv = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            aug_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            aug_names.append(image_name[:-4] + "_heq" + image_name[-4:])
            aug_imgs.append(aug_image)
        if "alb" in modes:
            aug_image = add_weather(image, "snow")
            aug_names.append(image_name[:-4] + "_snow" + image_name[-4:])
            aug_imgs.append(aug_image)

            aug_image = add_weather(image, "fog")
            aug_names.append(image_name[:-4] + "_fog" + image_name[-4:])
            aug_imgs.append(aug_image)

            aug_image = add_weather(image, "rain")
            aug_names.append(image_name[:-4] + "_rain" + image_name[-4:])
            aug_imgs.append(aug_image)

            aug_image = add_weather(image, "noise")
            aug_names.append(image_name[:-4] + "_noise" + image_name[-4:])
            aug_imgs.append(aug_image)
        if "aug" in modes:
            augims = apply_corruption("ns", np.asarray(image))
            for ind_im in range(len(augims)):
                aug_names.append(
                    image_name[:-4] + "_ns" + str(ind_im) + image_name[-4:]
                )
                aug_imgs.append(augims[ind_im])

            augims = apply_corruption("mb", np.asarray(image))
            for ind_im in range(len(augims)):
                aug_names.append(
                    image_name[:-4] + "_mb" + str(ind_im) + image_name[-4:]
                )
                aug_imgs.append(augims[ind_im])

            augims = apply_corruption("ct", np.asarray(image))
            for ind_im in range(len(augims)):
                aug_names.append(
                    image_name[:-4] + "_ct" + str(ind_im) + image_name[-4:]
                )
                aug_imgs.append(augims[ind_im])

            augims = apply_corruption("br", np.asarray(image))
            for ind_im in range(len(augims)):
                aug_names.append(
                    image_name[:-4] + "_br" + str(ind_im) + image_name[-4:]
                )
                aug_imgs.append(augims[ind_im])
        if "flip" in modes:
            aug_image = cv2.flip(np.asarray(image), 0)  # vertical
            aug_names.append(image_name[:-4] + "_vflip" + image_name[-4:])
            aug_imgs.append(aug_image)

            aug_image = cv2.flip(np.asarray(image), 1)  # horizontal
            aug_names.append(image_name[:-4] + "_hflip" + image_name[-4:])
            aug_imgs.append(aug_image)

        aug_names = [tf.convert_to_tensor(img_name) for img_name in aug_names]
        aug_imgs = [tf.convert_to_tensor(img) for img in aug_imgs]
        return aug_names, aug_imgs

    def _read_infer_images(self, input_images_names):
        """Read inference images from list of names and apply corruptions if applicable

        Args:
            input_images_names (list): Images names for inference
        """

        def __load_and_augment_image(file_path):
            def ___aug_numpy(file_path, original_image):
                collect_names = [file_path]
                collect_ims = [original_image]
                aug_names, aug_imgs = self._augment_inference_image(
                    file_path.numpy().decode("utf-8"), original_image.numpy()
                )
                collect_names += aug_names
                collect_ims += aug_imgs
                return collect_names, collect_ims

            original_image = tf.io.decode_image(
                tf.io.read_file(file_path), channels=3, expand_animations=False
            )
            if self.model_params["infer_augment"]:
                collect_names, collect_ims = tf.py_function(
                    func=___aug_numpy,
                    inp=[file_path, original_image],
                    Tout=[tf.string, tf.uint8],
                )
            else:
                collect_ims = [original_image]
                collect_names = [file_path]
            return collect_names, collect_ims

        dataset = tf.data.Dataset.from_tensor_slices(input_images_names)
        self.imgs_names_data = dataset.flat_map(
            lambda x: tf.data.Dataset.from_tensor_slices(__load_and_augment_image(x))
        )

    def _compare_highlow_epal(self):
        """For box uncertainty, save a split % of the highest epistemic/lowest aleatoric and vice versa in a separate folder"""

        if not os.path.exists(self.uncert_path + "/lowal_highep/"):
            os.mkdir(self.uncert_path + "/lowal_highep/")

        if not os.path.exists(self.uncert_path + "/highal_lowep/"):
            os.mkdir(self.uncert_path + "/highal_lowep/")

        # Read max aleatoric
        f = open(self.uncert_path + "/uncert_albox.txt", "r")
        mx_unc = f.readlines()
        if len(mx_unc) != self.len_images:
            print(
                "Mismatch saved images: "
                + str(len(mx_unc))
                + " "
                + str(self.len_images)
            )
        imgs_sorted = []
        uncert_vals = []
        for i in range(len(mx_unc)):
            val = mx_unc[i].split("\n")[0]
            if val != "nan":
                imgs_sorted.append(self.current_image)
                uncert_vals.append(val)
        al_images = [x for _, x in sorted(zip(uncert_vals, imgs_sorted))]
        al_unc = sorted(uncert_vals)

        # Read max epistemic
        f = open(self.uncert_path + "/uncert_mcbox.txt", "r")
        mx_unc = f.readlines()
        if len(mx_unc) != self.len_images:
            print(
                "mismatch saved images: "
                + str(len(mx_unc))
                + " "
                + str(self.len_images)
            )
        imgs_sorted = []
        uncert_vals = []
        for i in range(len(mx_unc)):
            val = mx_unc[i].split("\n")[0]
            if val != "nan":
                imgs_sorted.append(self.current_image)
                uncert_vals.append(val)
        mc_images = [x for _, x in sorted(zip(uncert_vals, imgs_sorted))]
        mc_unc = sorted(uncert_vals)

        mc_unc = np.asarray(mc_unc).astype(float)
        al_unc = np.asarray(al_unc).astype(float)

        # Normalize uncertainty
        al_unc = al_unc / max(al_unc)
        mc_unc = mc_unc / max(mc_unc)

        # Sort one array with the other with images in common
        sort_ind = [mc_images.index(al_images[i]) for i in range(len(al_images))]
        sorted_al_unc = al_unc[sort_ind]

        split = math.ceil(0.1 * len(imgs_sorted))
        if (2 * split) < len(imgs_sorted):

            hal_lep = np.asarray(mc_images)[(mc_unc - sorted_al_unc).argsort()[:split]]
            lal_hep = np.asarray(mc_images)[(mc_unc - sorted_al_unc).argsort()[-split:]]
            with open(
                self.uncert_path + "/highal_lowep/highaleatoric_lowepistemic.txt", "w"
            ) as f:
                f.write(
                    "Image name, difference, aleatoric uncertainty, epistemic uncertainty\n"
                )
                for itm in list(
                    zip(
                        hal_lep,
                        sorted(mc_unc - sorted_al_unc)[:split],
                        sorted_al_unc[(mc_unc - sorted_al_unc).argsort()[:split]],
                        mc_unc[(mc_unc - sorted_al_unc).argsort()[:split]],
                    )
                ):
                    f.write(str(itm) + "\n")
            with open(
                self.uncert_path + "/lowal_highep/lowaleatoric_highepistemic.txt", "w"
            ) as f:
                f.write(
                    "Image name, difference, aleatoric uncertainty, epistemic uncertainty\n"
                )
                for itm in list(
                    zip(
                        lal_hep,
                        sorted(mc_unc - sorted_al_unc)[-split:],
                        sorted_al_unc[(mc_unc - sorted_al_unc).argsort()[-split:]],
                        mc_unc[(mc_unc - sorted_al_unc).argsort()[-split:]],
                    )
                ):
                    f.write(str(itm) + "\n")

            for i in range(split):
                copyfile(
                    self.save_dir
                    + "/"
                    + lal_hep[i].split("/")[-1].split(".")[0]
                    + ".png",
                    self.uncert_path
                    + "/lowal_highep/"
                    + lal_hep[i].split("/")[-1].split(".")[0]
                    + ".png",
                )
                copyfile(
                    self.save_dir
                    + "/"
                    + hal_lep[i].split("/")[-1].split(".")[0]
                    + ".png",
                    self.uncert_path
                    + "/highal_lowep/"
                    + hal_lep[i].split("/")[-1].split(".")[0]
                    + ".png",
                )
        else:
            print("Not enough images to compare " + str(2 * split) + " of")

    def _sort_maxuncert(self, uncert_name):
        """Sort saved image names based on uncertainty in the saved file

        Args:
            uncert_name (str): Name of the uncertainty; albox, mcbox, mcclass ..
        """

        f = open(self.uncert_path + "/uncert" + uncert_name + ".txt", "r")
        uncert = f.readlines()
        if len(uncert) != self.len_images:
            print(
                "Mismatch saved images: "
                + str(len(uncert))
                + " "
                + str(self.len_images)
            )
        imgs_sorted = []
        uncert_vals = []
        for i in range(len(uncert)):
            val = uncert[i].split("\n")[0]
            if val != "nan":
                imgs_sorted.append(self.current_image)
                uncert_vals.append(val)
        imgs_sorted = [x for _, x in sorted(zip(uncert_vals, imgs_sorted))]
        uncert_vals = sorted(uncert_vals)

        with open(self.uncert_path + "/uncert" + uncert_name + ".txt", "w") as f:
            for itm in list(zip(imgs_sorted, uncert_vals)):
                f.write(str(itm) + "\n")
        return imgs_sorted

    def _collect_highlow_uncert(self, imgs_sorted, uncert_name):
        """Save top and bottom x% of images based on their uncertainty

        Args:
            imgs_sorted (list): Sorted images names
            uncert_name (str): Name of the uncertainty; albox, mcbox, mcclass ..
        """
        if not os.path.exists(self.uncert_path + "/lower_uncert/"):
            os.mkdir(self.uncert_path + "/lower_uncert/")

        if not os.path.exists(self.uncert_path + "/upper_uncert/"):
            os.mkdir(self.uncert_path + "/upper_uncert/")

        if not os.path.exists(self.uncert_path + "/lower_uncert/" + uncert_name):
            os.mkdir(self.uncert_path + "/lower_uncert/" + uncert_name)

        if not os.path.exists(self.uncert_path + "/upper_uncert/" + uncert_name):
            os.mkdir(self.uncert_path + "/upper_uncert/" + uncert_name)

        split = math.ceil(0.1 * len(imgs_sorted))
        if (2 * split) < len(imgs_sorted):
            for i in range(split):
                copyfile(
                    self.save_dir
                    + "/"
                    + imgs_sorted[i].split("/")[-1].split(".")[0]
                    + ".png",
                    self.uncert_path
                    + "/lower_uncert/"
                    + uncert_name
                    + imgs_sorted[i].split("/")[-1].split(".")[0]
                    + ".png",
                )
                copyfile(
                    self.save_dir
                    + "/"
                    + imgs_sorted[-i - 1].split("/")[-1].split(".")[0]
                    + ".png",
                    self.uncert_path
                    + "/upper_uncert/"
                    + uncert_name
                    + imgs_sorted[-i - 1].split("/")[-1].split(".")[0]
                    + ".png",
                )
        else:
            print("Not enough images to compare " + str(2 * split) + " of")

    def iterate_infer(self):
        """Main function to feed the model during inference and collect results"""
        val_file_path = (
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            + "/results/validation/"
            + self.model_name
            + "/average_score.txt"
        )
        if os.path.exists(val_file_path):
            with open(val_file_path, "r") as file:
                average_score = file.read()
            average_score = float(average_score[:3])
        else:
            average_score = 0
        if self.ssl_score:
            min_score = 0.1
        else:
            min_score = (
                self.model_params["nms_configs"]["score_thresh"] or average_score or 0.4
            )

        i = 0
        for im_name, im in self.imgs_names_data:
            self.current_image = im_name.numpy().decode("utf-8")
            im = tf.expand_dims(im, axis=0)
            img_name = self.current_image.split("/")[-1][:-4]
            print(img_name)
            detections_bs = self.driver.serve(im)
            # tf.keras.backend.clear_session()

            # If logits available
            if self.model_params["enable_softmax"]:
                boxes, scores, classes, _, logits = tf.nest.map_structure(
                    np.array, detections_bs
                )
                probab_logits = [stable_softmax(logits[0])]
                entropy = -np.sum(
                    probab_logits[0]
                    * np.nan_to_num(np.log2(np.maximum(probab_logits[0], 10**-7))),
                    axis=1,
                )
                select_entropy = entropy  # Select entropy to be saved
            else:
                boxes, scores, classes, _ = tf.nest.map_structure(
                    np.array, detections_bs
                )
                logits = [None]

            # If uncertainty estimation available
            if (
                self.model_params["mc_boxheadrate"]
                or self.model_params["mc_dropoutrate"]
            ) and not self.model_params["loss_attenuation"]:
                mc_boxhead = np.nan_to_num(boxes[:, :, 4:])
                al_boxhead = None
            elif (
                self.model_params["mc_boxheadrate"]
                or self.model_params["mc_dropoutrate"]
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
            if (
                self.model_params["mc_classheadrate"]
                or self.model_params["mc_dropoutrate"]
            ):
                mc_classhead = np.nan_to_num(classes[:, :, 1:])
                classes = classes[:, :, 0]
            else:
                mc_classhead = None
            if mc_boxhead is not None or al_boxhead is not None:
                boxes = boxes[:, :, :4]

            if al_boxhead is not None:
                select_albox = al_boxhead[
                    0
                ]  # Select uncertainty to be saved for analysis
            if mc_boxhead is not None:
                select_mcbox = mc_boxhead[
                    0
                ]  # Select uncertainty to be saved for analysis
            if mc_classhead is not None:
                select_mcclass = mc_classhead[
                    0
                ]  # Select uncertainty to be saved for analysis

            # Calibrate regression uncertainty
            if self.model_params["calibrate_regression"]:
                if mc_boxhead is not None:
                    calibrated_mcbox = CalibrateBoxUncert(
                        self.model_params,
                        self.model_name + "/regression/mcdropout/",
                        general_path=self.general_path,
                    ).calibrate_boxuncert(mc_boxhead[0], classes[0], boxes[0])
                    (
                        temp_select_mcbox,
                        iso_all_mcbox,
                        ts_all_mcbox,
                        ts_percoo_mcbox,
                        iso_percoo_mcbox,
                        iso_perclscoo_mcbox,
                        rel_iso_perclscoo_mcbox,
                    ) = calibrated_mcbox
                    if temp_select_mcbox.size > 0:
                        select_mcbox = temp_select_mcbox
                if al_boxhead is not None:
                    calibrated_albox = CalibrateBoxUncert(
                        self.model_params,
                        self.model_name + "/regression/aleatoric/",
                        general_path=self.general_path,
                    ).calibrate_boxuncert(al_boxhead[0], classes[0], boxes[0])
                    (
                        temp_select_albox,
                        iso_all_albox,
                        ts_all_albox,
                        ts_percoo_albox,
                        iso_percoo_albox,
                        iso_perclscoo_albox,
                        rel_iso_perclscoo_albox,
                    ) = calibrated_albox
                    if temp_select_albox.size > 0:
                        select_albox = temp_select_albox

            if mc_boxhead is not None:
                relative_select_mcbox = relativize_uncert(boxes[0], select_mcbox[0])
            if al_boxhead is not None:
                relative_select_al = relativize_uncert(boxes[0], select_albox[0])

            # Calibrate classification uncertainty
            if self.model_params["calibrate_classification"]:
                if mc_classhead is not None:
                    calib_class = CalibrateClass(
                        logits[0],
                        self.model_name,
                        self.model_params["calib_method_class"],
                        mc_classhead[0],
                        general_path=self.general_path,
                    ).calibrate_class()
                    (
                        temp_select_mcclass,
                        temp_select_entropy,
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
                    ) = calib_class
                    if temp_select_mcclass.size > 0:
                        temp_select_mcclass = select_mcclass
                else:
                    calib_class = CalibrateClass(
                        logits[0],
                        self.model_name,
                        self.model_params["calib_method_class"],
                        general_path=self.general_path,
                    ).calibrate_class()
                    (
                        temp_select_entropy,
                        ts_all_probab,
                        ts_all_entropy,
                        ts_perc_probab,
                        ts_perc_entropy,
                        iso_all_probab,
                        iso_all_entropy,
                        iso_perc_probab,
                        iso_perc_entropy,
                    ) = calib_class
                if temp_select_entropy.size > 0:
                    select_entropy = temp_select_entropy

            if self.auto_labeling:  # Check optimal uncertainty below threshold
                thr_sel_uncert = self.model_params["thr_sel_uncert"]
                thr_uncerts = []
                if "ENT" in thr_sel_uncert:
                    thr_uncerts.append(select_entropy)
                if "ALBOX" in thr_sel_uncert:
                    thr_uncerts.append(np.mean(relative_select_al, axis=-1))
                opt_uncert = sum(
                    opt_param * uncert
                    for opt_param, uncert in zip(self.opt_params, thr_uncerts)
                )
                if np.all(
                    opt_uncert[scores[0] > min_score] < np.mean(self.opt_thrs)
                ):  # Select threshold optimized for 0.5
                    output_image_path = os.path.join(
                        self.save_dir, "labeled", img_name + ".png"
                    )
                    self.count_auto += 1
                else:
                    output_image_path = os.path.join(
                        self.save_dir, "examine", img_name + ".png"
                    )
                    self.count_skip += 1
            else:
                output_image_path = os.path.join(self.save_dir, img_name + ".png")

            if self.consistency_ssl:

                def ious_post_augment(aug_mode):
                    # Augment inference image and check iou consistency and class agreement
                    if aug_mode == "flip":
                        aug_im = tf.expand_dims(tf.image.flip_left_right(im[0]), axis=0)
                    elif aug_mode == "blur":
                        aug_im = tf.expand_dims(
                            cv2.GaussianBlur(np.asarray(im[0]), (9, 9), 0), axis=0
                        )
                    elif aug_mode == "noise":
                        row, col, ch = im[0].shape
                        mean = 0
                        var = 0.5
                        sigma = var**0.5
                        gauss = np.random.normal(mean, sigma, (row, col, ch))
                        gauss = gauss.reshape(row, col, ch)
                        noisy = im[0] + gauss
                        aug_im = tf.expand_dims(noisy, axis=0)
                    detections_bs_aug = self.driver.serve(aug_im)
                    if self.model_params["enable_softmax"]:
                        boxes_aug, _, classes_aug, _, _ = tf.nest.map_structure(
                            np.array, detections_bs_aug
                        )
                    else:
                        boxes_aug, _, classes_aug, _ = tf.nest.map_structure(
                            np.array, detections_bs_aug
                        )
                    if (
                        self.model_params["mc_classheadrate"]
                        or self.model_params["mc_dropoutrate"]
                    ):
                        classes_aug = classes_aug[:, :, 0]
                    if mc_boxhead is not None or al_boxhead is not None:
                        boxes_aug = boxes_aug[:, :, :4]
                    if aug_mode == "flip":
                        w = im.shape[2]
                        boxes_aug[0] = [
                            [b[0], w - b[3], b[2], w - b[1]] for b in boxes_aug[0]
                        ]
                    ious = np.asarray(
                        [calc_iou_np([b], boxes_aug[0]) for b in boxes[0]]
                    )
                    # from matplotlib import pyplot as plt
                    # plt.figure(figsize=(20,20))
                    # plt.imshow(aug_im[0])
                    # plt.savefig("augmented_image.png")
                    # plt.close()
                    return np.max(ious, axis=-1), classes_aug

                flip_iou, cls_flip = ious_post_augment("flip")
                blur_iou, cls_blur = ious_post_augment("blur")
                noise_iou, cls_noise = ious_post_augment("noise")

                ious_aug = np.mean(
                    np.stack([flip_iou, blur_iou, noise_iou], axis=-1), axis=-1
                )
                class_common = [
                    num.is_integer()
                    for num in np.mean(
                        np.stack([cls_flip, cls_blur, cls_noise], axis=-1)[0], axis=-1
                    )
                ]
            filtered_max_albox = []
            filtered_max_mcbox = []
            filtered_mcclass = []
            filtered_entropy = []

            uncert_data = {}
            uncert_data["image_name"] = img_name + ".jpg"
            uncert_data["score_thresh"] = min_score
            uncert_data["top_5scores"] = list(scores[0][:5])

            # Save prediction data
            for sel in np.where(scores[0] > min_score)[0]:  # Filter above score
                uncert_data["det_score"] = scores[0][sel]
                uncert_data["bbox"] = list(boxes[0][sel])
                uncert_data["class"] = classes[0][sel]
                if self.consistency_ssl:
                    uncert_data["cons_iou"] = ious_aug[sel]
                    uncert_data["cons_cls"] = class_common[sel]
                if self.model_params["enable_softmax"]:
                    uncert_data = add_array_dict(uncert_data, logits[0], "logits", sel)
                    uncert_data = add_array_dict(uncert_data, entropy, "entropy", sel)
                    uncert_data["probab"] = list(probab_logits[0][sel])
                    filtered_entropy.append(
                        np.around(select_entropy[sel].astype(np.float64), 6)
                    )
                    if self.model_params["calibrate_classification"]:
                        uncert_data = add_array_dict(
                            uncert_data, ts_all_probab, "ts_all_probab", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data, ts_perc_probab, "ts_percls_probab", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data, iso_all_probab, "iso_all_probab", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data, iso_perc_probab, "iso_percls_probab", sel
                        )

                        uncert_data = add_array_dict(
                            uncert_data, ts_all_entropy, "ts_all_entropy", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data, ts_perc_entropy, "ts_percls_entropy", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data, iso_all_entropy, "iso_all_entropy", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data, iso_perc_entropy, "iso_percls_entropy", sel
                        )
                if mc_classhead is not None:
                    uncert_data = add_array_dict(
                        uncert_data, mc_classhead[0], "uncalib_mcclass", sel
                    )
                    filtered_mcclass.append(
                        select_mcclass[sel][int(classes[0][sel] - 1)].astype(np.float64)
                    )
                    if self.model_params["calibrate_classification"]:
                        uncert_data = add_array_dict(
                            uncert_data, ts_all_mcclass, "ts_all_mcclass", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data, ts_perc_mcclass, "ts_percls_mcclass", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data, iso_all_mcclass, "iso_all_mcclass", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data, iso_perc_mcclass, "iso_percls_mcclass", sel
                        )
                if al_boxhead is not None:
                    uncert_data = add_array_dict(
                        uncert_data, al_boxhead[0], "uncalib_albox", sel
                    )
                    filtered_max_albox.append(np.max(relative_select_al[sel]))
                    if self.model_params["calibrate_regression"]:
                        uncert_data = add_array_dict(
                            uncert_data, iso_all_albox, "iso_all_albox", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data, ts_all_albox, "ts_all_albox", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data, ts_percoo_albox, "ts_percoo_albox", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data, iso_percoo_albox, "iso_percoo_albox", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data, iso_perclscoo_albox, "iso_perclscoo_albox", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data,
                            rel_iso_perclscoo_albox,
                            "rel_iso_perclscoo_albox",
                            sel,
                        )
                if mc_boxhead is not None:
                    uncert_data = add_array_dict(
                        uncert_data, mc_boxhead[0], "uncalib_mcbox", sel
                    )
                    filtered_max_mcbox.append(np.max(relative_select_mcbox[sel]))
                    if self.model_params["calibrate_regression"]:
                        uncert_data = add_array_dict(
                            uncert_data, iso_all_mcbox, "iso_all_mcbox", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data, ts_all_mcbox, "ts_all_mcbox", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data, ts_percoo_mcbox, "ts_percoo_mcbox", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data, iso_percoo_mcbox, "iso_percoo_mcbox", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data, iso_perclscoo_mcbox, "iso_perclscoo_mcbox", sel
                        )
                        uncert_data = add_array_dict(
                            uncert_data,
                            rel_iso_perclscoo_mcbox,
                            "rel_iso_perclscoo_mcbox",
                            sel,
                        )
                if not self.ssal:
                    with open(output_image_path[:-3] + "txt", "a") as f:
                        f.write(str(uncert_data) + "\n")
                with open(self.save_dir + "/prediction_data.txt", "a") as f:
                    f.write(str(uncert_data) + "\n")

            # Save uncertainty per image
            if self.infer and (
                mc_boxhead is not None
                or al_boxhead is not None
                or mc_classhead is not None
            ):
                if al_boxhead is not None:
                    save_uncert(self.uncert_path, filtered_max_albox, "_albox")

                    img = self.driver.visualize(
                        np.array(im)[0],
                        boxes[0],
                        classes[0],
                        scores[0],
                        uncertainty=np.mean(relative_select_al, axis=-1),
                        min_score_thresh=min_score,
                        max_boxes_to_draw=self.model_params["nms_configs"][
                            "max_output_size"
                        ],
                    )
                    if len(np.where(scores[0] > min_score)[0]) != 0:
                        print(
                            "Writing file to %s" % output_image_path.split(".png")[0]
                            + "_mean_albox.png"
                        )
                        Image.fromarray(img).save(
                            output_image_path.split(".png")[0] + "_mean_albox.png",
                            dpi=(1000, 1000),
                        )

                if mc_boxhead is not None:
                    save_uncert(self.uncert_path, filtered_max_mcbox, "_mcbox")

                    img = self.driver.visualize(
                        np.array(im)[0],
                        boxes[0],
                        classes[0],
                        scores[0],
                        uncertainty=np.mean(relative_select_mcbox, axis=-1),
                        min_score_thresh=min_score,
                        max_boxes_to_draw=self.model_params["nms_configs"][
                            "max_output_size"
                        ],
                    )
                    if len(np.where(scores[0] > min_score)[0]) != 0:
                        print(
                            "Writing file to %s" % output_image_path.split(".png")[0]
                            + "_mean_epbox.png"
                        )
                        Image.fromarray(img).save(
                            output_image_path.split(".png")[0] + "_mean_epbox.png",
                            dpi=(1000, 1000),
                        )

                    if mc_classhead is not None:
                        save_uncert(self.uncert_path, filtered_mcclass, "_mcclass")

                        img = self.driver.visualize(
                            np.array(im)[0],
                            boxes[0],
                            classes[0],
                            scores[0],
                            uncertainty=np.max(select_mcclass, axis=-1),
                            min_score_thresh=min_score,
                            max_boxes_to_draw=self.model_params["nms_configs"][
                                "max_output_size"
                            ],
                        )
                        if len(np.where(scores[0] > min_score)[0]) != 0:
                            print(
                                "Writing file to %s"
                                % output_image_path.split(".png")[0]
                                + "_max_epcls.png"
                            )
                            Image.fromarray(img).save(
                                output_image_path.split(".png")[0] + "_max_epcls.png",
                                dpi=(1000, 1000),
                            )

                if self.model_params["enable_softmax"]:
                    save_uncert(self.uncert_path, filtered_entropy, "_entropy")

            if not self.ssal:
                img = self.driver.visualize(
                    np.array(im)[0],
                    boxes[0],
                    classes[0],
                    scores[0],
                    min_score_thresh=min_score,
                    max_boxes_to_draw=self.model_params["nms_configs"][
                        "max_output_size"
                    ],
                )

                # Save inference images
                if len(np.where(scores[0] > min_score)[0]) != 0:
                    print("Writing file to %s" % output_image_path)
                    Image.fromarray(img).save(output_image_path, dpi=(1000, 1000))
                else:
                    print("No detections")
            i += 1

        if self.auto_labeling:
            print(
                "Auto-labeled: "
                + str(self.count_auto)
                + " and skipped: "
                + str(self.count_skip)
            )
        elif self.infer:
            if self.model_params["loss_attenuation"] and (
                self.model_params["mc_boxheadrate"]
                or self.model_params["mc_dropoutrate"]
            ):
                self._compare_highlow_epal()

            if self.model_params["loss_attenuation"]:
                self._collect_highlow_uncert(self._sort_maxuncert("_albox"), "albox/")

            if (
                self.model_params["mc_boxheadrate"]
                or self.model_params["mc_dropoutrate"]
            ):
                self._collect_highlow_uncert(self._sort_maxuncert("_mcbox"), "mcbox/")

            if (
                self.model_params["mc_classheadrate"]
                or self.model_params["mc_dropoutrate"]
            ):
                self._collect_highlow_uncert(
                    self._sort_maxuncert("_mcclass"), "mcclass/"
                )

            if self.model_params["enable_softmax"]:
                self._collect_highlow_uncert(
                    self._sort_maxuncert("_entropy"), "entropy/"
                )
