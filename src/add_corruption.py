# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Augments images with different augmentations """

import random

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf


def apply_corruption(mode, image):
    """Adds corruptions to images based on given mode

    Args:
        mode (str): Corruption name
        image (array): Input image

    Returns:
        list: List of augmented images
    """
    if mode == "br":
        x_ax = np.linspace(0, 100, 10)  # Brightness
    elif mode == "ct":
        x_ax = np.linspace(0, 1, 10)  # Contrast
    elif mode == "bl":
        x_ax = np.linspace(1, 50, 20, dtype=int)
        x_ax = x_ax[x_ax % 2 == 1]  # Gaussian blurr
    elif mode == "ns":
        x_ax = np.linspace(0, 1000, 10)  # Noise
    elif mode == "nsp":
        x_ax = np.linspace(0, 1000, 10)  # Noise partial
    elif mode == "mb":
        x_ax = np.linspace(1, 51, 10, dtype=int)  # Motion blurr

    # itereate over augmentation scales
    aug_imgs = []
    for i in x_ax:
        if mode == "br":
            alpha = 1
            beta = i
        elif mode == "ct":
            alpha = 1 - i
            beta = 0
        if mode == "br" or mode == "ct":
            noisy_img = cv2.addWeighted(
                image, alpha, np.zeros(image.shape, image.dtype), 0, beta
            )
        elif mode == "bl":
            noisy_img = cv2.GaussianBlur(image, (i, i), 0)
        elif mode == "ns":
            mean = 0
            var = i
            sigma = var**0.5
            gaussian = np.random.normal(mean, sigma, image.shape)
            noisy_img = image + gaussian
            noisy_img = (noisy_img * 255 / noisy_img.max()).astype(np.uint8)
        elif mode == "nsp":
            mean = 0
            var = i
            sigma = var**0.5
            size = 200
            halfsize = int(size / 2)
            sq = [size, size, 3]
            cent = [int(image.shape[0] / 2), int(image.shape[1] / 2)]
            gaussian = np.random.normal(mean, sigma, sq)
            noisy_img = image.copy().astype("float64")
            noisy_img[
                cent[0] - halfsize : cent[0] + halfsize,
                cent[1] - halfsize : cent[1] + halfsize,
            ] += gaussian
            noisy_img = (noisy_img * 255 / noisy_img.max()).astype(np.uint8)
        elif mode == "mb":
            kernel_size = i
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel /= kernel_size
            noisy_img = cv2.filter2D(image, -1, kernel)
        aug_imgs.append(tf.convert_to_tensor(noisy_img, dtype=tf.uint8))
    return aug_imgs


def add_weather(image, weather_condition, val1=None, val2=None, val3=None):
    """Define augmentation pipeline based on weather condition

    Args:
        image (array): Input image
        weather_condition (str): Name of the augmentation
        val1 (float, optional):  Parameter value for augmentation. Defaults to None.
        val2 (float, optional):  Parameter value for augmentation. Defaults to None.
        val3 (float, optional):  Parameter value for augmentation. Defaults to None.

    Raises:
        ValueError: In case the given augmentation name is unavailable

    Returns:
        array: Augmented image
    """
    image = np.asarray(image)
    random.seed(7)
    if weather_condition == "fog":
        transform = A.Compose(
            [
                A.RandomFog(
                    fog_coef_lower=val1 or 0.2,
                    fog_coef_upper=val2 or 0.4,
                    alpha_coef=val3 or 0.5,
                    p=1,
                )
            ]
        )
        augmented_image = transform(image=image)["image"]
    elif weather_condition == "rain":
        transform = A.Compose(
            [
                A.RandomRain(
                    brightness_coefficient=val1 or 0.9,
                    drop_width=val2 or 1,
                    blur_value=val3 or 5,
                    p=1,
                )
            ]
        )
        augmented_image = transform(image=image)["image"]
    elif weather_condition == "snow":
        transform = A.Compose(
            [
                A.RandomSnow(
                    brightness_coeff=val1 or 2.5,
                    snow_point_lower=val2 or 0.3,
                    snow_point_upper=val3 or 0.5,
                    p=1,
                )
            ]
        )
        augmented_image = transform(image=image)["image"]
    elif weather_condition == "noise":
        mean = val1 or 0
        stddev = val2 or 5
        noise = np.random.normal(mean, stddev, image.shape)
        img_noise = image + noise
        augmented_image = np.uint8(img_noise)
    else:
        raise ValueError("Invalid weather condition provided")
    # cv2.imwrite('augmented_image.jpg', cv2.cvtColor(augmented_image,cv2.COLOR_BGR2RGB))
    return augmented_image
