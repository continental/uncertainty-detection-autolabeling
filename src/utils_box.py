# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Utils for localization functions """


import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import stats


def calc_nll(residuals, box_uncert):
    """Negative log likelihood for regression predictions
    Args:
        residuals (array): Absolute difference between prediction and ground truth bounding boxes
        uncert (array): Predicted uncertainty
    Returns:
        Negative log likelihood
    """
    nll_list = np.nan_to_num(stats.norm.logpdf(residuals, scale=box_uncert))
    nll = -1 * np.sum(nll_list)
    nll = nll / len(nll_list)
    return nll


def calc_ece(gt_boxes, pred_boxes, box_uncert):
    """Calculate expected calibration error

    Args:
        gt_boxes (array): Ground truth bounding boxes
        pred_boxes (array): Predicted bounding boxes
        box_uncert (array): Predicted uncertainty
    """
    n_intervals = 100
    p_m = np.linspace(0, 1, n_intervals)
    emp_conf = [0] * n_intervals
    for i in range(n_intervals):
        interval_fit = np.less_equal(
            np.abs(pred_boxes - gt_boxes),
            np.abs(box_uncert * stats.norm.ppf((1 - p_m[i]) / 2)),
        )
        emp_conf[i] = np.mean(interval_fit, axis=0)

    if len(gt_boxes.shape) == 1:  # For one coordinate
        ece = np.mean(np.abs(emp_conf - p_m))
    else:
        ece = np.mean(np.abs(emp_conf - np.swapaxes([p_m] * 4, 0, 1)))
    return ece


def calc_iou_np(gt_boxes, pred_boxes):
    """Numpy function to calculate IoU

    Args:
        gt_boxes (array): Ground truth bounding boxes
        pred_boxes (array): Predicted bounding boxes

    """
    if isinstance(gt_boxes, list):
        gt_boxes = np.asarray(gt_boxes)
    if isinstance(pred_boxes, list):
        pred_boxes = np.asarray(pred_boxes)
    yA = np.maximum(gt_boxes[:, 0], pred_boxes[:, 0])
    xA = np.maximum(gt_boxes[:, 1], pred_boxes[:, 1])
    yB = np.minimum(gt_boxes[:, 2], pred_boxes[:, 2])
    xB = np.minimum(gt_boxes[:, 3], pred_boxes[:, 3])

    interArea = np.maximum(0.0, (xB - xA)) * np.maximum(0.0, (yB - yA)).astype(
        np.float64
    )
    boxAarea = np.abs(gt_boxes[:, 3] - gt_boxes[:, 1]) * np.abs(
        gt_boxes[:, 2] - gt_boxes[:, 0]
    ).astype(np.float64)
    boxBarea = np.abs(pred_boxes[:, 3] - pred_boxes[:, 1]) * np.abs(
        pred_boxes[:, 2] - pred_boxes[:, 0]
    ).astype(np.float64)

    iou = np.divide(
        interArea,
        (boxAarea + boxBarea - interArea),
        out=np.zeros_like(interArea),
        where=(boxAarea + boxBarea - interArea) != 0,
    )
    return iou


def calc_rmse(gt_boxes, pred_boxes):
    """Calculate RMSE

    Args:
        gt_boxes (array): Ground truth bounding boxes
        pred_boxes (array): Predicted bounding boxes

    """
    return tf.math.sqrt(
        tf.reduce_mean(tf.math.square(pred_boxes - gt_boxes)[gt_boxes != 0.0])
    )


def decode_uncert(pred_boxes, box_uncert, anchor_boxes, method="l-norm", n_samples=30):
    """Transform anchor relative distributions to absolute distributions

    Network predictions are normalized and relative to a given anchor; this
    reverses the transformation and outputs absolute coordinates for the input
    image.

    Args:
      pred_boxes (array): predicted box regression targets.
      box_uncert (array): predicted uncertainty targets.
      anchor_boxes (array): anchors on all feature levels.
      method (str): Distribution propagation method
      n_samples (int): Number of samples for the sampling approach
    Returns:
      Decoded bounding boxes
    """
    # st = time.time()
    orig_type = pred_boxes.dtype

    # Anchors
    anchor_boxes = tf.cast(anchor_boxes, dtype=tf.float64)
    ycenter_a = (anchor_boxes[..., 0] + anchor_boxes[..., 2]) / 2
    xcenter_a = (anchor_boxes[..., 1] + anchor_boxes[..., 3]) / 2
    ha = anchor_boxes[..., 2] - anchor_boxes[..., 0]
    wa = anchor_boxes[..., 3] - anchor_boxes[..., 1]

    # Bounding boxes
    pred_boxes = tf.cast(pred_boxes, dtype=tf.float64)
    ty, tx, th, tw = tf.unstack(pred_boxes, num=4, axis=-1)

    # Uncertainty
    box_uncert = tf.cast(box_uncert, dtype=tf.float64)
    pred_var = tf.math.square(box_uncert)
    dty, dtx, dth, dtw = tf.unstack(pred_var, num=4, axis=-1)

    if method == "l-norm":
        w = tf.math.exp(tw + dtw / 2) * wa
        h = tf.math.exp(th + dth / 2) * ha

        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a
        ymin = ycenter - h / 2.0
        xmin = xcenter - w / 2.0
        ymax = ycenter + h / 2.0
        xmax = xcenter + w / 2.0

        dw = (tf.math.exp(dtw) - 1) * tf.math.exp(2 * tw + dtw) * wa**2
        dh = (tf.math.exp(dth) - 1) * tf.math.exp(2 * th + dth) * ha**2

        dycenter = dty * ha**2
        dxcenter = dtx * wa**2

        dymin = dycenter + dh / 4.0
        dxmin = dxcenter + dw / 4.0
        dymax = dycenter + dh / 4.0
        dxmax = dxcenter + dw / 4.0

    elif method == "sample":
        n_dist = tfp.distributions.MultivariateNormalDiag(
            loc=[ty, tx, th, tw], scale_diag=tf.math.sqrt([dty, dtx, dth, dtw])
        )
        generated_samples = n_dist.sample(n_samples)
        sampled_y = generated_samples[:, 0, :, :]
        sampled_x = generated_samples[:, 1, :, :]
        sampled_h = generated_samples[:, 2, :, :]
        sampled_w = generated_samples[:, 3, :, :]

        w = tf.math.exp(sampled_w) * wa
        h = tf.math.exp(sampled_h) * ha
        ycenter = sampled_y * ha + ycenter_a
        xcenter = sampled_x * wa + xcenter_a
        ymin = ycenter - h / 2.0
        xmin = xcenter - w / 2.0
        ymax = ycenter + h / 2.0
        xmax = xcenter + w / 2.0

        ymin, dymin = tf.nn.moments(ymin, axes=[0])
        xmin, dxmin = tf.nn.moments(xmin, axes=[0])
        ymax, dymax = tf.nn.moments(ymax, axes=[0])
        xmax, dxmax = tf.nn.moments(xmax, axes=[0])

    elif method == "n-flow":
        n_y = tfp.distributions.Normal(loc=ty, scale=tf.math.sqrt(dty))
        n_x = tfp.distributions.Normal(loc=tx, scale=tf.math.sqrt(dtx))
        n_h = tfp.distributions.LogNormal(loc=th, scale=tf.math.sqrt(dth))
        n_w = tfp.distributions.LogNormal(loc=tw, scale=tf.math.sqrt(dtw))

        bij_wa = tfp.bijectors.Scale(wa)
        bij_ha = tfp.bijectors.Scale(ha)
        bij_cx = tfp.bijectors.Shift(xcenter_a)
        bij_cy = tfp.bijectors.Shift(ycenter_a)

        t_n_y = tfp.distributions.TransformedDistribution(
            distribution=n_y,
            bijector=tfp.bijectors.Chain(list(reversed([bij_ha, bij_cy]))),
        )

        t_n_x = tfp.distributions.TransformedDistribution(
            distribution=n_x,
            bijector=tfp.bijectors.Chain(list(reversed([bij_wa, bij_cx]))),
        )

        t_n_h = tfp.distributions.TransformedDistribution(
            distribution=n_h, bijector=bij_ha
        )

        t_n_w = tfp.distributions.TransformedDistribution(
            distribution=n_w, bijector=bij_wa
        )

        # Implementation with exponential bijector
        # n_h = tfp.distributions.Normal(loc=th, scale=tf.math.sqrt(dth))
        # n_w = tfp.distributions.Normal(loc=tw, scale=tf.math.sqrt(dtw))

        # bij_exp = tfp.bijectors.Exp()

        # t_n_h = tfp.distributions.TransformedDistribution(
        #     distribution=n_h,
        #     bijector=tfp.bijectors.Chain(list(reversed([bij_exp, bij_ha]))))

        # t_n_w = tfp.distributions.TransformedDistribution(
        #     distribution=n_w,
        #     bijector=tfp.bijectors.Chain(list(reversed([bij_exp, bij_wa]))))

        ycenter, xcenter, h, w = t_n_y.mean(), t_n_x.mean(), t_n_h.mean(), t_n_w.mean()
        dycenter, dxcenter, dh, dw = (
            t_n_y.variance(),
            t_n_x.variance(),
            t_n_h.variance(),
            t_n_w.variance(),
        )

        ymin = ycenter - h / 2.0
        xmin = xcenter - w / 2.0
        ymax = ycenter + h / 2.0
        xmax = xcenter + w / 2.0

        dymin = dycenter + dh / 4.0
        dxmin = dxcenter + dw / 4.0
        dymax = dycenter + dh / 4.0
        dxmax = dxcenter + dw / 4.0

    elif method == "falsedec":
        w = tf.math.exp(tw) * wa
        h = tf.math.exp(th) * ha
        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a
        ymin = ycenter - h / 2.0
        xmin = xcenter - w / 2.0
        ymax = ycenter + h / 2.0
        xmax = xcenter + w / 2.0

        dw = tf.math.exp(dtw) * wa
        dh = tf.math.exp(dth) * ha

        dycenter = dty * ha + ycenter_a
        dxcenter = dtx * wa + xcenter_a

        dymin = tf.abs(dycenter - dh / 2.0)
        dxmin = tf.abs(dxcenter - dw / 2.0)
        dymax = dycenter + dh / 2.0
        dxmax = dxcenter + dw / 2.0

    coords = tf.cast(tf.stack([ymin, xmin, ymax, xmax], axis=-1), dtype=orig_type)
    uncerts = tf.cast(
        tf.math.sqrt(tf.stack([dymin, dxmin, dymax, dxmax], axis=-1)), dtype=orig_type
    )  # Revert to std

    # elapsed_time = time.time() - st
    # txt_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/misc/misc_plots/'+mode+'_decodingmethod_runtime.txt'
    # with open(txt_path,'a') as f: f.write(str(elapsed_time)+'\n')
    return coords, uncerts


def relativize_uncert(pred_boxes, box_uncert):
    """Normalizes the uncertainty per coordinate with the width and height
    Args:
        pred_boxes (array): Predicted bounding boxes
        box_uncert (array): Predicted uncertainty

    """
    pred_boxes = np.asarray(pred_boxes)
    box_uncert = np.asarray(box_uncert)
    width = pred_boxes[:, 3] - pred_boxes[:, 1]
    height = pred_boxes[:, 2] - pred_boxes[:, 0]
    scaling_fac = np.swapaxes([height, width, height, width], 0, 1)
    box_uncert = box_uncert / scaling_fac
    return box_uncert


class CalibrateBoxUncert:
    """Class to calibrate localization uncertainty during inference on images"""

    def __init__(
        self,
        model_params,
        calib_path,
        general_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ):
        """Constructs all the necessary attributes for recalibration

        Args:
            model_params (dict): Dictionary with model parameters
            calib_path (str): Path to calibration models including model name
            general_path (str): Path to working space
        """
        self.model_params = model_params
        self.calibrators_path = (
            general_path + "/results/calibration/" + calib_path + "/regres_models/"
        )
        if model_params["calibrate_regression"]:
            (
                iso_calib_percoo,
                self.iso_calib_all,
                self.temp_regres_all,
                temp_regres_per,
                self.iso_perclscoo,
                self.iso_perclscoo_rel,
            ) = self._import_calib_regress()

            self.ymin_calib, self.xmin_calib, self.ymax_calib, self.xmax_calib = (
                [],
                [],
                [],
                [],
            )
            if iso_calib_percoo:
                self.ymin_calib, self.xmin_calib, self.ymax_calib, self.xmax_calib = (
                    iso_calib_percoo
                )

            (
                self.ymin_calib_temp,
                self.xmin_calib_temp,
                self.ymax_calib_temp,
                self.xmax_calib_temp,
            ) = ([], [], [], [])
            if temp_regres_per:
                (
                    self.ymin_calib_temp,
                    self.xmin_calib_temp,
                    self.ymax_calib_temp,
                    self.xmax_calib_temp,
                ) = temp_regres_per

    def _import_calib_regress(self):
        """Import calibration models for localization calibration

        Returns:
          iso_percoo (list): Four isotonic regression models
          iso_all (object): Isotonic regression model for all coordinates
          ts_all (float): Temperature for all coordinates
          ts_percoo (list): Temperature per coordinate
          iso_perclscoo (list): Isotonic regression models per-class and per-coordinate
          iso_perclscoo_rel (list): Isotonic regression models per-class and per-coordinate, fit with relative uncertainty
        """
        iso_percoo = []
        if os.path.exists(self.calibrators_path + "regression_calib_iso_pcoo"):
            with open(self.calibrators_path + "regression_calib_iso_pcoo", "rb") as fp:
                iso_percoo.append(pickle.load(fp))  # ymin_calib
                iso_percoo.append(pickle.load(fp))  # xmin_calib
                iso_percoo.append(pickle.load(fp))  # ymax_calib
                iso_percoo.append(pickle.load(fp))  # xmax_calib

        iso_all = []
        if os.path.exists(self.calibrators_path + "regression_calib_iso_all"):
            with open(self.calibrators_path + "regression_calib_iso_all", "rb") as fp:
                iso_all = pickle.load(fp)

        ts_all = []
        if os.path.exists(self.calibrators_path + "regression_calib_ts_all"):
            with open(self.calibrators_path + "regression_calib_ts_all", "rb") as fp:
                ts_all = pickle.load(fp)

        ts_percoo = []
        if os.path.exists(self.calibrators_path + "regression_calib_ts_pcoo"):
            with open(self.calibrators_path + "regression_calib_ts_pcoo", "rb") as fp:
                ts_percoo.append(pickle.load(fp))  # ymin_calib_temp
                ts_percoo.append(pickle.load(fp))  # xmin_calib_temp
                ts_percoo.append(pickle.load(fp))  # ymax_calib_temp
                ts_percoo.append(pickle.load(fp))  # xmax_calib_temp

        iso_perclscoo = []
        if os.path.exists(self.calibrators_path + "regression_calib_iso_perclscoo"):
            with open(
                self.calibrators_path + "regression_calib_iso_perclscoo", "rb"
            ) as fp:
                iso_perclscoo = pickle.load(fp)

        iso_perclscoo_rel = []
        if os.path.exists(
            self.calibrators_path + "regression_calib_iso_perclscoo_relative"
        ):
            with open(
                self.calibrators_path + "regression_calib_iso_perclscoo_relative", "rb"
            ) as fp:
                iso_perclscoo_rel = pickle.load(fp)
        return iso_percoo, iso_all, ts_all, ts_percoo, iso_perclscoo, iso_perclscoo_rel

    def calibrate_boxuncert(self, uncert, classes, boxes):
        """_summary_

        Args:
            uncert (array): Localization uncertainty
            classes (array): Ground truth classes for per-class calibration
            boxes (array): Predicted bounding boxes

        Returns:
            select_uncert: Selected uncertainty for further analysis
            iso_all, temp_all, ts_perb, iso_perb, iso_percl, rel_iso_percl: Calibration models
        """
        uncert = np.nan_to_num(uncert)
        iso_all = np.array([])
        if self.iso_calib_all:
            iso_all = self.iso_calib_all.predict(uncert.flatten()).reshape([-1, 4])

        temp_all = np.array([])
        if self.temp_regres_all:
            temp_all = uncert / self.temp_regres_all

        iso_perb = np.array([])
        if self.ymin_calib:
            iso_perb = np.swapaxes(
                [
                    self.ymin_calib.predict(uncert[:, 0]),
                    self.xmin_calib.predict(uncert[:, 1]),
                    self.ymax_calib.predict(uncert[:, 2]),
                    self.xmax_calib.predict(uncert[:, 3]),
                ],
                0,
                1,
            )

        ts_perb = np.array([])
        if self.ymin_calib_temp:
            ts_perb = np.swapaxes(
                [
                    uncert[:, 0] / self.ymin_calib_temp,
                    uncert[:, 1] / self.xmin_calib_temp,
                    uncert[:, 2] / self.ymax_calib_temp,
                    uncert[:, 3] / self.xmax_calib_temp,
                ],
                0,
                1,
            )

        iso_percl = np.array([])
        if self.iso_perclscoo:
            calibrators = np.asarray(self.iso_perclscoo).reshape(
                self.model_params["num_classes"], 4
            )
            iso_percl = np.zeros_like(uncert)
            for ci in range(1, self.model_params["num_classes"] + 1):
                if np.any(classes.astype(int) == ci):
                    for j in range(4):
                        iso_percl[:, j][classes.astype(int) == ci] = calibrators[
                            ci - 1, j
                        ].predict(uncert[:, j][classes.astype(int) == ci])
                else:
                    continue

        rel_iso_percl = np.array([])
        if self.iso_perclscoo_rel:
            width = np.asarray(boxes[:, 3] - boxes[:, 1])
            height = np.asarray(boxes[:, 2] - boxes[:, 0])
            norm = np.swapaxes([height, width, height, width], 0, 1)

            rel_uncert = np.divide(
                uncert,
                norm,
                out=np.zeros_like(uncert),
                where=norm != 0,
                dtype=np.float16,
            )
            rel_calibrators = np.asarray(self.iso_perclscoo_rel).reshape(
                self.model_params["num_classes"], 4
            )
            rel_calib_uncertpc = np.zeros_like(uncert)
            for ci in range(1, self.model_params["num_classes"] + 1):
                if np.any(classes.astype(int) == ci):
                    for j in range(4):
                        rel_calib_uncertpc[:, j][classes.astype(int) == ci] = (
                            rel_calibrators[ci - 1, j].predict(
                                rel_uncert[:, j][classes.astype(int) == ci]
                            )
                        )
                else:
                    continue
            rel_iso_percl = rel_calib_uncertpc * norm

        if self.model_params["calib_method_box"] == "ts_all" and temp_all.any():
            select_uncert = temp_all
        elif self.model_params["calib_method_box"] == "ts_percoo" and ts_perb.any():
            select_uncert = ts_perb
        elif self.model_params["calib_method_box"] == "iso_all" and iso_all.any():
            select_uncert = iso_all
        elif self.model_params["calib_method_box"] == "iso_percoo" and iso_perb.any():
            select_uncert = iso_perb
        elif (
            self.model_params["calib_method_box"] == "iso_perclscoo" and iso_percl.any()
        ):
            select_uncert = iso_percl
        elif (
            self.model_params["calib_method_box"] == "rel_iso_perclscoo"
            and rel_iso_percl.any()
        ):
            select_uncert = rel_iso_percl
        else:
            select_uncert = np.array([])
            print("Unknown calibration method")

        return (
            select_uncert,
            iso_all,
            temp_all,
            ts_perb,
            iso_perb,
            iso_percl,
            rel_iso_percl,
        )
