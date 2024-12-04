# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Utils for different functions """


import io
import itertools
import os

import numpy as np
import scipy
import sklearn.metrics
import tensorflow as tf
import uncertainty_toolbox as uct
import uncertainty_toolbox.viz as uvi
from absl import logging
from matplotlib import pyplot as plt
from utils_box import calc_ece, calc_iou_np, calc_rmse, relativize_uncert

logging.set_verbosity(logging.INFO)


def calc_jsd(dist_1, dist_2):
    """Calculate the Jensen–Shannon divergence between two distributions"""
    mean_1, std_1 = np.mean(dist_1), np.std(dist_1)
    mean_2, std_2 = np.mean(dist_2), np.std(dist_2)
    pdf_1 = scipy.stats.norm(loc=mean_1, scale=std_1).pdf
    pdf_2 = scipy.stats.norm(loc=mean_2, scale=std_2).pdf
    x = np.linspace(
        min(mean_1 - 3 * std_1, mean_2 - 3 * std_2),
        max(mean_1 + 3 * std_1, mean_2 + 3 * std_2),
        1000,
    )
    return scipy.spatial.distance.jensenshannon(pdf_1(x), pdf_2(x))


def dict_tf_to_np(vals):
    """Turn dict values from tensors to numpy arrays"""
    return {i: np.asarray(vals[i]) for i in vals}


def gt_box_assigner(sorting_method, gt_box, boxes, i):
    """Assign a GT box the suitable predicted box from NMS output

    Args:
        sorting_method (str): Sorting method pre-defined parameter
        gt_box (array): GT bounding box
        boxes (array): Predicted boxes
        i (int): Current iteration index

    Returns:
        Corrected index
    """
    if sorting_method == "MSE":
        correct_index = np.argmin(
            np.mean(np.square([gt_box[i]] * len(boxes) - boxes), axis=1)
        )
    elif sorting_method == "IoU":
        correct_index = np.argmax(calc_iou_np([gt_box[i]] * len(boxes), boxes))
    else:
        correct_index = i
    return correct_index


def add_array_dict(data_dict, source_array, target_key, select_index):
    """Add array values to a dict

    Args:
        data_dict (dict): Target dictionary
        source_array (array): Array to save
        target_key (str): Dict key
        select_index (int):
    """
    if source_array.size > 0:
        vals = np.nan_to_num(np.around(source_array[select_index].astype("float32"), 4))
        if source_array[select_index].size > 1:
            vals = list(vals)
        data_dict[target_key] = vals
    return data_dict


def update_arrays(target_array, source_array, select_index):
    """Adds an array to another array

    Args:
        target_array (array): Array to append to
        source_array (array): Array to append
        select_index (int): Index to select item from

    Returns:
        Updated array
    """
    if source_array.size > 0:
        if target_array.size > 0:
            target_array = np.vstack((target_array, [source_array[select_index]]))
        else:
            target_array = np.array([source_array[select_index]])
    return target_array


def save_uncert(path, uncert, title):
    """Add uncertainty per image to file

    Args:
        path (str): Path to saving file
        uncert (array): Uncertainty per image for all objects
        title (str): Title to save under
    """
    with open(path + "/uncert" + title + ".txt", "a") as f:
        if len(uncert) != 0:
            uncert = np.max(uncert)  # Per image
        else:
            uncert = np.NaN
        f.write(str(uncert) + "\n")


def mc_infer(driver, image, T=10):
    """Takes the serving driver, feeds it an image T times and aggregates results

    Args:
        driver (object): Serving driver to the model
        image (array): Input image
        T (int, optional): Number of MC samples. Defaults to 10.

    Returns:
        Aggregated detections
    """
    d0 = driver.serve(image)
    detections_bs = [[d0[i]] for i in range(len(d0))]
    for _ in range(1, T):
        det_temp = driver.serve(image)
        for i in range(len(det_temp)):
            detections_bs[i].append(det_temp[i])

    for i in range(len(detections_bs)):
        detections_bs[i] = tf.stack(detections_bs[i], axis=0)
    return detections_bs


def mc_eval(mc_model, images, config):
    """Used for evaluation of the model with MC dropout, feeds the model and aggregates output.

    Args:
        mc_model (object): EfficientDet model
        images (tensor): Batch of images
        config (dict): Dictionary with hyperparams

    Returns:
        Stacked output
    """
    if config.mc_classheadrate or config.mc_dropoutrate:
        cls_outputs0, cls_outputs1, cls_outputs2, cls_outputs3, cls_outputs4 = (
            [],
            [],
            [],
            [],
            [],
        )
    if config.mc_boxheadrate or config.mc_dropoutrate:
        box_outputs0, box_outputs1, box_outputs2, box_outputs3, box_outputs4 = (
            [],
            [],
            [],
            [],
            [],
        )
    output = []
    for _ in range(config.mc_dropoutsamp):
        cls, box = mc_model(images, training=False)
        if config.mc_classheadrate or config.mc_dropoutrate:
            cls_outputs0.append(cls[0])
            cls_outputs1.append(cls[1])
            cls_outputs2.append(cls[2])
            cls_outputs3.append(cls[3])
            cls_outputs4.append(cls[4])
        if config.mc_boxheadrate or config.mc_dropoutrate:
            box_outputs0.append(box[0])
            box_outputs1.append(box[1])
            box_outputs2.append(box[2])
            box_outputs3.append(box[3])
            box_outputs4.append(box[4])

    if config.mc_classheadrate or config.mc_dropoutrate:
        cls_concat = stack_mcpred(
            [cls_outputs0, cls_outputs1, cls_outputs2, cls_outputs3, cls_outputs4]
        )
    else:
        cls_concat = cls  # Deterministic
    if config.mc_boxheadrate or config.mc_dropoutrate:
        box_concat = stack_mcpred(
            [box_outputs0, box_outputs1, box_outputs2, box_outputs3, box_outputs4]
        )
    else:
        box_concat = box
    output.extend([cls_concat, box_concat])
    return output


def stack_mcpred(output):
    """Collects and stacks predictions with MC dropout activated

    Args:
        output (list): Model output from multiple MC iterations

    Returns:
        Stacked output
    """
    output0, output1, output2, output3, output4 = output
    output0 = tf.stack(output0, axis=0)
    output1 = tf.stack(output1, axis=0)
    output2 = tf.stack(output2, axis=0)
    output3 = tf.stack(output3, axis=0)
    output4 = tf.stack(output4, axis=0)
    splitoutputs = [output0, output1, output2, output3, output4]
    return splitoutputs


def get_mcuncert(output):
    """Calcultates the mean and uncertainty of multiple predictions

    Args:
        output (list): Model output from multiple MC iterations

    Returns:
        Mean and uncertainty
    """
    output0, output1, output2, output3, output4 = output
    mean = [
        tf.reduce_mean(output0, axis=0),
        tf.reduce_mean(output1, axis=0),
        tf.reduce_mean(output2, axis=0),
        tf.reduce_mean(output3, axis=0),
        tf.reduce_mean(output4, axis=0),
    ]
    uncert = [
        tf.math.reduce_std(output0, axis=0),
        tf.math.reduce_std(output1, axis=0),
        tf.math.reduce_std(output2, axis=0),
        tf.math.reduce_std(output3, axis=0),
        tf.math.reduce_std(output4, axis=0),
    ]
    return mean, uncert


def bin_values(x, y, n_bins):
    """Bins the x values into n bins and calculates the mean and standard deviation of y in each bin

    Args:
        x (list): x values
        y (list): y values
        n_bins (int): Number of bins

    Returns:
        mean: Mean of y in each bin
        std: Standard deviation of y in each bin
        centers: Center location on the x axis of each bin
        count: Count of y in each bin
    """
    x = np.asarray(x)
    y = np.asarray(y)
    b = np.linspace(0.0, 1 + 1e-8, n_bins + 1)
    b[-1] -= 1e-8
    bins = np.quantile(x, b)
    bins = np.unique(bins).astype(np.float64)
    bins[-1] = 1 + 1e-8
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    mean = np.asarray(
        [
            np.mean(
                y[[bins[i - 1] < x][0] * [x <= bins[i]][0]]
                if len(y[[bins[i - 1] < x][0] * [x <= bins[i]][0]]) != 0
                else 0
            )
            for i in range(0, len(bins))
        ]
    )
    std = np.asarray(
        [
            np.std(
                y[[bins[i - 1] < x][0] * [x <= bins[i]][0]]
                if len(y[[bins[i - 1] < x][0] * [x <= bins[i]][0]]) != 0
                else 0
            )
            for i in range(0, len(bins))
        ]
    )
    count = np.asarray(
        [
            len(np.where([bins[i - 1] < x][0] * [x <= bins[i]][0] == True)[0])
            for i in range(0, len(bins))
        ]
    )
    centers = np.asarray(
        [0] + [(bins[i] + bins[i - 1]) / 2 for i in range(1, len(bins))]
    )
    return mean, std, centers, count


def mtplt_to_img(figure):
    """Converts a matplotlib figure to a PNG image"""
    buf = io.BytesIO()
    plt.savefig(buf, format="png")  # Save plot to memory
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(
        buf.getvalue(), channels=4
    )  # Convert to tensor png image
    image = tf.expand_dims(image, 0)
    return image


def plot_roc(y_true, y_pred, class_names):
    """Plots ROC curve"""
    y_true = tf.one_hot(y_true, len(class_names))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    figure = plt.figure()
    linewidth = 2
    colors = itertools.cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=linewidth,
            label="ROC curve of class {0} (AUC = {1:0.2f})".format(
                class_names[i], roc_auc[i]
            ),
        )
    plt.plot([0, 1], [0, 1], "k--", lw=linewidth, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    return figure


def plot_conf_matrix(conf_matrix, class_names):
    """Plots confusion matrix"""

    figure = plt.figure()
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize confusion matrix
    conf_matrix = np.around(
        conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis], decimals=2
    )
    threshold = conf_matrix.max() * 0.8
    for i, j in itertools.product(
        range(conf_matrix.shape[0]), range(conf_matrix.shape[1])
    ):
        color = (
            "white" if conf_matrix[i, j] > threshold else "black"
        )  # Use white text if squares are dark; otherwise black
        plt.text(j, i, conf_matrix[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


class ValidUncertPlot:
    """Plots different metrics and correlations about the localization and its uncertainty on the validation set"""

    def __init__(
        self,
        gt_boxes,
        pred_boxes,
        gt_classes,
        pred_classes,
        uncert,
        calib_uncert,
        save_path,
        model_params,
    ):
        """
        Args:
          gt_boxes (array): Ground truth bounding boxes
          pred_boxes (array): Predicted bounding boxes
          gt_classes (array): Ground truth classes
          pred_classes (array): Predicted classes
          uncert (array): Predicted uncertainty
          calib_uncert (array): Calibrated predicted uncertainty
          save_path (str): Saving path
          model_params (dict): Model parameters
        """
        self.gt_boxes = gt_boxes
        self.pred_boxes = pred_boxes
        self.gt_classes = gt_classes
        self.pred_classes = pred_classes
        self.uncert = uncert
        self.calib_uncert = calib_uncert
        self.save_path = save_path
        self.model_params = model_params
        self.num_classes = model_params["num_classes"]

        self.relative_uncert = relativize_uncert(pred_boxes, uncert)
        if calib_uncert is not None:
            self.relative_uncert_calib = relativize_uncert(pred_boxes, calib_uncert)

        self.ious = calc_iou_np(gt_boxes, pred_boxes)
        md = len(self.ious[self.ious == 0]) / len(self.ious) * 100
        misc = len(np.where(gt_classes != pred_classes)[0]) / len(gt_classes)
        miou = np.mean(self.ious)
        rmse = float(calc_rmse(gt_boxes, pred_boxes))
        ece = calc_ece(gt_boxes.flatten(), pred_boxes.flatten(), uncert.flatten())
        if os.path.exists(os.path.dirname(self.save_path) + "/model_performance.txt"):
            with open(
                os.path.dirname(self.save_path) + "/model_performance.txt", "a"
            ) as f:
                f.write(
                    "ECE " + self.save_path.split("/")[-1] + ": " + str(ece) + " \n"
                )
        else:
            with open(
                os.path.dirname(self.save_path) + "/model_performance.txt", "a"
            ) as f:
                f.write("% Missing Detections: " + str(md) + " \n")
                f.write("Misclassification rate: " + str(misc) + " \n")
                f.write("mIoU: " + str(miou) + " \n")
                f.write("RMSE: " + str(rmse) + " \n")
                f.write(
                    "Uncertainty Decoding: "
                    + model_params["uncert_adjust_method"]
                    + " \n"
                )
                f.write(
                    "ECE Loc " + self.save_path.split("/")[-1] + ": " + str(ece) + " \n"
                )
        # Uncertainty quality
        self.collect_uvi()

    def _uvi_plot_metrics(self, gt_boxes, pred_boxes, uncert, title="uncalibrated_all"):
        """Plots and saves uncertainty metrics

        Args:
            uncert (array): Either calibrated or uncalibrated uncertainty
            title (str, optional): Saving title. Defaults to "uncalibrated".
        """
        if not np.all(uncert > 0.0):
            uncert[uncert == 0.0] += 1e-6

        plt.figure(figsize=(20, 10))
        ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
        ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
        ax4 = plt.subplot2grid((2, 6), (1, 0), colspan=2)
        ax5 = plt.subplot2grid((2, 6), (1, 2), colspan=2)

        ax1 = uvi.plot_xy(
            pred_boxes[0:100],
            uncert[0:100],
            gt_boxes[0:100],
            np.arange(len(gt_boxes[0:100])),
            ax=ax1,
        )
        ax2 = uvi.plot_intervals_ordered(pred_boxes, uncert, gt_boxes, ax=ax2)
        ax4 = uvi.plot_calibration(pred_boxes, uncert, gt_boxes, ax=ax4)
        ax5 = uvi.plot_sharpness(uncert, ax=ax5)

        plt.savefig(
            self.save_path + "/uncertainty_toolbox/uvi_" + title + ".png",
            bbox_inches="tight",
        )
        plt.close()

        metrics = uct.metrics.get_all_metrics(
            pred_boxes.astype(np.float32),
            uncert.astype(np.float32),
            gt_boxes.astype(np.float32),
        )
        with open(
            self.save_path + "/uncertainty_toolbox/uvi_" + title + ".txt", "w"
        ) as file:
            for metric_category, metric_values in metrics.items():
                file.write(f"{metric_category}:\n")
                for metric_name, metric_info in metric_values.items():
                    if isinstance(metric_info, dict):
                        file.write(f"  {metric_name}:\n")
                        for sub_metric, sub_value in metric_info.items():
                            if isinstance(sub_value, np.ndarray):
                                file.write(f"    {sub_metric}: {sub_value.tolist()}\n")
                            else:
                                file.write(f"    {sub_metric}: {sub_value}\n")
                    else:
                        file.write(f"  {metric_name}: {metric_info}\n")
                file.write("\n")

    def _cdf(self):
        """Calculate cumulative distribution function

        Returns:
          Cumulative distribution function values
        """
        n = len(self.pred_boxes.flatten())
        pcdf = np.zeros(n)
        for i in range(n):
            pcdf[i] = scipy.stats.norm.cdf(
                self.gt_boxes.flatten()[i],
                loc=self.pred_boxes.flatten()[i],
                scale=self.uncert.flatten()[i],
            )
        return pcdf

    def collect_uvi(self):
        """Plots uncertainty toolbox results"""
        if not os.path.exists(self.save_path + "/uncertainty_toolbox"):
            os.makedirs(self.save_path + "/uncertainty_toolbox")

        def _straight_line(a, b):
            """Plot an ax+b line"""
            ax = plt.gca()
            x = np.array(ax.get_xlim())
            y = a * x + b
            plt.plot(x, y, "--")

        def _plot_cdf(cdf, step=0.01):
            """Plot cumulative distribution function"""
            x = np.arange(0, 1, step) + step
            cumulative_cdf = np.zeros(len(x))
            for i in range(len(x)):
                cumulative_cdf[i] = np.mean(cdf <= x[i])
            fig, ax = plt.subplots()
            ax.scatter(x, cumulative_cdf, c="black")
            _straight_line(1, 0)
            plt.grid()
            plt.ylabel("Predicted cumulative distribution")
            plt.xlabel("Empirical cumulative distribution")
            plt.savefig(self.save_path + "/uncertainty_toolbox/cdf_allcoo.png")
            plt.close()

        _plot_cdf(self._cdf())
        self._uvi_plot_metrics(
            self.gt_boxes.flatten(), self.pred_boxes.flatten(), self.uncert.flatten()
        )
        if self.calib_uncert is not None:
            self._uvi_plot_metrics(
                self.gt_boxes.flatten(),
                self.pred_boxes.flatten(),
                self.uncert.flatten(),
                "calibrated_all",
            )

        coords_names = ["ymin", "xmin", "ymax", "xmax"]
        for i in range(4):
            self._uvi_plot_metrics(
                self.gt_boxes[:, i],
                self.pred_boxes[:, i],
                self.uncert[:, i],
                "uncalibrated_" + coords_names[i],
            )
            if self.calib_uncert is not None:
                self._uvi_plot_metrics(
                    self.gt_boxes[:, i],
                    self.pred_boxes[:, i],
                    self.uncert[:, i],
                    "calibrated_" + coords_names[i],
                )
