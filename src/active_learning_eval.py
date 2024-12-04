# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Specific for paper: Overcoming Common Challenges in Active Learning Pipelines for Object Detection via Similarity """

import json
import os
import re
import subprocess
import sys

import cv2
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import yaml
from matplotlib.colors import ListedColormap
from matplotlib.path import Path
from scipy.fftpack import dct
from scipy.spatial import cKDTree as KDTree
from scipy.stats import gaussian_kde, kendalltau, pearsonr
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(7, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.BDD100K.bdd_tf_creator import bdd_active_tfrecords
from datasets.CODA.coda_tf_creator import (
    coda_active_tfrecords,
    discard_classes,
    extract_class_info,
)
from datasets.KITTI.kitti_tf_creator import kitti_active_tfrecords

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

RANDOM_STATE_SEED = 42
np.random.seed(RANDOM_STATE_SEED)

GPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DPATH = "/project_data/SSAL/" or GPATH

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def extract_tf_data(tfpath):
    """Extracts data from TFRecord

    Args:
        tfpath (str): Path to TFRecord

    Returns:
        Lists of images, classes and boxes per image
    """
    dataset = tf.data.TFRecordDataset(tfpath)
    collect_images = []
    collect_labels = []
    collect_boxes = []
    collect_names = []
    for record in dataset:
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        collect_names.append(
            str(example.features.feature["image/filename"].bytes_list.value[0])
            .split(".")[0]
            .split("'")[1]
        )

        encoded_image = example.features.feature["image/encoded"].bytes_list.value[0]
        image = tf.image.decode_image(encoded_image).numpy()

        width = int(image.shape[1])
        height = int(image.shape[0])
        box_keys = [
            "image/object/bbox/ymin",
            "image/object/bbox/xmin",
            "image/object/bbox/ymax",
            "image/object/bbox/xmax",
        ]
        norm = [float(height), float(width), float(height), float(width)]
        boxes = [
            np.asarray(example.features.feature[bbox_key].float_list.value) * norm[x]
            for x, bbox_key in enumerate(box_keys)
        ]
        boxes_combined = []
        for i in range(0, len(boxes[0])):
            box_comb = [boxes[j][i] for j in range(len(boxes))]
            boxes_combined.append(box_comb)
        collect_boxes.append(boxes_combined)
        label = example.features.feature["image/object/class/text"].bytes_list.value
        labels = [x.decode("utf-8") for x in label]
        collect_images.append(image)
        collect_labels.append(labels)
    return collect_images, collect_labels, collect_boxes


def update_eval_config(yaml_path, new_model_dir, update_name=None, eval_samples=0):
    """Changes data in evaluation config file

    Args:
        yaml_path (str): Path to config file
        new_model_dir (str): Model directory name as defined in config
        update_name (any, optional): An additional name to the val TFRecord path in case needed. Defaults to None.
        eval_samples (int, optional): Number of samples in TFRecord. Defaults to 0.
    """
    with open(yaml_path, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    data["model_dir"] = new_model_dir
    if update_name is not None:
        data["val_file_pattern"] = (
            data["val_file_pattern"].split("/_val")[0]
            + "/_val_set"
            + str(update_name)
            + ".tfrecord"
        )
        data["eval_samples"] = int(eval_samples)
    if DPATH != GPATH:
        data["model_dir"] = DPATH + "models" + data["model_dir"].split("/models")[1]
        data["hparams"] = DPATH + "configs" + data["hparams"].split("/configs")[1]
        data["val_file_pattern"] = (
            DPATH + "datasets" + data["val_file_pattern"].split("datasets")[1]
        )

    with open(yaml_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def extract_eval_metrics_tensorboard(event_file):
    """Extract data from Tensorboard"""
    ap = []
    valloss = []
    miou = []
    mf1 = []
    f1 = []
    try:
        event_acc = EventAccumulator(event_file + "coco")
        event_acc.Reload()
        temp_ap = float(tf.make_ndarray(event_acc.Tensors("AP")[-1][2])) * 100
        temp_ap50 = float(tf.make_ndarray(event_acc.Tensors("AP50")[-1][2])) * 100
        temp_ap75 = float(tf.make_ndarray(event_acc.Tensors("AP")[-1][2])) * 100
        ap.append([temp_ap, temp_ap50, temp_ap75])

        event_acc = EventAccumulator(event_file + "train")
        event_acc.Reload()
        miou_val = float(tf.make_ndarray(event_acc.Tensors("mIou")[-1][2]))
        miou.append(miou_val)

        event_acc = EventAccumulator(event_file + "validation")
        event_acc.Reload()
        val_det_loss = float(
            tf.make_ndarray(event_acc.Tensors("epoch_det_loss")[-1][2])
        )
        valloss.append(val_det_loss)

        keys_with_F1 = [key for key in event_acc.Tags()["tensors"] if "_F1_" in key]
        tmp_mf1 = []
        for key in keys_with_F1:
            mf1_cv = float(tf.make_ndarray(event_acc.Tensors(key)[-1][2]))
            tmp_mf1.append(mf1_cv)
        f1.append(tmp_mf1)
        mf1.append(np.mean(tmp_mf1))
    except:
        valloss.append(np.nan)
        ap.append(np.nan)
        miou.append(np.nan)
        mf1.append(np.nan)
        f1.append(np.nan)
    return ap, valloss, miou, mf1, f1


def run_eval(output_file, dataset_number=2):
    """Run evaluation python file"""
    with open(output_file, "w") as file:
        subprocess.run(
            "python eval.py --dataset " + str(dataset_number),
            shell=True,
            stdout=file,
            stderr=subprocess.STDOUT,
        )


def parallel_plot(
    data, methods, n_iter, dataset, classes=None, per_class=False, ap_only=False
):
    """Generates a parallel plot (multiply y axes)

    Args:
        data (list): List with plotting values
        methods (_type_): _description_
        n_iter (int): AL iteration number for x label
        dataset (str): Dataset name for saving path
        classes (list, optional): List of class names. Defaults to None.
        per_class (bool, optional): Plots APs for each class. Defaults to False.
        ap_only (bool, optional): Plots APs only. Defaults to False.
    """
    if per_class:
        title = "performance_perc"
    elif ap_only:
        title = "performance_aponly"
    else:
        title = "performance_metrics"
    if per_class:
        x_labels = classes
        orig_y = np.asarray(
            [
                [val if not np.all(np.isnan(val)) else np.nan for val in data[i]]
                for i in range(len(data))
            ]
        )
    elif ap_only:
        x_labels = ["AP", "AP50", "AP75"]
        orig_y = np.asarray(
            [
                [val if not np.all(np.isnan(val)) else np.nan for val in data[i]]
                for i in range(len(data))
            ]
        )
    else:
        x_labels = ["AP", "AP50", "AP75", "mIoU", "mF1", "Val Loss"]
        data_AP = np.asarray(data)[:, 0][:, 0]
        data_AP = np.asarray(
            [val if not np.all(np.isnan(val)) else [np.nan] * 3 for val in data_AP]
        )
        data_valloss = np.asarray(data)[:, 1][:, 0]
        data_miou = np.asarray(data)[:, 2][:, 0]
        data_mf1 = np.asarray(data)[:, 3][:, 0]
        data = np.concatenate(
            [
                np.stack(data_AP),
                data_miou[:, np.newaxis],
                data_mf1[:, np.newaxis],
                data_valloss[:, np.newaxis],
            ],
            axis=-1,
        )
        orig_y = np.asarray(
            [
                [val if not np.all(np.isnan(val)) else np.nan for val in data[i]]
                for i in range(len(data))
            ]
        )
    ymins = np.asarray(
        [orig_y[:, i][~np.isnan(orig_y[:, i])].min() for i in range(len(x_labels))]
    )
    ymaxs = np.asarray(
        [orig_y[:, i][~np.isnan(orig_y[:, i])].max() for i in range(len(x_labels))]
    )
    ymins *= 0.95
    ymaxs *= 1.05  # 5% padding
    dys = ymaxs - ymins
    # Transform all data to be compatible with the main axis
    zs = np.zeros_like(orig_y)
    zs[:, 0] = orig_y[:, 0]
    zs[:, 1:] = (orig_y[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]
    num_axes = orig_y.shape[1]
    fig, host = plt.subplots()
    axes = [host] + [host.twinx() for i in range(num_axes - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        if ax != host:
            ax.yaxis.set_ticks_position("right")
            ax.spines["right"].set_position(("axes", i / (num_axes - 1)))
            ax.spines["left"].set_visible(False)

    host.set_xlim(0, num_axes - 1)
    host.set_xticks(range(num_axes))
    host.set_xticklabels(x_labels)
    host.tick_params(axis="x", which="major", pad=7)
    host.spines["right"].set_visible(False)
    host.set_xlabel("Active Learning Iteration " + str(n_iter))

    colors = plt.cm.tab10.colors
    handles = []
    x_values = np.linspace(0, num_axes - 1, num_axes)
    for j in range(len(methods)):
        verts = list(zip(x_values, zs[j, :]))
        handles.append(mlines.Line2D([], [], color=colors[j], label=methods[j]))
        host.add_patch(
            patches.PathPatch(
                Path(
                    verts, [Path.MOVETO] + [Path.LINETO for _ in range(len(verts) - 1)]
                ),
                facecolor="none",
                lw=1,
                edgecolor=colors[j],
            )
        )
        if len(zs[j, :][~np.isnan(zs[j, :])]) < len(x_labels) / 2:
            host.scatter(x_values, zs[j, :], marker="x", color=colors[j])
    # Save the figure without legend
    plt.tight_layout()
    plt.savefig(GPATH + "/misc/papers/paper_al/" + dataset + "/" + title + ".png")
    plt.close()

    # Create a separate legend plot
    fig_legend, ax_legend = plt.subplots()
    ax_legend.legend(handles=handles)
    ax_legend.axis("off")
    plt.tight_layout()
    fig_legend.savefig(
        GPATH + "/misc/papers/paper_al/" + dataset + "/" + title + "_legend.png"
    )
    plt.close("all")


def read_aps_from_log(model_paths, dpath):
    """Reads mAP results from log file saved during trianing"""
    parent_path = GPATH + "/misc/papers/paper_al/" + dpath
    if not os.path.exists(parent_path):
        os.mkdir(parent_path)
    aps = []
    for new_model_dir in model_paths:
        eval_path = parent_path + "/" + new_model_dir.split("/")[-2]
        if not os.path.exists(eval_path) and "split" not in dpath:
            script_folder = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_folder)
            if "CODA" in dpath:
                if "BDD" in new_model_dir:
                    eval_ds = "bc"
                    eval_ds_ind = 3
                else:
                    eval_ds = "kc"
                    eval_ds_ind = 2
            elif "BDD" in dpath:
                eval_ds = "b"
                eval_ds_ind = 1
            elif "KITTI" in dpath:
                eval_ds = "k"
                eval_ds_ind = 0
            update_eval_config(
                GPATH + "/configs/eval/eval_" + eval_ds + ".yaml", new_model_dir
            )
            run_eval(eval_path, dataset_number=eval_ds_ind)
        with open(eval_path, "r") as file:
            text = file.read()
        matches = re.findall(r"'AP[^']*':\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text)
        # Check if any matches are found
        if matches:
            # Convert the values to floats and save to a list
            ap_values = [float(match) for match in matches]
            aps.append(ap_values)
    return aps


def plot_eval_results(methods, classes, orig_classes, model_paths, dataset="KITTI"):
    """Plots the evaluation results"""
    script_folder = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_folder)
    if "KITTI" in dataset:
        for new_model_dir in model_paths:
            if not os.path.exists(
                GPATH
                + "/misc/papers/paper_al/KITTI_eval/"
                + new_model_dir.split("/")[-2]
            ):
                update_eval_config(GPATH + "/configs/eval/eval_k.yaml", new_model_dir)
                run_eval(
                    GPATH
                    + "/misc/papers/paper_al/KITTI_eval/"
                    + new_model_dir.split("/")[-2],
                    dataset_number=0,
                )

        perfmetrics = [
            extract_eval_metrics_tensorboard(event_file) for event_file in model_paths
        ]
        parallel_plot(perfmetrics, methods, 1, dataset)

        kitti_aps = read_aps_from_log(model_paths, "KITTI_eval")
        ap_cls = np.asarray(kitti_aps)[:, -len(classes) :]
        parallel_plot(ap_cls, methods, 1, dataset, classes, per_class=True)
    elif "BDD" in dataset:
        for new_model_dir in model_paths:
            if not os.path.exists(
                GPATH + "/misc/papers/paper_al/BDD_eval/" + new_model_dir.split("/")[-2]
            ):
                update_eval_config(GPATH + "/configs/eval/eval_b.yaml", new_model_dir)
                run_eval(
                    GPATH
                    + "/misc/papers/paper_al/BDD_eval/"
                    + new_model_dir.split("/")[-2],
                    dataset_number=1,
                )

        perfmetrics = [
            extract_eval_metrics_tensorboard(event_file) for event_file in model_paths
        ]
        parallel_plot(perfmetrics, methods, 1, dataset)

        bdd_aps = read_aps_from_log(model_paths, "BDD_eval")
        ap_cls = np.asarray(bdd_aps)[:, -len(classes) :]
        parallel_plot(
            ap_cls, methods, 1, dataset, [cls[:9] for cls in classes], per_class=True
        )
    elif "bCODA" in dataset:
        for new_model_dir in model_paths:
            if not os.path.exists(
                GPATH
                + "/misc/papers/paper_al/CODA_eval/"
                + new_model_dir.split("/")[-2]
            ):
                update_eval_config(GPATH + "/configs/eval/eval_s.yaml", new_model_dir)
                run_eval(
                    GPATH
                    + "/misc/papers/paper_al/CODA_eval/"
                    + new_model_dir.split("/")[-2],
                    dataset_number=4,
                )

        coda_aps = read_aps_from_log(model_paths, "CODA_eval")
        parallel_plot(np.asarray(coda_aps)[:, :3], methods, 1, dataset, ap_only=True)
        ap_cls = np.asarray(coda_aps)[
            :,
            [
                -len(orig_classes) + i
                for i in range(len(orig_classes))
                if orig_classes[i] in classes
            ],
        ]
        parallel_plot(ap_cls, methods, 1, dataset, classes, per_class=True)
    else:
        for new_model_dir in model_paths:
            if not os.path.exists(
                GPATH
                + "/misc/papers/paper_al/CODA_eval/"
                + new_model_dir.split("/")[-2]
            ):
                update_eval_config(GPATH + "/configs/eval/eval_c.yaml", new_model_dir)
                run_eval(
                    GPATH
                    + "/misc/papers/paper_al/CODA_eval/"
                    + new_model_dir.split("/")[-2]
                )

        coda_aps = read_aps_from_log(model_paths, "CODA_eval")
        parallel_plot(np.asarray(coda_aps)[:, :3], methods, 1, dataset, ap_only=True)
        ap_cls = np.asarray(coda_aps)[
            :,
            [
                -len(orig_classes) + i
                for i in range(len(orig_classes))
                if orig_classes[i] in classes
            ],
        ]
        parallel_plot(ap_cls, methods, 1, dataset, classes, per_class=True)


def emp_KL_divergence(sample_p, sample_q):
    """Calculates the Kullback-Leibler divergence between two probability distributions from samples.

    Args:
        sample_p (array): Each row represents a sample from distribution P.
        sample_q (array): Each row represents a sample from distribution Q.

    Returns:
        The estimated KL divergence, indicating the difference between the two distributions.

    References:
        Pérez-Cruz, F. (2008). Kullback-Leibler divergence estimation of continuous distributions.
        IEEE International Symposium on Information Theory.
    """
    num_samples_p, dimensions_p = sample_p.shape
    num_samples_q, dimensions_q = sample_q.shape

    if dimensions_p != dimensions_q:
        raise ValueError("Both sample sets must have the same number of dimensions.")

    # Construct KD trees from the samples for nearest neighbor queries
    tree_p = KDTree(sample_p)
    tree_q = KDTree(sample_q)

    # Compute nearest neighbour distances within same distribution and to the other distribution
    distances_p = tree_p.query(sample_p, k=2, eps=0.01, p=2)[0][
        :, 1
    ]  # Exclude self-match
    distances_q = tree_q.query(sample_p, k=1, eps=0.01, p=2)[0]  # Closest in Q

    # Calculate KL divergence using Eq 14 with fixed (-) sign
    kl_div = -np.log(
        distances_p / distances_q
    ).sum() * dimensions_p / num_samples_p + np.log(num_samples_q / (num_samples_p - 1))
    return kl_div


def empirical_jensen_shannon_divergence(P, Q, num_samples):
    """Calculate Jensen-Shannon divergence between two 2D data samples"""

    # Transform the data
    log_P = np.log(P.T)
    log_Q = np.log(Q.T)

    finite_mask_P = np.all(np.isfinite(log_P), axis=0)
    finite_mask_Q = np.all(np.isfinite(log_Q), axis=0)

    log_P = log_P[:, finite_mask_P]
    log_Q = log_Q[:, finite_mask_Q]

    # Check if data is empty after filtering
    if log_P.size == 0 or log_Q.size == 0:
        raise ValueError("Filtered data is empty, cannot proceed with KDE.")

    # Create KDEs from transformed data
    kde_p = gaussian_kde(log_P)
    kde_q = gaussian_kde(log_Q)

    # Sample from transformed KDEs
    log_samples_p = kde_p.resample(size=num_samples, seed=RANDOM_STATE_SEED)
    log_samples_q = kde_q.resample(size=num_samples, seed=RANDOM_STATE_SEED)

    # Create a synthetic midpoint distribution M by combining samples from both distributions
    log_samples_m = np.concatenate((log_samples_p, log_samples_q), axis=1)
    kde_m = gaussian_kde(log_samples_m)  # Create a KDE for the combined samples
    log_samples_m = kde_m.resample(
        size=num_samples, seed=RANDOM_STATE_SEED
    )  # Resample from the KDE of combined samples

    # Transform samples back
    samples_p = np.exp(log_samples_p)
    samples_q = np.exp(log_samples_q)
    samples_m = np.exp(log_samples_m)

    # Uncomment the following to plot hist bins for each dimension before and after sampling
    # fig, axes = plt.subplots(3, 1, figsize=(5, 7))  # Use a single column for clarity

    # # Plot for each dimension
    # for i in range(3):
    #     # Generate bins that work for both histograms
    #     bins = np.histogram(np.hstack((P[:, i], samples_p[i, :])), bins=30)[
    #         1
    #     ]  # Uniform bins

    #     # Original Data Histogram for Dimension i
    #     axes[i].hist(
    #         P[:, i],
    #         bins=bins,
    #         density=True,
    #         alpha=0.5,
    #         label=f"Original P - Dim {i+1}",
    #         color="blue",
    #     )

    #     # Resampled Data Histogram for Dimension i
    #     axes[i].hist(
    #         samples_p[i, :],
    #         bins=bins,
    #         density=True,
    #         alpha=0.5,
    #         label=f"Resampled P - Dim {i+1}",
    #         color="red",
    #     )

    #     axes[i].set_title(f"Histogram of P - Dimension {i+1}")
    #     axes[i].legend()

    # plt.tight_layout()
    # plt.savefig("KDE_fit.png")
    # plt.close()

    # The bandwidth in each dimension is the factor times the standard deviation of the data in that dimension
    # actual_bandwidths = bandwidth_factor * np.std(P, axis=0)

    # print("Bandwidth factor:", bandwidth_factor)
    # print("Actual bandwidths used in each dimension of P:", actual_bandwidths)
    # kde_p.set_bandwidth(bw_method=kde_p.factor * 2.0)  # Adjust the factor as needed

    # Calculate KL divergences
    kl_PM = emp_KL_divergence(samples_p.T, samples_m.T)
    kl_QM = emp_KL_divergence(samples_q.T, samples_m.T)

    # Calculate the Jensen-Shannon Divergence
    jsd = 0.5 * (kl_PM + kl_QM)

    return jsd


class Similarity:
    """Calculate similarity vs performance or vs evaluation reliability"""

    def __init__(self, dataset, performance=True, train=False, n_iter=1):
        """Initializes the similarity calculation

        Args:
            dataset (str): Dataset name
            performance (bool, optional): Sets the analysis to performance or evaluation reliability. Defaults to True.
            train (bool, optional): Performance vs train (D_pool) or val set. Defaults to False.
            n_iter (int, optiona): AL Iteration number
        """
        self.n_iter = n_iter
        if performance:
            dataset_title = "_val"
        else:
            dataset_title = "_splitval"
        if "KITTI" in dataset or "kCODA" in dataset:
            self.methods = [
                "Method 0",
                "Method 1",
                "Method 2",
                "Method 3",
                "Method 4",
                "Method 5",
                "Method 6",
                "Method 7",
                "Method 8",
                "Method 9",
            ]
            self.model_paths = [
                DPATH
                + "/models/trained_models/AL/KITTI_M0_"
                + str(self.n_iter)
                + "_seed21_V0/",
                DPATH
                + "/models/trained_models/AL/KITTI_M1_"
                + str(self.n_iter)
                + "_seed21_V0/",
                DPATH
                + "/models/trained_models/AL/KITTI_M2_"
                + str(self.n_iter)
                + "_seed21_V0/",
                DPATH
                + "/models/trained_models/AL/KITTI_M3_"
                + str(self.n_iter)
                + "_seed21_V0/",
                DPATH
                + "/models/trained_models/AL/KITTI_M4_"
                + str(self.n_iter)
                + "_eval_V0/",
                DPATH
                + "/models/trained_models/AL/KITTI_M5_"
                + str(self.n_iter)
                + "_eval_V0/",
                DPATH
                + "/models/trained_models/AL/KITTI_M6_"
                + str(self.n_iter)
                + "_seed0_V0/",
                DPATH
                + "/models/trained_models/AL/KITTI_M7_"
                + str(self.n_iter)
                + "_seed0_V0/",
                DPATH
                + "/models/trained_models/AL/KITTI_M8_"
                + str(self.n_iter)
                + "_seed21_V0/",
                DPATH
                + "/models/trained_models/AL/KITTI_M9_"
                + str(self.n_iter)
                + "_seed21_V0/",
            ]
            if performance:
                self.tfs = [
                    DPATH
                    + "/datasets/KITTI/tf_active/M0/V0/_train_"
                    + str(self.n_iter)
                    + "_seed21_V0.tfrecord",
                    DPATH
                    + "/datasets/KITTI/tf_active/M1/V0/_train_"
                    + str(self.n_iter)
                    + "_seed21_V0.tfrecord",
                    DPATH
                    + "/datasets/KITTI/tf_active/M2/V0/_train_"
                    + str(self.n_iter)
                    + "_seed21_V0.tfrecord",
                    DPATH
                    + "/datasets/KITTI/tf_active/M3/V0/_train_"
                    + str(self.n_iter)
                    + "_seed21_V0.tfrecord",
                    DPATH
                    + "/datasets/KITTI/tf_active/M4/V0/_train_"
                    + str(self.n_iter)
                    + "_eval_V0.tfrecord",
                    DPATH
                    + "/datasets/KITTI/tf_active/M5/V0/_train_"
                    + str(self.n_iter)
                    + "_eval_V0.tfrecord",
                    DPATH
                    + "/datasets/KITTI/tf_active/M6/V0/_train_"
                    + str(self.n_iter)
                    + "_seed0_V0.tfrecord",
                    DPATH
                    + "/datasets/KITTI/tf_active/M7/V0/_train_"
                    + str(self.n_iter)
                    + "_seed0_V0.tfrecord",
                    DPATH
                    + "/datasets/KITTI/tf_active/M8/V0/_train_"
                    + str(self.n_iter)
                    + "_seed21_V0.tfrecord",
                    DPATH
                    + "/datasets/KITTI/tf_active/M9/V0/_train_"
                    + str(self.n_iter)
                    + "_seed21_V0.tfrecord",
                    DPATH + "/datasets/KITTI/tf/_val.tfrecord",
                ]
            else:
                self.tfs = [
                    DPATH + "/datasets/KITTI/tf/_val.tfrecord",
                    DPATH + "/datasets/CODA/tf/_val.tfrecord",
                ]
            self.classes = self.orig_classes = [
                "car",
                "van",
                "truck",
                "pedestrian",
                "person_sitting",
                "cyclist",
                "tram",
            ]
            self.coda_classes = ["car", "truck", "pedestrian", "cyclist"]
            self.get_dataset_tfcreator = kitti_active_tfrecords
            self.data_dir = DPATH + "/datasets/KITTI/training/"
            self.tf_active_path = DPATH + "/datasets/" + dataset + "/tf_val_split/"
            self.label_path = DPATH + "/datasets/KITTI/kitti.pbtxt"
            self.kitti_val_indices = []
            with open(DPATH + "/datasets/KITTI/vaL_index_list.txt", "r") as file:
                for line in file:
                    self.kitti_val_indices.append(int(line.strip()))
            self.available_paths = np.asarray(
                sorted(os.listdir(DPATH + "/datasets/KITTI/training/image_2/"))
            )[self.kitti_val_indices]
            self.eval_ds = "ks"
            self.eval_ds_ind = 4

            if "bKITTI" in dataset:
                self.tfs[-1] = DPATH + "/datasets/BDD100K/tf/_val100k.tfrecord"
                self.classes = ["pedestrian"]
            elif "kCODA" in dataset:
                if performance:
                    self.tfs[-1] = DPATH + "/datasets/CODA/tf/_val.tfrecord"
                else:
                    self.tfs = [
                        DPATH + "/datasets/CODA/tf/_val.tfrecord",
                        DPATH + "/datasets/KITTI/tf/_val.tfrecord",
                    ]
                self.classes = self.coda_classes
                self.label_path = DPATH + "/datasets/CODA/coda_data.txt"
                self.eval_ds = "cks"
                self.eval_ds_ind = 7
                classes_to_use = extract_class_info("KITTI")[0]

            if train and performance:
                self.tfs[-1] = DPATH + "/datasets/KITTI/tf/_train.tfrecord"
                dataset_title = "_train"
        elif dataset == "BDD" or dataset == "bCODA":
            self.methods = [
                "Method 0",
                "Method 1",
                "Method 2",
                "Method 3",
                "Method 4",
                "Method 5",
                "Method 6",
                "Method 7",
                "Method 8",
                "Method 9",
            ]
            self.model_paths = [
                DPATH
                + "/models/trained_models/AL/BDD100K_M0_"
                + str(self.n_iter)
                + "_seed21_V0/",
                DPATH
                + "/models/trained_models/AL/BDD100K_M1_"
                + str(self.n_iter)
                + "_seed21_V0/",
                DPATH
                + "/models/trained_models/AL/BDD100K_M2_"
                + str(self.n_iter)
                + "_seed21_V0/",
                DPATH
                + "/models/trained_models/AL/BDD100K_M3_"
                + str(self.n_iter)
                + "_seed21_V0/",
                DPATH
                + "/models/trained_models/AL/BDD100K_M4_"
                + str(self.n_iter)
                + "_seed21_V0/",
                DPATH
                + "/models/trained_models/AL/BDD100K_M5_"
                + str(self.n_iter)
                + "_seed21_V0/",
                DPATH
                + "/models/trained_models/AL/BDD100K_M6_"
                + str(self.n_iter)
                + "_seed21_V0/",
                DPATH
                + "/models/trained_models/AL/BDD100K_M7_"
                + str(self.n_iter)
                + "_seed21_V0/",
                DPATH
                + "/models/trained_models/AL/BDD100K_M8_"
                + str(self.n_iter)
                + "_seed21_V0/",
                DPATH
                + "/models/trained_models/AL/BDD100K_M9_"
                + str(self.n_iter)
                + "_seed21_V0/",
            ]
            if performance:
                self.tfs = [
                    DPATH
                    + "/datasets/BDD100K/tf_active/M0/V0/_train_"
                    + str(self.n_iter)
                    + "_seed21_V0.tfrecord",
                    DPATH
                    + "/datasets/BDD100K/tf_active/M1/V0/_train_"
                    + str(self.n_iter)
                    + "_seed21_V0.tfrecord",
                    DPATH
                    + "/datasets/BDD100K/tf_active/M2/V0/_train_"
                    + str(self.n_iter)
                    + "_seed21_V0.tfrecord",
                    DPATH
                    + "/datasets/BDD100K/tf_active/M3/V0/_train_"
                    + str(self.n_iter)
                    + "_seed21_V0.tfrecord",
                    DPATH
                    + "/datasets/BDD100K/tf_active/M4/V0/_train_"
                    + str(self.n_iter)
                    + "_seed21_V0.tfrecord",
                    DPATH
                    + "/datasets/BDD100K/tf_active/M5/V0/_train_"
                    + str(self.n_iter)
                    + "_seed21_V0.tfrecord",
                    DPATH
                    + "/datasets/BDD100K/tf_active/M6/V0/_train_"
                    + str(self.n_iter)
                    + "_seed21_V0.tfrecord",
                    DPATH
                    + "/datasets/BDD100K/tf_active/M7/V0/_train_"
                    + str(self.n_iter)
                    + "_seed21_V0.tfrecord",
                    DPATH
                    + "/datasets/BDD100K/tf_active/M8/V0/_train_"
                    + str(self.n_iter)
                    + "_seed21_V0.tfrecord",
                    DPATH
                    + "/datasets/BDD100K/tf_active/M9/V0/_train_"
                    + str(self.n_iter)
                    + "_seed21_V0.tfrecord",
                    DPATH + "/datasets/BDD100K/tf/_val100k.tfrecord",
                ]
            else:
                self.tfs = [
                    DPATH + "/datasets/BDD100K/tf/_val100k.tfrecord",
                    DPATH + "/datasets/CODA/tf_BDD/_val.tfrecord",
                ]
            self.classes = self.orig_classes = [
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
            self.coda_classes = [
                "pedestrian",
                "car",
                "truck",
                "bus",
                "motorcycle",
                "bicycle",
                "traffic light",
                "traffic sign",
            ]
            self.get_dataset_tfcreator = bdd_active_tfrecords
            self.data_dir = DPATH + "/datasets/BDD100K/bdd100k/"
            self.tf_active_path = DPATH + "/datasets/" + dataset + "100K/tf_val_split/"
            self.label_path = DPATH + "/datasets/BDD100K/bdd.pbtxt"
            self.available_paths = sorted(
                os.listdir(DPATH + "/datasets/BDD100K/bdd100k/images/100k/val/")
            )
            self.eval_ds = "bs"
            self.eval_ds_ind = 5

            if "bCODA" in dataset:
                if performance:
                    self.tfs[-1] = DPATH + "/datasets/CODA/tf_BDD/_val.tfrecord"
                else:
                    self.tfs = [
                        DPATH + "/datasets/CODA/tf_BDD/_val.tfrecord",
                        DPATH + "/datasets/BDD100K/tf/_val100k.tfrecord",
                    ]
                self.classes = self.coda_classes
                self.label_path = DPATH + "/datasets/CODA/coda_data_BDD.txt"
                self.eval_ds = "cbs"
                self.eval_ds_ind = 6
                classes_to_use = extract_class_info("BDD")[0]
            if train and performance:
                self.tfs[-1] = DPATH + "/datasets/BDD100K/tf/_train100k.tfrecord"
                dataset_title = "_train"
        if "CODA" in dataset:
            self.get_dataset_tfcreator = coda_active_tfrecords
            self.data_dir = DPATH + "/datasets/CODA/"
            self.tf_active_path = DPATH + "/datasets/CODA/tf_val_split/" + dataset + "/"
            with open(os.path.join(self.data_dir, "annotations.json")) as file:
                img_anno = json.load(file)["annotations"]
            result = []
            current_image_id = None
            current_annotations = []
            for annotation in img_anno:
                image_id = annotation["image_id"]
                if image_id != current_image_id:
                    if current_annotations:
                        result.append(current_annotations)
                    current_annotations = []
                    current_image_id = image_id
                current_annotations.append(annotation)
            if current_annotations:
                result.append(current_annotations)
            self.coda_val_indices = []
            for img_num, _ in enumerate(result):
                annotation_for_image = discard_classes(result[img_num], classes_to_use)
                if len(annotation_for_image) != 0:
                    self.coda_val_indices.append(img_num)
            self.available_paths = np.asarray(
                sorted(os.listdir(DPATH + "/datasets/CODA/images/"))
            )[self.coda_val_indices]

        self.dataset = dataset + dataset_title
        print(self.dataset)
        if performance:
            if not os.path.exists(GPATH + "/misc/papers/paper_al/" + self.dataset):
                os.makedirs(GPATH + "/misc/papers/paper_al/" + self.dataset)
        else:
            if not os.path.exists(self.tf_active_path):
                os.makedirs(self.tf_active_path)
            if not os.path.exists(
                GPATH + "/misc/papers/paper_al/" + self.dataset + "/plots"
            ):
                os.makedirs(GPATH + "/misc/papers/paper_al/" + self.dataset + "/plots")
        self.performance = performance

    @staticmethod
    def calculate_set_similarity(
        crops_metrics_perc, classes, methods, return_perclass=False
    ):
        """Calculates similarity between sets

        Args:
            crops_metrics_perc (list): List containing the metrics over the crops for each class
            classes (list): List with class names
            methods (list): List with AL methods names = set names
            return_perclass (bool, optional): Returns the similarity per class as well. Defaults to False.

        Returns:
            The AL methodand their set similarity to the reference set, the class weighting activation status, the similarity per class
        """
        jsds, class_ratio = [], []
        for cl in classes:
            jsds_temp, class_ratio_temp = [], []
            val_data = np.asarray(crops_metrics_perc[-1][cl])
            for i in range(len(crops_metrics_perc) - 1):
                if len(crops_metrics_perc[i][cl]) > 0:
                    iter_data = np.asarray(crops_metrics_perc[i][cl])
                    class_ratio_temp.append(
                        len(crops_metrics_perc[-1][cl][0])
                        / len(crops_metrics_perc[i][cl][0])
                    )
                    jsdiv = empirical_jensen_shannon_divergence(
                        iter_data.T, val_data.T, num_samples=10000
                    )
                    jsds_temp.append(jsdiv)
                else:
                    class_ratio_temp.append(np.nan)
                    jsds_temp.append(np.nan)
            class_ratio.append(class_ratio_temp)
            jsds.append(jsds_temp)
        total_dets = [
            np.sum([len(dist[cl][0]) if len(dist[cl]) > 0 else 0 for cl in classes])
            for dist in crops_metrics_perc[:-1]
        ]
        class_weights = np.mean(
            [
                (
                    [
                        (
                            len(crops_metrics_perc[i][cl][0])
                            if len(crops_metrics_perc[i][cl]) > 0
                            else 0
                        )
                        for i in range(len(crops_metrics_perc) - 1)
                    ]
                )
                / np.asarray(total_dets)
                for cl in classes
            ],
            axis=-1,
        )  # Average detections per class
        classes_low_dets = class_weights < np.percentile(class_weights, 25)
        class_weights = 1 / class_weights
        activate_class_weight = (
            np.round(np.nanstd(class_weights) / np.nanmean(class_weights), 1) > 1.3
        )  #  Too much class proprtion variation
        if activate_class_weight:
            class_weights[classes_low_dets] = 0
            print("Class Balance Activated")
        else:
            class_weights = np.ones_like(class_weights)
        beta = np.maximum(
            1, np.asarray(total_dets / np.percentile(total_dets, 25), dtype="int")
        )
        combined_metrics_perc = []
        for c in range(len(classes)):
            metrics = np.add(jsds[c], 0.25 * (class_ratio[c] * beta) + 0.5)
            # metrics = np.add(jsds[c], class_ratio[c] * beta)
            metrics[np.isinf(metrics)] = np.nan
            combined_metrics_perc.append(metrics)

        sim = np.nansum(
            1 / np.asarray(combined_metrics_perc) * class_weights.reshape([-1, 1]),
            axis=0,
        ) / np.sum(class_weights)
        methods_sim = {methods[i]: sim[i] for i in range(len(methods))}
        sorted_methods_sim = sorted(methods_sim.items(), key=lambda x: x[1])
        if not return_perclass:
            combined_metrics_perc = None
        return sorted_methods_sim, activate_class_weight, combined_metrics_perc

    @staticmethod
    def collect_metrics(tfpath, classes):
        """Extacts images and ground truth, then crops the images, then calculates metrics on the crops

        Args:
            tfpath (str): Path to TFRecord
            classes (list): List with classes names

        Returns:
            List of the metrics per class
        """
        dataset = tf.data.TFRecordDataset(tfpath)
        metrcs_class = {cl: [] for cl in classes}
        for record in dataset:
            example = tf.train.Example()
            example.ParseFromString(record.numpy())

            encoded_image = example.features.feature["image/encoded"].bytes_list.value[
                0
            ]
            image = tf.image.decode_image(encoded_image).numpy()

            width = int(image.shape[1])
            height = int(image.shape[0])
            box_keys = [
                "image/object/bbox/ymin",
                "image/object/bbox/xmin",
                "image/object/bbox/ymax",
                "image/object/bbox/xmax",
            ]
            norm = [float(height), float(width), float(height), float(width)]
            boxes = [
                np.asarray(example.features.feature[bbox_key].float_list.value)
                * norm[x]
                for x, bbox_key in enumerate(box_keys)
            ]
            boxes_combined = []
            for i in range(0, len(boxes[0])):
                box_comb = [boxes[j][i] for j in range(len(boxes))]
                boxes_combined.append(box_comb)
            label = example.features.feature["image/object/class/text"].bytes_list.value
            labels = [x.decode("utf-8") for x in label]
            for class_name in np.unique(labels):
                if class_name in classes:
                    cls_inds = np.where(np.asarray(labels) == class_name)[0]
                    if len(cls_inds) > 0:
                        boxes_cls = [np.asarray(boxes_combined[i]) for i in cls_inds]
                        crops = []
                        for j in range(len(boxes_cls)):
                            y1, x1, y2, x2 = map(int, boxes_cls[j])
                            cropped_img = image[y1:y2, x1:x2, :]
                            # Check if the crop is not empty
                            if np.min([cropped_img.shape[0], cropped_img.shape[1]]) > 2:
                                crops.append(cropped_img)
                        if len(crops) > 0:  # Calculate metrics
                            aratio_tf1 = [
                                image.shape[1] / image.shape[0] for image in crops
                            ]
                            avg_dct_tf1 = [
                                np.mean(
                                    dct(
                                        dct(np.array(image), axis=0, norm="ortho"),
                                        axis=1,
                                        norm="ortho",
                                    )
                                )
                                for image in crops
                            ]
                            avg_hist_tf1 = [
                                np.mean(
                                    cv2.calcHist(
                                        image,
                                        [0, 1, 2],
                                        None,
                                        [8, 8, 8],
                                        [0, 256, 0, 256, 0, 256],
                                    ).flatten()
                                )
                                for image in crops
                            ]
                            metrcs_class[class_name].append(
                                [aratio_tf1, avg_dct_tf1, avg_hist_tf1]
                            )
        # Concatenate over all crops and combine
        for cl in classes:
            if len(metrcs_class[cl]) > 0:
                aratio_tf1 = np.concatenate([det[0] for det in metrcs_class[cl]])
                avg_dct_tf1 = np.concatenate([det[1] for det in metrcs_class[cl]])
                avg_hist_tf1 = np.nan_to_num(
                    np.concatenate([det[2] for det in metrcs_class[cl]]), nan=1
                )
                metrcs_class[cl] = [aratio_tf1, avg_dct_tf1, avg_hist_tf1]
        return metrcs_class

    def eval_metrics_perclass(self, crops_metrics_perc, ap_cls, sim_perc):
        """Plot similarity vs per class AP and calculate ranking correlation based on similarity and per class AP

        Args:
            crops_metrics_perc (list): List of per class metrics
            ap_cls (list): List of per class AP
            sim_perc (_type_): List of per class similarity
        """
        n_c = len(self.classes)
        sim_rank_perc = [
            list(np.asarray(self.methods)[np.argsort(sim_perc[i])])
            for i in range(len(sim_perc))
        ]
        method_rank_mapping = {method: rank for rank, method in enumerate(self.methods)}
        kend = []
        for i in range(n_c):
            methods_ypred = [method_rank_mapping[method] for method in sim_rank_perc[i]]
            methods_ytrue = [
                method_rank_mapping[method]
                for method in [self.methods[j] for j in np.argsort(1 / ap_cls[:, i])]
            ]
            kend.append(kendalltau(methods_ytrue, methods_ypred)[0])
        average_kend = round(np.mean(kend), 3)
        rounded_kend = [round(value, 3) for value in kend]
        ex1_positional_kend = kendalltau([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0])[0]
        ex2_positional_kend = kendalltau([0, 1, 2, 3, 4, 5], [2, 3, 5, 1, 0, 4])[0]
        ex3_positional_kend = kendalltau([0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0])[0]
        ex4_positional_kend = kendalltau([0, 1, 2, 3, 4, 5], [0, 1, 3, 2, 4, 5])[0]

        # Save to text file
        with open(
            GPATH
            + "/misc/papers/paper_al/"
            + self.dataset
            + "/sorting_perc_metrics"
            + str(self.n_iter)
            + ".txt",
            "w",
        ) as file:
            file.write("Kendall's Tau correlation coefficient values:\n")
            file.write(", ".join(map(str, rounded_kend)) + "\n")
            file.write(
                f"Average Kendall's Tau correlation coefficient (with best=1): {average_kend}\n\n"
            )

            file.write(
                f"Example shift [0,1,2,3,4,5]/[1,2,3,4,5,0]: {np.round(ex1_positional_kend,3)}\n"
            )
            file.write(
                f"Example mix [0,1,2,3,4,5]/[2,3,5,1,0,4]: {np.round(ex2_positional_kend,3)}\n"
            )
            file.write(
                f"Example reverse [0,1,2,3,4,5]/[5,4,3,2,1,0]: {np.round(ex3_positional_kend,3)}\n"
            )
            file.write(
                f"Example reverse [0,1,2,3,4,5]/[0,1,3,2,4,5]: {np.round(ex4_positional_kend,3)}\n\n"
            )

        total_dets = [
            np.sum([len(dtf[cl][0]) if len(dtf[cl]) > 0 else 0 for cl in self.classes])
            for dtf in crops_metrics_perc[:-1]
        ]
        class_proportion = np.round(
            np.mean(
                [
                    (
                        [
                            (
                                len(crops_metrics_perc[i][cl][0])
                                if len(crops_metrics_perc[i][cl]) > 0
                                else 0
                            )
                            for i in range(len(crops_metrics_perc) - 1)
                        ]
                    )
                    / np.asarray(total_dets)
                    for cl in self.classes
                ],
                axis=-1,
            )
            * 100,
            1,
        )
        num_subplots, _ = ap_cls.T.shape
        num_rows = int(np.ceil(np.sqrt(num_subplots)))
        num_cols = int(np.ceil(num_subplots / num_rows))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, 5))
        axes = axes.flatten()
        # Sort indices based on perc_dets

        sorted_indices = np.argsort(class_proportion)
        nonzero_indices = [idx for idx in sorted_indices if class_proportion[idx] != 0]
        for i, idx in enumerate(nonzero_indices):
            y_values = 1 / np.asarray(sim_perc[idx])
            x_values = ap_cls.T[idx] * 100

            sns.regplot(
                x=x_values,
                y=y_values,
                ax=axes[i],
                ci=95,
                color="#1f77b4",
                line_kws={"color": "red"},
            )
            pearson_corr, pearson_pval = pearsonr(x_values, y_values)
            # spearman_corr, spearman_pval = spearmanr(x_values, y_values)
            axes[i].text(
                0.05,
                0.9,
                f"r={pearson_corr:.2f}\np={pearson_pval:.4f}",
                transform=axes[i].transAxes,
                verticalalignment="top",
                fontsize=9,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="white",
                    alpha=0.6,
                ),
            )
            axes[i].set_title(
                self.classes[idx].title() + ", " + str(class_proportion[idx]) + "%"
            )
            axes[i].set_ylabel("Similarity")
            axes[i].set_xlabel("AP")
            axes[i].grid(True)

        # Remove excess subplots
        for j in range(num_rows * num_cols - len(nonzero_indices)):
            fig.delaxes(axes[len(nonzero_indices) + j])
        plt.tight_layout()
        plt.savefig(
            GPATH
            + "/misc/papers/paper_al/"
            + self.dataset
            + "/sorting_perc_curve"
            + str(self.n_iter)
            + ".png"
        )
        plt.savefig(
            GPATH
            + "/misc/papers/paper_al/"
            + self.dataset
            + "/sorting_perc_curve"
            + str(self.n_iter)
            + ".svg"
        )
        plt.close()

    def _generate_and_save_table(self, table_data, title, act_class_weight):
        """Save table with sorted AL methods based on similarity"""
        fig, ax = plt.subplots()
        table = ax.table(
            cellText=table_data,
            colLabels=["Method", title],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        table.auto_set_column_width([0, 1])
        ax.axis("off")
        plt.title("Class weight: " + str(act_class_weight))
        plt.tight_layout()
        plt.savefig(
            GPATH
            + "/misc/papers/paper_al/"
            + self.dataset
            + "/table_rank_"
            + title
            + ".png"
        )
        plt.close()

    def similarity_vs_performance(self):
        """Calculate similarity vs performance"""
        # Per class analysis
        if "KITTI" in self.dataset:
            ap_cls = np.asarray(read_aps_from_log(self.model_paths, "KITTI_eval"))[
                :, -len(self.classes) :
            ]
        elif "BDD" in self.dataset:
            ap_cls = np.asarray(read_aps_from_log(self.model_paths, "BDD_eval"))[
                :, -len(self.classes) :
            ]
        else:
            ap_cls = np.asarray(read_aps_from_log(self.model_paths, "CODA_eval"))[
                :,
                [
                    -len(self.orig_classes) + i
                    for i in range(len(self.orig_classes))
                    if self.orig_classes[i] in self.classes
                ],
            ]

        crops_metrics_perc = [
            self.collect_metrics(tfpath, self.classes) for tfpath in self.tfs
        ]
        sim, act_class_weight, sim_perclass = self.calculate_set_similarity(
            crops_metrics_perc, self.classes, self.methods, return_perclass=True
        )
        self.eval_metrics_perclass(crops_metrics_perc, ap_cls, sim_perclass)

        # Plot similarity vs performance
        methods_sim = dict(sim)
        sorted_methods_sim = sorted(methods_sim.items(), key=lambda x: x[1])
        methods_mean_ap = {
            method: values
            for method, values in list(zip(self.methods, np.mean(ap_cls, axis=-1)))
        }
        sorted_methods_ap = sorted(methods_mean_ap.items(), key=lambda x: x[1])
        y_sim = [methods_sim[m] for m in np.asarray(sorted_methods_ap)[:, 0]]
        y_ap = np.asarray(sorted_methods_ap)[:, 1].astype(float) * 100
        np.save(
            GPATH
            + "/misc/papers/paper_al/"
            + self.dataset
            + "/"
            + self.dataset
            + "_compare_simvsap_slope"
            + str(self.n_iter)
            + ".npy",
            np.stack([y_ap, y_sim], axis=-1),
        )

        # Plot similarity vs mAP curves
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.scatter(
            np.asarray(sorted_methods_ap)[:, 0], y_ap, color="b", marker="1", label="AP"
        )
        ax1.set_ylabel("mAP", color="b")
        ax1.tick_params("y", colors="b")
        ax1.set_xticklabels(np.asarray(sorted_methods_ap)[:, 0], rotation=90)

        ax2 = ax1.twinx()
        ax2.scatter(
            np.asarray(sorted_methods_ap)[:, 0],
            y_sim,
            color="r",
            marker="+",
            label="Sim",
        )
        ax2.set_ylabel("Similarity", color="r")
        ax2.tick_params("y", colors="r")

        x_values = np.arange(len(self.methods))
        coefficients = np.polyfit(x_values, y_ap, 3)
        curve_values = np.polyval(coefficients, x_values)
        ax1.plot(x_values, curve_values, color="b")

        x_values = np.arange(len(self.methods))
        coefficients = np.polyfit(x_values, y_sim, 3)
        curve_values = np.polyval(coefficients, x_values)
        ax2.plot(x_values, curve_values, color="r")
        plt.tight_layout()
        plt.title("Class weight: " + str(act_class_weight))
        plt.savefig(
            GPATH
            + "/misc/papers/paper_al/"
            + self.dataset
            + "/compare_simvsap"
            + str(self.n_iter)
            + ".png"
        )
        plt.close()

        # Plot similarity vs mAP in one curve
        fig = plt.figure(figsize=(5, 4))
        for m in range(len(np.asarray(sorted_methods_ap)[:, 0])):
            plt.scatter(y_ap[m], y_sim[m], label=np.asarray(sorted_methods_ap)[:, 0][m])
        coeffs = np.polyfit(y_ap, y_sim, 1)
        poly_func = np.poly1d(coeffs)
        x_values = np.linspace(min(y_ap), max(y_ap), 100)
        plt.plot(x_values, poly_func(x_values), color="black", label="Slope")
        plt.legend()
        plt.xlabel("mAP")
        plt.ylabel("Similarity")
        plt.title("Class weight: " + str(act_class_weight))
        plt.savefig(
            GPATH
            + "/misc/papers/paper_al/"
            + self.dataset
            + "/compare_simvsap_slope"
            + str(self.n_iter)
            + ".png"
        )
        plt.close()
        table_data_sim = [
            (method, np.round(val, 3)) for method, val in sorted_methods_sim
        ]
        table_data_ap = [
            (method, np.round(val, 3)) for method, val in sorted_methods_ap
        ]
        self._generate_and_save_table(table_data_sim, "Similarity", act_class_weight)
        self._generate_and_save_table(table_data_ap, "mAP", act_class_weight)
        return y_ap, y_sim

    def _subset_eval(
        self,
        selected_subset_names,
        selected_title,
        selected_data,
        selected_classes,
        selected_ref_ap,
        selected_comp_ap,
        save_title,
    ):
        """Evaluates similarity vs performance matching then checks if subsets available with better performance matching to reference set

        Args:
            selected_subset_names (list): List with subset names, for example subset1,subset2, ...
            selected_title (str): Val or test set
            selected_data (list): List with per clas crop metrics
            selected_classes (list): List with classes names
            selected_ref_ap (list): List with mAP on reference set
            selected_comp_ap (list): List with mAP on comparison set/subsets
            save_title (str): Title to save results under in the path
        """

        if len(selected_subset_names) <= 10:
            colors = plt.cm.tab10.colors
        elif 10 < len(selected_subset_names) <= 20:
            colors = plt.cm.tab20.colors
        else:
            colors = np.concatenate(
                [plt.cm.tab20.colors] * int(len(selected_subset_names) / 20 + 1)
            )

        method_rank_mapping = {method: rank for rank, method in enumerate(self.methods)}
        kend = []
        for i in range(len(selected_subset_names)):
            ypred_methods = np.asarray(self.methods)[
                np.argsort(1 / selected_comp_ap[i])
            ]
            ytrue_methods = np.asarray(self.methods)[np.argsort(1 / selected_ref_ap)]
            methods_ypred = [method_rank_mapping[method] for method in ypred_methods]
            methods_ytrue = [method_rank_mapping[method] for method in ytrue_methods]
            kendall_tau_corr, _ = kendalltau(methods_ytrue, methods_ypred)
            kend.append(kendall_tau_corr)
        kend = np.asarray(kend)
        method_colors = plt.cm.get_cmap("tab10", len(self.methods))(
            np.arange(len(self.methods))
        )
        custom_cmap = ListedColormap(method_colors)

        # Plot ranking with colors on each comparison and reference set
        plt.figure(figsize=(int(len(selected_subset_names) / 8 + 7), 4))
        ranks_plot = np.swapaxes(
            np.argsort(
                np.argsort(
                    np.concatenate(
                        (selected_comp_ap, selected_ref_ap.reshape([1, -1])), axis=0
                    ),
                    axis=-1,
                ),
                axis=-1,
            ),
            0,
            1,
        )
        np.save(
            GPATH
            + "/misc/papers/paper_al/"
            + self.dataset
            + "/plots/"
            + self.dataset
            + "_"
            + save_title
            + "_ranks.npy",
            ranks_plot,
        )
        plt.imshow(ranks_plot, cmap=custom_cmap, aspect="auto")
        [
            plt.plot(
                [i - 0.5] * (len(self.methods) + 1),
                np.arange(len(self.methods) + 1) - 0.5,
                color="black",
            )
            for i in range(len(selected_subset_names) + 1)
        ]
        [
            plt.plot(
                np.arange(len(selected_subset_names) + 2) - 0.5,
                [i - 0.5] * (len(selected_subset_names) + 2),
                color="black",
            )
            for i in range(len(self.methods))
        ]
        plt.xlabel("Subsets")
        plt.yticks(np.arange(len(self.methods)), self.methods)
        selected_subset_names[:100] = [
            "Subset " + str(i) for i in np.arange(len(selected_subset_names[:100]))
        ]
        plt.xticks(
            np.arange(len(selected_subset_names) + 1),
            selected_subset_names + [selected_title],
            rotation="vertical",
        )
        plt.colorbar(label="mAP-based Rank")
        plt.tight_layout()
        plt.savefig(
            GPATH
            + "/misc/papers/paper_al/"
            + self.dataset
            + "/plots/"
            + save_title
            + "_rank_persubset.png"
        )
        plt.close()

        # Calculate similarity
        sim, act_class_weight, _ = self.calculate_set_similarity(
            selected_data, selected_classes, selected_subset_names
        )
        sim = np.asarray([dict(sim)[cluster] for cluster in selected_subset_names])
        np.save(
            GPATH
            + "/misc/papers/paper_al/"
            + self.dataset
            + "/plots/"
            + self.dataset
            + "_"
            + save_title
            + "_kend_sim.npy",
            np.stack([kend, sim], axis=-1),
        )

        # Plot simlarity vs performance matching
        fig, axs = plt.subplots(figsize=(6, 5))
        for i, cluster in enumerate(selected_subset_names):
            axs.scatter(kend[i], sim[i], color=colors[i], marker="o", s=100)
        axs.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        coeffs = np.polyfit(kend, sim, 1)
        poly_func = np.poly1d(coeffs)
        x_values = np.linspace(min(kend), max(kend), 100)
        axs.plot(x_values, poly_func(x_values), color="black", label="Slope")
        if "test" in save_title:
            axs.scatter(
                kend[-1],
                sim[-1],
                label="Orig. Val Set",
                color="black",
                marker="o",
                s=100,
            )

        plt.ylabel("Similarity")
        plt.xlabel("Kendall's Tau")
        plt.legend()
        plt.title("Class weight: " + str(act_class_weight))
        plt.tight_layout()
        plt.savefig(
            GPATH
            + "/misc/papers/paper_al/"
            + self.dataset
            + "/plots/"
            + save_title
            + "_scatter_subsets.png"
        )
        plt.close()

        # Find subsets with higher similarity to reference set
        if "test" in save_title:
            sorted_indices = np.argsort(sim)[::-1]
            top_s = sorted_indices[0]
            methods_ap = [selected_comp_ap[-1], selected_comp_ap[top_s]]
            methods_name = ["Orig. Val. Set", "Top-s"]
            method_rank_mapping = {
                method: rank for rank, method in enumerate(self.methods)
            }
            kendal_values = []
            for i in range(len(methods_name)):
                aps = methods_ap[i]
                ypred_methods = np.asarray(self.methods)[np.argsort(1 / aps)]
                ytrue_methods = np.asarray(self.methods)[
                    np.argsort(1 / selected_ref_ap)
                ]
                methods_ypred = [
                    method_rank_mapping[method] for method in ypred_methods
                ]
                methods_ytrue = [
                    method_rank_mapping[method] for method in ytrue_methods
                ]
                kendal_values.append(kendalltau(methods_ytrue, methods_ypred)[0])
            np.save(
                GPATH
                + "/misc/papers/paper_al/"
                + self.dataset
                + "/plots/"
                + self.dataset
                + "_"
                + save_title
                + "_subset_kend.npy",
                kendal_values,
            )
            np.save(
                GPATH
                + "/misc/papers/paper_al/"
                + self.dataset
                + "/plots/"
                + self.dataset
                + "_"
                + save_title
                + "_val_test_sub_ap.npy",
                np.stack(
                    [
                        selected_ref_ap,
                        selected_comp_ap[-1],
                        selected_comp_ap[top_s],
                        selected_comp_ap[np.argmax(kend)],
                    ]
                ),
            )

            plt.figure(figsize=(5, 5))
            plt.bar(methods_name, kendal_values, color="skyblue")
            plt.xlabel("Eval Method")
            plt.xticks(np.arange(len(methods_name)), methods_name, rotation="vertical")
            plt.ylabel("Kendall's Tau")
            plt.tight_layout()
            plt.title("Class weight: " + str(act_class_weight))
            plt.savefig(
                GPATH
                + "/misc/papers/paper_al/"
                + self.dataset
                + "/plots/"
                + save_title
                + "_subset_kend.png"
            )
            plt.close()

    def _gen_tfrecord_run_eval(self, train_inds, set_name, tf_train):
        """Generates TFRecord from given indices and runs evaluation on it

        Args:
            train_inds (list): List of training indices
            set_name (str): Subset name
            tf_train (bool): Bool used on the tfrecord generating files under dataset
        """
        if not os.path.exists(
            self.tf_active_path + "_val_set" + set_name + ".tfrecord"
        ):
            self.get_dataset_tfcreator(
                data_dir=self.data_dir,
                output_path=self.tf_active_path,
                classes_to_use=self.classes,
                label_map_path=self.label_path,
                train_indices=train_inds,
                current_iteration="set" + set_name,
                train=tf_train,
            )
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        for model_path in self.model_paths:
            save_folder = (
                GPATH
                + "/misc/papers/paper_al/"
                + self.dataset
                + "/set"
                + set_name
                + "/"
            )
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if not os.path.exists(save_folder + model_path.split("/")[-2]):
                update_eval_config(
                    GPATH + "/configs/eval/eval_" + self.eval_ds + ".yaml",
                    model_path,
                    update_name=set_name,
                    eval_samples=len(train_inds),
                )
                run_eval(
                    save_folder + model_path.split("/")[-2],
                    dataset_number=self.eval_ds_ind,
                )

    def similarity_vs_evaluation(self):
        """Calculate similarity vs evaluation reliability"""
        # Add name to subsets if running multiple experiments to avoid overwriting
        exp_name = "randomtenten"
        split_factor = 10
        len_images = len(extract_tf_data(self.tfs[0])[0])
        if "kCODA" in self.dataset:
            tf_train = True
        else:
            tf_train = False
        # Generate TFRecords and run evaluation
        for x in range(1, 11):
            np.random.seed(x)
            rand_batches = np.random.permutation(np.arange(len_images))
            if "KITTI" in self.dataset:
                new_val_indices = [
                    self.kitti_val_indices[index] for index in rand_batches
                ]
            elif "CODA" in self.dataset:
                new_val_indices = [
                    self.coda_val_indices[index] for index in rand_batches
                ]
            else:
                new_val_indices = list(rand_batches)
            batch_indices = list(
                np.arange(
                    0, len(new_val_indices), int(len(new_val_indices) / split_factor)
                )
            )
            if (
                len(new_val_indices) - 10
                < batch_indices[-1]
                < len(new_val_indices) + 10
            ):
                batch_indices[-1] = len(new_val_indices)
            else:
                batch_indices.append(len(new_val_indices))
            for b in range(1, len(batch_indices)):
                self._gen_tfrecord_run_eval(
                    np.asarray(new_val_indices)[
                        batch_indices[b - 1] : batch_indices[b]
                    ],
                    exp_name + str(b) + "seed" + str(x),
                    tf_train,
                )
            np.random.seed(RANDOM_STATE_SEED)

        sets_names = [
            "Rand " + str(b + 1) + "seed" + str(x)
            for b in range(len(batch_indices) - 1)
            for x in range(1, 11)
        ]
        sets_names.append("Val Set")

        self.tfs = [
            self.tf_active_path
            + "/_val_set"
            + exp_name
            + str(b + 1)
            + "seed"
            + str(x)
            + ".tfrecord"
            for b in range(len(batch_indices) - 1)
            for x in range(1, 11)
        ]
        if "KITTI" in self.dataset:
            self.tfs.append(DPATH + "/datasets/KITTI/tf/_val.tfrecord")
            self.tfs.append(DPATH + "/datasets/CODA/tf/_val.tfrecord")
        elif "CODA" in self.dataset:
            if "b" in self.dataset:
                self.tfs.append(DPATH + "/datasets/CODA/tf_BDD/_val.tfrecord")
                self.tfs.append(DPATH + "/datasets/BDD100K/tf/_val100k.tfrecord")
            else:
                self.tfs.append(DPATH + "/datasets/CODA/tf/_val.tfrecord")
                self.tfs.append(DPATH + "/datasets/KITTI/tf/_val.tfrecord")
        else:
            self.tfs.append(DPATH + "/datasets/BDD100K/tf/_val100k.tfrecord")
            self.tfs.append(DPATH + "/datasets/CODA/tf_BDD/_val.tfrecord")

        ap_sets = [
            read_aps_from_log(
                self.model_paths,
                self.dataset + "/set" + exp_name + str(b + 1) + "seed" + str(x),
            )
            for b in range(len(batch_indices) - 1)
            for x in range(1, 11)
        ]
        if "CODA" in self.dataset:
            ap_sets.append(read_aps_from_log(self.model_paths, "CODA_eval"))
            ap_cls_coda = np.asarray(ap_sets)[:, :, 0]
        else:
            ap_sets.append(
                read_aps_from_log(
                    self.model_paths, self.dataset.split("_")[0] + "_eval"
                )
            )
            ap_sets = np.asarray(ap_sets)
            ap_cls_coda = np.mean(
                ap_sets[:, :, -len(self.classes) :][
                    :,
                    :,
                    [
                        i
                        for i in range(len(self.classes))
                        if self.classes[i] in self.coda_classes
                    ],
                ],
                axis=-1,
            )

        if "CODA" in self.dataset:
            if "b" in self.dataset:
                raw_ap = read_aps_from_log(self.model_paths, "BDD_eval")
            else:
                raw_ap = read_aps_from_log(self.model_paths, "KITTI_eval")
            ref_ap = np.mean(
                np.asarray(raw_ap)[:, -len(self.orig_classes) :][
                    :,
                    [
                        i
                        for i in range(len(self.orig_classes))
                        if self.orig_classes[i] in self.coda_classes
                    ],
                ],
                axis=-1,
            )
        else:
            raw_ap = read_aps_from_log(self.model_paths, "CODA_eval")
            ref_ap = np.asarray(raw_ap)[:, 0]

        crops_metrics_perc = [
            self.collect_metrics(tfpath, self.classes) for tfpath in self.tfs
        ]

        # Test mode
        self._subset_eval(
            sets_names,
            "Test Set",
            crops_metrics_perc,
            self.coda_classes,
            ref_ap,
            ap_cls_coda,
            "test",
        )

        # Val mode
        if "CODA" not in self.dataset:
            self._subset_eval(
                sets_names[:-1],
                "Val Set",
                crops_metrics_perc[:-1],
                self.classes,
                ap_sets[-1, :, 0],
                ap_sets[:-1, :, 0],
                "val",
            )

    def _plot_multiple_iterations(self, y_aps, y_sims):
        """Plots performance correlation curve for multiple AL iterations"""
        y_aps = np.asarray(y_aps)
        y_sims = np.asarray(y_sims)
        np.save(
            GPATH
            + "/misc/papers/paper_al/"
            + self.dataset
            + "/"
            + self.dataset
            + "_multiple_iterations_plot.npy",
            np.stack([y_aps, y_sims], axis=-1),
        )

        num_subplots = len(y_aps)
        proportions = [5, 5, 5, 10, 20, 30, 25]
        num_rows = int(np.ceil(np.sqrt(num_subplots)))
        num_cols = int(np.ceil(num_subplots / num_rows))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, 5))
        axes = axes.flatten()
        for i in range(num_subplots):
            y_values = y_sims[i]
            x_values = y_aps[i]

            sns.regplot(
                x=x_values,
                y=y_values,
                ax=axes[i],
                ci=95,
                color="#1f77b4",
                line_kws={"color": "red"},
            )
            pearson_corr, pearson_pval = pearsonr(x_values, y_values)
            # spearman_corr, spearman_pval = spearmanr(x_values, y_values)
            axes[i].text(
                0.05,
                0.9,
                f"r={pearson_corr:.2f}\np={pearson_pval:.4f}",
                transform=axes[i].transAxes,
                verticalalignment="top",
                fontsize=9,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="white",
                    alpha=0.6,
                ),
            )
            axes[i].set_title(
                "AL Iteration "
                + str(i + 1)
                + ", "
                + str(np.sum(proportions[: i + 2]))
                + "%"
            )
            axes[i].set_ylabel("Similarity")
            axes[i].set_xlabel("mAP")
            axes[i].grid(True)

        # Remove excess subplots
        for j in range(num_rows * num_cols - num_subplots):
            fig.delaxes(axes[num_subplots + j])
        plt.tight_layout()
        plt.savefig(
            GPATH
            + "/misc/papers/paper_al/"
            + self.dataset
            + "/multiple_iterations_plot.png"
        )
        plt.savefig(
            GPATH
            + "/misc/papers/paper_al/"
            + self.dataset
            + "/multiple_iterations_plot.svg"
        )
        plt.close()

    def run_similarity_analysis(self, multiple=True):
        """Main function to run similarity vs performance or performance matching"""
        # Uncomment if you want to plot performance of the methods
        # eval_plot_ap(self.methods, self.classes, self.orig_classes, self.model_paths, dataset=self.dataset)
        if self.performance:
            if multiple:
                yaps, ysims = [], []
                for i in range(1, 4):
                    self.n_iter = i
                    if i > 1:
                        self.model_paths = [
                            path.replace(
                                "_" + str(self.n_iter - 1) + "_",
                                "_" + str(self.n_iter) + "_",
                            )
                            for path in self.model_paths
                        ]
                        self.tfs = [
                            path.replace(
                                "_" + str(self.n_iter - 1) + "_",
                                "_" + str(self.n_iter) + "_",
                            )
                            for path in self.tfs
                        ]

                    yap, ysim = self.similarity_vs_performance()
                    yaps.append(yap)
                    ysims.append(ysim)
                self._plot_multiple_iterations(yaps, ysims)
            else:
                yap, ysim = self.similarity_vs_performance()
        else:
            self.similarity_vs_evaluation()
        print("Analaysis done")


# Similarity("KITTI").run_similarity_analysis()
# Similarity("BDD").run_similarity_analysis()
# Similarity("kCODA").run_similarity_analysis()
# Similarity("bCODA").run_similarity_analysis()

# Similarity("KITTI", train=True).run_similarity_analysis()
# Similarity("BDD", train=True).run_similarity_analysis()

# Similarity("KITTI", performance=False).run_similarity_analysis()
# Similarity("kCODA", performance=False).run_similarity_analysis()
# Similarity("BDD", performance=False).run_similarity_analysis()
# Similarity("bCODA", performance=False).run_similarity_analysis()

# Similarity("bCODA", n_iter=1).run_similarity_analysis(multiple=False)
