# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================

import json
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ssl_utils.parent import Parent_SSL

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({"font.size": 10})
plt.rcParams["hatch.linewidth"] = 0.1


class Advanced_Pseudo_Labels(Parent_SSL):
    """Advanced Pseudo Labels Selection Strategy in Correlation with Missing Detection Rate and Class Distribution"""

    def __init__(
        self,
        beta,
        added_pseudo_name,
        top_k,
        *args,
        **kwargs,
    ):
        """
        Initializes the PLS.
        Args:
            beta (float): The beta value for PLS.
            added_pseudo_name (str): The name added to saving paths.
            top_k (int): The ratio of top predictions to consider.
            *args: Variable length argument list. Belongs to parent class.
            **kwargs: Arbitrary keyword arguments. Belongs to parent class.
        """

        super().__init__(*args, **kwargs)  # Call parent constructor
        self.beta = beta
        self.top_k = top_k
        self.added_pseudo_name = added_pseudo_name
        self.images_data = self.read_pred_folder()
        self.extract_pseudo_gt_data()
        self._pls()
        print("Pseudo Labels Processed")

    def _gen_selected_pseudo(self, inds_list, added_name="top_"):
        """
        Generates selected pseudo labels based on the given indices list.
        Args:
            inds_list (list): List of indices representing the selected pseudo labels.
            added_name (str, optional): Additional name to be added to the new pseudo label folder. Defaults to "top_".
        Returns:
            str: Path to the newly created pseudo label folder.
        """

        added_name = self.added_pseudo_name + added_name
        keep_dets = [self.images_data[s] for s in inds_list]
        if self.dataset == "KITTI":
            new_det_folder = (
                "/".join(self.det_folder.split("/")[:-2])
                + "/"
                + added_name
                + self.det_folder.split("/")[-2]
                + "/"
            )
        else:
            p1 = self.det_folder.split("pseudo_labels")[-1]
            new_det_folder = (
                self.dataset_path
                + "/pseudo_labels/"
                + p1.split("/")[1]
                + "/"
                + added_name
                + p1.split("/")[-1]
            )
        if os.path.exists(new_det_folder):
            shutil.rmtree(new_det_folder)
        os.makedirs(new_det_folder)
        if self.dataset == "KITTI":
            [shutil.copy(self.det_folder + v, new_det_folder + v) for v in keep_dets]
        else:
            with open(self.det_folder + "/pseudo_labels.json", "r") as file:
                pred_bdd_data = json.load(file)
            filtered_data = [
                item for item in pred_bdd_data if item["name"] in keep_dets
            ]

            with open(new_det_folder + "/pseudo_labels.json", "w") as json_file:
                json.dump(filtered_data, json_file, indent=4)
        return new_det_folder

    def _pls(self):
        """Run PLS"""

        def plot_roc_score(save_path):
            """Plot ROC curve for different scores vs. missing detection rate"""
            thresholds = [0, 0.25, 0.5, 0.75]
            plt.figure(figsize=(10, 5))

            for i, threshold in enumerate(thresholds, 1):
                gt_missing_dets = (n_missing_dets > threshold).astype(int)
                metrics = {
                    "C_i": c_i,
                    "D_i": d_i,
                    "S_i": s_i,
                    "Max Drop": max_drop,
                    "Mean Drop": mean_drop,
                    "STD Drop": std_drop,
                    "# Dets": n_det,
                    "Avg. Score": avg_score,
                }
                plt.subplot(2, 2, i)

                for label, scores in metrics.items():
                    if (
                        "C_i" in label
                        or "S_i" in label
                        or "D_i" in label
                        or "Score" in label
                    ):
                        ps = 0
                    else:
                        ps = 1
                    fpr, tpr, _ = roc_curve(gt_missing_dets, scores, pos_label=ps)
                    auc_val = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc_val:.2f})")

                plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"MD % > {threshold}")
                plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

            plt.tight_layout()
            plt.savefig(save_path + "/compare_roc_score.png")
            plt.close()

        def plot_hist_score(score, save_path):
            """Plot histogram of scores"""
            counts, bins = np.histogram(score, bins=10)
            plt.figure()
            for i in range(len(bins) - 1):
                plt.bar(
                    bins[i],
                    counts[i],
                    width=bins[i + 1] - bins[i],
                    color="blue",
                    edgecolor="black",
                )
            plt.xlabel("Score")
            plt.ylabel("Frequency")
            plt.savefig(save_path + "/hist_score.png")
            plt.close()

        n_missing_dets = (
            np.asarray(self.n_gts_perim) - np.asarray(self.n_gt_matches)
        ) / np.asarray(self.n_gts_perim)
        pred_data = self.read_predictions(self.inference_path, "score")
        pred_im_names, score_perim = pred_data[:2]
        clean_perd_im_names = np.asarray([i.split(".")[0] for i in pred_im_names])
        clean_im_names = [i.split(".")[0] for i in self.images_data]
        match_inds = [np.where(clean_perd_im_names == m)[0][0] for m in clean_im_names]
        match_score_perim = [score_perim[i] for i in match_inds]
        drate = [
            np.asarray([np.sum(np.asarray(sp) >= i) for sp in match_score_perim])
            for i in np.linspace(0, 1, 11)
        ]
        n_det = [len(a) for a in match_score_perim]
        delta_s = int(self.det_folder.split("thr_")[-1][1])

        s_i = drate[delta_s] / drate[0]
        differences = [drate[i - 1] - drate[i] for i in range(1, len(drate))]
        max_drop = np.max(differences, axis=0)
        mean_drop = np.mean(differences, axis=0)
        std_drop = np.std(differences, axis=0)
        avg_score = [np.mean(a) for a in match_score_perim]

        all_pred_classes = np.concatenate(self.collect_pseudo_classes)
        len_perc = [
            1 - np.sum(all_pred_classes == c) / len(all_pred_classes)
            for c in self.used_classes
        ]
        len_perc = {c: v for c, v in zip(self.used_classes, len_perc)}
        c_i = [
            np.mean([len_perc[c] for c in pred_classes])
            for pred_classes in self.collect_pseudo_classes
        ]

        print_n_missing_dets = np.asarray(self.n_gts_perim) - np.asarray(
            self.n_gt_matches
        )
        d_i = (1 - self.beta) * np.asarray(s_i) + np.asarray(c_i) * self.beta

        threshold = np.percentile(d_i, self.top_k * 100)
        new_score_top = np.where(d_i >= threshold)[0]
        new_score_bot = np.where(d_i < threshold)[0]

        new_clstop = np.concatenate(
            [self.allocated_dets["gt"]["class"][i] for i in new_score_top]
        )
        new_clsbot = np.concatenate(
            [self.allocated_dets["gt"]["class"][i] for i in new_score_bot]
        )
        new_score_rand = np.arange(len(self.collect_pseudo_classes))
        np.random.shuffle(new_score_rand)
        new_score_rand = new_score_rand[: len(new_score_top)]
        new_clsrand = np.concatenate(
            [self.allocated_dets["gt"]["class"][i] for i in new_score_rand]
        )

        save_path_top = self._gen_selected_pseudo(
            inds_list=new_score_top, added_name="_top_"
        )
        save_path_bot = self._gen_selected_pseudo(
            inds_list=new_score_bot, added_name="_bot_"
        )
        save_path_rand = self._gen_selected_pseudo(
            inds_list=new_score_rand, added_name="_rand_"
        )
        save_path = (
            save_path_rand.split("rand")[0] + "plots" + save_path_rand.split("rand")[1]
        )
        os.makedirs(save_path, exist_ok=True)
        # Generate plots
        plot_roc_score(save_path)
        plot_hist_score(s_i, save_path)
        np.save(save_path + "/score_si.npy", s_i)
        np.save(save_path + "/score_ci.npy", c_i)
        np.save(save_path + "/max_drop.npy", max_drop)
        np.save(save_path + "/mean_drop.npy", mean_drop)
        np.save(save_path + "/avg_score.npy", avg_score)
        np.save(save_path + "/std_drop.npy", std_drop)
        np.save(save_path + "/n_det.npy", n_det)
        np.save(save_path + "/md.npy", n_missing_dets)
        np.save(save_path + "/md.npy", n_missing_dets)
        np.save(
            save_path + "/gt_cls.npy",
            np.array(self.allocated_dets["gt"]["class"], dtype=object),
        )
        analysis_print = (
            f"orignal md: {np.sum(print_n_missing_dets)/np.sum(self.n_gts_perim)*100} "
            f"and cls distribution "
            f"{[np.sum(np.concatenate([self.allocated_dets['gt']['class'][i] for i in range(len(self.matched_preds))]) == c) for c in self.used_classes]} \n"
        )
        analysis_print += f"bot score: {np.sum(print_n_missing_dets[new_score_bot])/np.sum(np.asarray(self.n_gts_perim)[new_score_bot])*100} and cls distribution {[np.sum(new_clsbot == c) for c in self.used_classes]} \n"
        analysis_print += f"top score: {np.sum(print_n_missing_dets[new_score_top])/np.sum(np.asarray(self.n_gts_perim)[new_score_top])*100} and cls distribution {[np.sum(new_clstop == c) for c in self.used_classes]} \n"
        analysis_print += f"random: {np.sum(print_n_missing_dets[new_score_rand])/np.sum(np.asarray(self.n_gts_perim)[new_score_rand])*100} and cls distribution {[np.sum(new_clsrand == c) for c in self.used_classes]} \n"

        tau = float("0." + self.det_folder.split("thr")[-1].split("_")[1][-1])
        v_nummer = self.det_folder.split("_V")[-1]
        stac_commands = [
            f"PYTHONPATH=/{self.general_path}/src/ python -m SSL_stac --gpu 0 --dataset {self.dataset} --portion_labeled {self.added_name.split('_')[-1]} --tau {tau} --selection_strategy {self.added_pseudo_name+'_'+ rank +'_' + self.det_folder.split('/')[-1].split('_thr')[0]} --version_num {v_nummer} --num_epochs 200"
            for rank in ["top", "bot", "rand"]
        ]
        print(stac_commands)
        # Calculate data
        original_data = self.print_data
        self.det_folder = save_path_top
        self.images_data = self.read_pred_folder()
        self.extract_pseudo_gt_data(new_dets=True)
        top_data = self.print_data
        self.det_folder = save_path_rand
        self.images_data = self.read_pred_folder()
        self.extract_pseudo_gt_data(new_dets=True)
        rand_data = self.print_data
        self.det_folder = save_path_bot
        self.images_data = self.read_pred_folder()
        self.extract_pseudo_gt_data(new_dets=True)
        bot_data = self.print_data

        filename = save_path + "/output.txt"

        # Open the file in write mode and redirect print statements to it
        with open(filename, "w") as file:
            # Save original MD and class distribution
            file.write(analysis_print)
            file.write(f"original: {original_data} \n")
            file.write(f"bot: {bot_data} \n")
            file.write(f"top: {top_data} \n")
            file.write(f"rand: {rand_data} \n")


# Create an instance of the child class
advanced_pseudo_labels = Advanced_Pseudo_Labels(
    # dataset="KITTI",
    # det_folder="pseudo_labels/num_labeled_10/score_thr_04_V0",
    # model_name="EXP_KITTI_STAC_teacher10_V0",
    # labeled_indices_path="num_labeled_10/V0/_train_init_V0.txt",
    # added_name="num_labeled_10",
    dataset="BDD100K",
    det_folder="pseudo_labels/num_labeled_1/score_thr_04_V0",
    model_name="EXP_BDD100K_STAC_teacher1_V0",
    labeled_indices_path="num_labeled_1/V0/_train_init_V0.txt",
    added_name="num_labeled_1",
    beta=0.1,
    added_pseudo_name="pls_new_beta01_topk60",
    top_k=0.6,
)
