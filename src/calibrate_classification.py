# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Calibrate classification uncertainty """


import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from dataset_data import get_dataset_data
from matplotlib import pyplot as plt
from sklearn.isotonic import IsotonicRegression
from tensorflow.python.training import gradient_descent
from utils_class import stable_softmax


class ClassificationCalib:
    """Class to visualize the classification calibration of a model"""

    def __init__(self, y_true, y_pred, ep_unc, model_name, general_path):
        """Constructs all the necessary attributes for class calibration

        Args:
            y_true (array): True labels, as class labels
            y_pred (array): Predicted labels, as logits
            ep_unc (array/None): Predicted classification epistemic uncertainty
            model_name (str): Model name in order to save files under it
            per_classcalibration (bool): Select if calibration is done per class
            general_path (str): Path to working space

        """
        self.saving_path = general_path
        self.ep_unc = ep_unc
        self.calib_percls = False
        num_classes = y_pred.shape[1]

        self.y_true = np.squeeze(np.eye(num_classes)[y_true.astype(int).reshape(-1)])
        self.y_pred = y_pred

        self.orig_ytrue = np.squeeze(
            np.eye(num_classes)[y_true.astype(int).reshape(-1)]
        )
        self.orig_ypred = y_pred

        self.n_bins = 10
        self.classes_label = get_dataset_data(model_name)[2]
        self.model_name = model_name.split("/")[-1]

    def iso_regression(self):
        """Creates isotonic regressors for each class or for all classes at once

        Returns:
            Isotonic regression model/s
        """
        if self.calib_percls:
            isos = [
                IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
                    self.y_pred[:, i], self.y_true[:, i]
                )
                for i in range(len(self.classes_label))
            ]
        else:
            isos = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
                self.y_pred.flatten(), self.y_true.flatten()
            )
        return isos

    def temp_scaling(self):
        """Applies temperature scaling on the predicted logits for the validation set

        Returns:
            Temperature/s
        """
        if self.calib_percls:
            x = tf.Variable(
                [1.0] * len(self.classes_label), trainable=True, dtype=tf.float32
            )
        else:
            x = tf.Variable(1.0, trainable=True, dtype=tf.float32)

        @tf.autograph.experimental.do_not_convert
        def f_x():
            """Loss function to be reduced with gradient descent"""
            return tf.keras.losses.CategoricalCrossentropy(from_logits=True)(
                self.y_true, tf.divide(self.y_pred, x)
            )

        for _ in range(100):
            gradient_descent.GradientDescentOptimizer(0.1).minimize(f_x)
        temp = x.numpy()
        self.temper = temp
        return self.temper

    def _calibr_bins(self, y_true, y_pred):
        """Compute ece, mce and the bins for calibration curve

        Args:
            y_true: True labels
            y_pred: Predicted probabilities

        """
        if self.all_classes:
            maxim = y_pred.argmax(axis=-1)
            indices = np.arange(y_pred.shape[0])
            ytrue = y_true[indices, maxim].T.astype(np.float64)
            ypred = y_pred[indices, maxim].T.astype(np.float64)

        else:
            ytrue = y_true
            ypred = y_pred

        if self.quantilestrategy:  # Determine bin edges by distribution of data
            b = np.linspace(0.0, 1.0 + 1e-8, self.n_bins + 1)
            b[-1] -= 1e-8
            bins = np.quantile(ypred, b)
            bins = np.unique(bins).astype(np.float64)
            bins[-1] = 1 + 1e-8
        else:  # Linear distribution of bins
            bins = np.linspace(0.0, 1.0 + 1e-8, self.n_bins + 1)

        binids = np.digitize(ypred, bins)

        bin_sums = np.bincount(binids, weights=ypred, minlength=len(bins))
        bin_true = np.bincount(binids, weights=ytrue, minlength=len(bins))
        bin_total = np.bincount(binids, minlength=len(bins))  # count in each bin

        # To be able to divide it must be unequal to 0
        nonzero = bin_total != 0
        prob_true = bin_true[nonzero] / bin_total[nonzero]
        prob_pred = bin_sums[nonzero] / bin_total[nonzero]

        self.freq = prob_true
        self.prob = prob_pred

        self.bin_prob = np.insert(bins[nonzero], 0, 0)
        self.bin_count = bin_total[nonzero]
        self.ece = np.sum(
            np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(ytrue))
        )
        self.mce = np.max(np.abs(prob_true - prob_pred))

    def _bincount_bars(self, ax, color="blue", label="", alpha=1.0):
        """Draw the number of samples in each bin as bars

        Args:
            ax: Matplotlib axis to plot the bar plot
            color (str): Color of the bars
            label (str): Label of the bars
            alpha (float): Transperancy level of bars

        """

        w = [
            self.bin_prob[i] - self.bin_prob[i - 1]
            for i in range(1, len(self.bin_prob))
        ]
        centers = [
            (self.bin_prob[i] + self.bin_prob[i - 1]) / 2
            for i in range(1, len(self.bin_prob))
        ]
        ax.bar(
            centers,
            self.bin_count,
            linewidth=1,
            width=w,
            edgecolor="orange",
            color=color,
            label=label,
            alpha=alpha,
        )
        ax.scatter(centers, self.bin_count, color="orange")
        ax.set_xlim([-0.05, 1.05])
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")

    def _calibr_bars(self, ax, colors=["skyblue", "crimson"], alpha=1.0):
        """Plots the calibration bars with the difference to the perfect calibration

        Args:
            ax: Matplotlib axis to plot the bar plot
            colors (list): Plot colors
            alpha (float): Transperancy level of bars

        """

        l1 = "true"
        l2 = "Miscalibration Area"

        w = [
            self.bin_prob[i] - self.bin_prob[i - 1]
            for i in range(1, len(self.bin_prob))
        ]
        centers = [
            (self.bin_prob[i] + self.bin_prob[i - 1]) / 2
            for i in range(1, len(self.bin_prob))
        ]

        ax.bar(
            centers,
            self.freq,
            linewidth=1,
            width=w,
            edgecolor="black",
            color=colors[0],
            label=l1,
            alpha=alpha,
        )
        ax.bar(
            self.prob,
            self.freq - self.prob,
            bottom=self.prob,
            width=0.005,
            color=colors[1],
            label=l2,
            alpha=alpha,
        )

        ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
        ax.set_xlim([-0.05, 1.05])
        ax.set_xlabel("Predicted Probability")
        ax.set_ylim([0, 1])
        ax.set_ylabel("Proportion of Positives")

    def _calibr_lines(self, ax, colors, labels, alpha=1.0):
        """Plots the calibration lines and fills the difference to the perfect calibration

        Args:
            colors (list): Colors for the line and the filling
            labels (list): Labels for the line and filling
        """
        ax.plot(
            self.prob, self.freq, "s-", color=colors[0], label=labels[0], alpha=alpha
        )
        ax.fill_between(
            self.prob,
            self.freq,
            self.prob,
            color=colors[1],
            label=labels[1],
            alpha=alpha,
        )
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Proportion of Positives")
        ax.legend(prop={"size": 8}, frameon=False, handletextpad=0.2)
        ax.set_ylim([-0.05, 1.05])

    def _calibr_plot(
        self, quantilestrategy=False, calib=False, T=False, split=0, calib_percls=False
    ):
        """Plots the classification calibration plots (reliability diagrams)

        Args:
            quantilestrategy (bool): Selects the bins definition method (uniform or quantile based)
            calib (bool): Selects if calibration is to be plotted or not
            T (bool): Selects temperature scaling for calibration, else isotonic regression
            split (float): Dataset split for trian/val

        """
        self.calib_percls = calib_percls
        self.freq = []
        self.prob = []
        self.bin_count = []
        self.bin_prob = []
        self.all_classes = False

        self.temper = 1
        self.ece = 0
        self.mce = 0
        self.nll = 0
        self.calib_nll = 0

        if split == 0:
            split = int(len(self.orig_ypred) * 0.8)

        self.nll = tf.keras.losses.CategoricalCrossentropy()(
            self.orig_ytrue[split:], self.orig_ypred[split:]
        ).numpy()
        if calib:
            self.y_pred = self.orig_ypred[:split]
            self.y_true = self.orig_ytrue[:split]
            if T:
                self.temp_scaling()
                self.y_pred = stable_softmax(
                    self.orig_ypred[split:] / self.temper
                )  # Necessary to recalibrate into sum 1, take val split
            else:
                self.y_pred = stable_softmax(self.y_pred)
                iso = self.iso_regression()
                self.y_pred = stable_softmax(self.orig_ypred[split:])  # Take val split

            if self.calib_percls:
                if T:
                    calib_methd = "perclass_calib_TS"
                else:
                    calib_methd = "perclass_calib_IR"
                    ycalibs = [
                        iso[i].predict(self.y_pred[:, i])
                        for i in range(len(self.classes_label))
                    ]
                    ycalibs = np.stack(ycalibs, axis=1)
                    self.y_pred = ycalibs / np.stack(
                        [np.sum(ycalibs, axis=-1)] * len(self.classes_label), axis=-1
                    )

            else:
                if T:
                    calib_methd = "all_calib_TS"
                else:
                    calib_methd = "all_calib_IR"
                    ycalibs = iso.predict(self.y_pred.flatten()).reshape(
                        self.y_pred.shape
                    )
                    self.y_pred = ycalibs / np.stack(
                        [np.sum(ycalibs, axis=-1)] * len(self.classes_label), axis=-1
                    )
        else:
            self.y_pred = stable_softmax(self.orig_ypred[split:])
            calib_methd = ""
        self.y_true = self.orig_ytrue[split:]

        self.calib_nll = tf.keras.losses.CategoricalCrossentropy()(
            self.y_true, self.y_pred
        ).numpy()
        self.quantilestrategy = quantilestrategy

        if self.quantilestrategy:
            err = "ACE "
        else:
            err = "ECE "

        c1 = "skyblue"
        c2 = "crimson"
        l1 = ""
        l2 = "Miscalibration Area"

        # Plot the calibration curve for every class
        fig = plt.figure(figsize=(15, 6 * len(self.classes_label)))
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        plt.rcParams.update({"font.size": 10})
        plt.rcParams["hatch.linewidth"] = 0.1

        eces = []
        targets = range(self.y_true.shape[-1])
        subplot_num = 1
        for target in targets:
            f1 = plt.subplot(3 * int(targets[-1] / 3 + 1), 3, subplot_num)
            f1.plot([0, 1], [0, 1], "k:", label="Ideal")

            self._calibr_bins(
                y_true=self.y_true[:, target],
                y_pred=self.y_pred[:, target].astype("float64"),
            )
            self._calibr_lines(f1, [c1, c2], [self.classes_label[target] + l1, l2])
            # from sklearn.calibration import calibration_curve # test passed
            # fop, mpv = calibration_curve(self.y_true[:, target],self.y_pred[:, target].astype("float64"), n_bins=10)
            # f1.plot(mpv, fop, marker='.')
            eces.append(np.around(self.ece, 3))

            f2 = plt.subplot(3 * int(targets[-1] / 3 + 1), 3, subplot_num + 1)
            self._calibr_bars(f2, colors=[c1, c2])

            f3 = plt.subplot(3 * int(targets[-1] / 3 + 1), 3, subplot_num + 2)
            subplot_num += 3
            self._bincount_bars(f3, label="# samples" + l1)
            fig.subplots_adjust(hspace=0.5)
            brier = np.mean((self.y_pred[:, target] - self.y_true[:, target]) ** 2)
            f1.set_title(
                err
                + str(np.around(self.ece, 3))
                + "\nMCE "
                + str(np.around(self.mce, 3))
                + "     Brier: "
                + str(np.around(brier, 3))
            )

        fig.tight_layout()
        plt.savefig(
            self.saving_path
            + self.model_name
            + "/classification/class_plot_"
            + calib_methd
            + ".png",
            bbox_inches="tight",
        )

        # Plot the Calibration Curve for all classes
        brier = np.mean((self.y_pred - self.y_true) ** 2)
        self.all_classes = True
        self._calibr_bins(y_true=self.y_true, y_pred=self.y_pred)
        plt.figure(figsize=(15, 15))

        fin_1 = plt.subplot(3, 3, 1)
        fin_1.plot([0, 1], [0, 1], "k:", label="Ideal")
        self._calibr_lines(fin_1, [c1, c2], ["All classes" + l1, l2])
        fin_2 = plt.subplot(3, 3, 2)
        self._calibr_bars(fin_2, colors=[c1, c2])
        fin_3 = plt.subplot(3, 3, 3)
        self._bincount_bars(fin_3, label="# samples" + l1)
        if calib:
            nll = self.calib_nll
        else:
            nll = self.nll
        fin_1.set_title(
            err
            + str(np.around(self.ece, 3))
            + "     NLL: "
            + str(np.around(nll, 3))
            + "     SCE: "
            + str(np.around(np.mean(eces), 3))
            + "\n"
            + "MCE "
            + str(np.around(self.mce, 3))
            + "     Brier: "
            + str(np.around(brier, 3))
        )
        if calib and T:
            try:
                len(self.temper)
            except:
                self.temper = np.asarray([self.temper])
            temp = str([np.around(self.temper[i], 3) for i in range(len(self.temper))])
            if len(temp) > 30:  # Split to fit to title
                t1, t2 = temp[: len(temp) // 2], temp[len(temp) // 2 :]
                temp = t1 + "\n" + t2
            plt.title("Temperature: " + temp)
        # import tensorflow_probability as tfp # test passed
        # print(tfp.stats.expected_calibration_error(10, logits=list(self.orig_ypred[split:].astype(np.float32)), labels_true=list(np.argmax(self.y_true,axis=-1).astype(int))))
        fig.tight_layout()
        plt.savefig(
            self.saving_path
            + self.model_name
            + "/classification/all_plot_"
            + calib_methd
            + ".png",
            bbox_inches="tight",
        )

    def class_calibration(self):
        """Main function to calibrate predicted classification logits and plot calibration results"""

        def _generate_calib(y_true, y_pred, save_name=""):
            """Save calibration models on full set for further usage

            Args:
                y_true (array): True labels, as class labels
                y_pred (array): Predicted labels, as logits
            """
            self.y_pred = y_pred
            self.y_true = y_true
            self.calib_percls = False
            temps = self.temp_scaling()
            with open(
                self.saving_path
                + self.model_name
                + "/classification/"
                + save_name
                + "classification_ts_all",
                "wb",
            ) as fp:
                pickle.dump(temps, fp)

            self.calib_percls = True
            temps = self.temp_scaling()
            with open(
                self.saving_path
                + self.model_name
                + "/classification/"
                + save_name
                + "classification_ts_percls",
                "wb",
            ) as fp:
                pickle.dump(temps, fp)

            self.y_pred = stable_softmax(y_pred)
            self.y_true = y_true
            self.calib_percls = False
            isos = self.iso_regression()
            with open(
                self.saving_path
                + self.model_name
                + "/classification/"
                + save_name
                + "classification_iso_all",
                "wb",
            ) as fp:
                pickle.dump(isos, fp)
            self.calib_percls = True
            isos = self.iso_regression()
            with open(
                self.saving_path
                + self.model_name
                + "/classification/"
                + save_name
                + "classification_iso_percls",
                "wb",
            ) as fp:
                pickle.dump(isos, fp)

        # Start plotting
        print("Calibrating classification probabilities")
        # Temp scaling
        self._calibr_plot(quantilestrategy=False, calib=True, T=True)
        self._calibr_plot(quantilestrategy=False, calib=True, T=True, calib_percls=True)

        # Iso regression
        self._calibr_plot(quantilestrategy=False, calib=True)
        self._calibr_plot(quantilestrategy=False, calib=True, calib_percls=True)

        # No calibration
        self._calibr_plot(quantilestrategy=False, calib=False)

        _generate_calib(self.orig_ytrue, self.orig_ypred)

        # Consider uncertainty during calibration
        if self.ep_unc is not None:
            logits_dist = tfp.distributions.Normal(
                loc=self.orig_ypred, scale=self.ep_unc
            )
            samples = logits_dist.sample(10)
            sample_ypred = np.asarray(samples).reshape([-1, self.orig_ypred.shape[-1]])
            sample_ytrue = np.stack([self.orig_ytrue] * 10, axis=0).reshape(
                [-1, self.orig_ypred.shape[-1]]
            )
            _generate_calib(sample_ytrue, sample_ypred, save_name="unc_")

        print(
            "Plotted classification calibration on 20% of the validation dataset, the rest is used for fiting"
        )
