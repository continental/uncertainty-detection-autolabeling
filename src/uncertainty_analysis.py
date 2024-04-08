# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Specific for paper: Cost-Sensitive Uncertainty-Based Failure Recognition for Object Detection.
    Analysis of the uncertainty for optimal failure recognition.
"""

import random
import os
import ast 
import shutil
import copy
from collections import Counter
from collections import defaultdict

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import roc_curve, auc
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
import optuna
from PIL import Image

from utils_box import relativize_uncert, calc_iou_np
from utils_infer import visualize_image
from utils_extra import calc_jsd
from dataset_data import get_dataset_data
from hparams_config import default_detection_configs

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 20})

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPR_TPR = default_detection_configs().thr_fpr_tpr
FIX_CD = default_detection_configs().thr_cd
IOU_THRS = default_detection_configs().thr_iou_thrs
SELECT_UNCERT = default_detection_configs().thr_sel_uncert

def roc_metrics(uncert, y_true):
    """ Automatically determines the threshold for a given FPR/FNR alongside the error and AUC """
    fpr, tpr, thresholds = roc_curve(y_true, uncert, pos_label=0)
    roc_auc = auc(fpr, tpr)
    if FIX_CD:
        if all(fpr > 1-FPR_TPR):
            # No threshold allows FPR < 1-FPR_TPR
            return 0
        elif all(fpr <= 1-FPR_TPR):
            # All thresholds allow FPR <= FPR_TPR, so find lowest possible FPR
            idxs = [i for i, x in enumerate(1-fpr) if x >= 1]
            return min(map(lambda idx: thresholds[idx], idxs)), min(map(lambda idx: (1-tpr)[idx], idxs)), roc_auc
        else:
            # Linear interp between values to get actual FPR at (FPR == 1-FPR_TPR:TPR)
            roc_fpr = 1 - np.interp(1-FPR_TPR, fpr, tpr)
            idx = np.argmin(np.abs(1-tpr - roc_fpr))
            thr = thresholds[idx]
            return thr, roc_fpr, roc_auc    
    else:            
        if all(tpr < FPR_TPR):
            # No threshold allows TPR >= FPR_TPR
            return 0
        elif all(tpr >= FPR_TPR):
            # All thresholds allow TPR >= FPR_TPR, so find lowest possible FPR
            idxs = [i for i, x in enumerate(tpr) if x >= 1]
            return min(map(lambda idx: thresholds[idx], idxs)), min(map(lambda idx: fpr[idx], idxs)), roc_auc
        else:
            # Linear interp between values to get FPR at TPR == FPR_TPR
            fpr95 = np.interp(FPR_TPR, tpr, fpr)
            idx = np.argmin(np.abs(fpr - fpr95))
            thr = thresholds[idx]
            return thr, fpr95, roc_auc
            
class UncertOptimal:
    """ Optimally combines the different uncertainties """
    def __init__(self, gt_classes=None, tps_class=None, ious=None, uncert=None, added_name="", source_path="", per_cls=False, method="optuna"):     
        """ Constructs the necessary attributes to determine the optimal combination of the uncertainties.
        If None, it reads from file. If not None, it performs all steps and saves optimal parameters to file.

        Args:
            gt_classes (array,optional): Ground Truth classes for per class optimization.
            tps_class (array, optional): Boolean array for correct classification for each detection. Defaults to None.
            ious (array, optional): Iou of each detection. Defaults to None.
            uncert (list, optional): List of uncertainties to optimally combine. Defaults to None.
            added_name(str, optional): Can be used to add a class name to the save txt file with the parameters.
            source_path (str, optional): Source path wtih model name to add to saving path. Defaults to "".
            per_cls(bool, optional): Optimizes a larger space per class.
            method(str, optional): Method to use for optimization, "hebo" or "optuna". Defaults to "optuna".
        """
        self.source_path =  source_path
        self.added_name = added_name
        self.per_cls = per_cls
        self.gt_classes = gt_classes
        self.method=method
        if tps_class is not None:
            self.tps_class = tps_class
            self.ious = ious
            self.uncert = uncert
            self.opt_params = [0, 0]

    def _extract_optimal_params(self, break_iter=10):
        """ Performs Bayesian optimization to determine the optimal combination """
        def _f_x(params):
            """ Loss function for optimization """
            opt_fpr = []
            for iou_thr in IOU_THRS:
                correct_detections = np.asarray((self.ious >= iou_thr) * self.tps_class, dtype=int)
                if self.per_cls:
                    collected_uncert = copy.deepcopy(self.uncert)
                    n_iter = 0
                    for i in range(self.num_classes):
                        for j in range(len(self.uncert)):
                            collected_uncert[j][self.gt_classes==i+1] *= params[n_iter] # Important test with pred
                            n_iter+=1
                    uncert = np.sum(collected_uncert,axis=0)
                else: uncert = sum(param * uncert for param, uncert in zip(params, self.uncert))
                roc_fpr = roc_metrics(uncert, correct_detections)[1]
                if np.isnan(roc_fpr): roc_fpr = 1
                opt_fpr.append(roc_fpr*100)
            return np.mean(opt_fpr)

        def _f(x):
            """ Wraps _f_x() """
            x = x.values
            n_particles = x.shape[0]
            j = [_f_x(x[i]) for i in range(n_particles)]
            return np.asarray(j)
        
        if self.per_cls:
            self.num_classes = int(max(self.gt_classes))
            num_params = len(self.uncert)*self.num_classes
        else: num_params = len(self.uncert)

        if self.method=="hebo":
            space = DesignSpace().parse([{'name': f'param_{i}', 'type': 'num', 'lb': 0, 'ub': 1} for i in range(1, num_params + 1)])
        
            optimizer  = HEBO(space)
            prev_min_f = np.inf
            num_iter_since_change = 0
            for i in range(300):
                rec = optimizer.suggest(n_suggestions=5)
                optimizer.observe(rec, _f(rec))
                min_f = optimizer.y.min()
                if min_f == prev_min_f:
                    num_iter_since_change += 1
                    if num_iter_since_change == break_iter:
                        print('Minimum objective value has not changed for 10 iterations. Breaking early.')
                        break
                else:
                    prev_min_f = min_f
                    num_iter_since_change = 0
                print('After %d iterations, best obj is %.2f' % (i, min_f))   
            opt_iter = np.argmin(optimizer.y)
            self.opt_params = optimizer.X.iloc[opt_iter]
        else:
            study = optuna.create_study(direction='minimize')
            # Set the initial minimum value and the counter for unchanged minimum
            min_value = float('inf')
            unchanged_count = 0
            max_unchanged_epochs = 300  # Set the desired number of epochs for convergence
            for epoch in range(1, 1500):  # Assuming you want to run for 300 epochs
                # Optimize a trial
                trial = study.ask()
                suggested_params = [trial.suggest_float(f'param_{i}', 0, 1) for i in range(1, num_params + 1)]
                value = _f_x(suggested_params)
                study.tell(trial, value)
                print(f"Epoch {epoch}, Value {value}, Best Value {min_value}")

                # Check for convergence
                if value < min_value:
                    min_value = value
                    unchanged_count = 0
                else:
                    unchanged_count += 1

                if unchanged_count >= max_unchanged_epochs:
                    print(f"Convergence reached. Minimum value has not changed for {max_unchanged_epochs} epochs.")
                    break

            # Get the best parameters
            self.opt_params = [study.best_params[f'param_{i}'] for i in range(1, num_params + 1)]

        if FIX_CD: budget="cd"
        else: budget="fd"
        with open(self.source_path+"/optimal_params_"+budget+"_"+str(FPR_TPR)+"_iou_"+str(np.min(IOU_THRS))+"_"+str(np.max(IOU_THRS))+self.added_name+".txt", "w") as file:
            file.write(str(self.opt_params))
        
        # Save optimal threshold
        thrs = []
        for iou_thr in IOU_THRS:
            correct_detections = np.asarray((self.ious >= iou_thr) * self.tps_class, dtype=int)
            if self.per_cls:
                collected_uncert = copy.deepcopy(self.uncert)
                n_iter = 0
                for i in range(self.num_classes):
                    for j in range(len(self.uncert)):
                        collected_uncert[j][self.gt_classes==i+1] *= self.opt_params[n_iter] # Important test with pred
                        n_iter+=1
                uncert = np.sum(collected_uncert,axis=0)
            else: uncert = sum(param * uncert for param, uncert in zip(self.opt_params, self.uncert))
            thrs.append(roc_metrics(uncert, correct_detections)[0])
        
        with open(self.source_path+"/optimal_thrs_"+budget+"_"+str(FPR_TPR)+"_iou_"+str(np.min(IOU_THRS))+"_"+str(np.max(IOU_THRS))+self.added_name+".txt", "w") as file:
            file.write(str(np.asarray(thrs, dtype="object")))
            
    def get_optimal_uncertainty(self, break_iter=10):    
        """ Main function to either read, if available, or deermine the optimal weight vector for the sum combination of the uncertainties """    
        if FIX_CD: budget="cd"
        else: budget="fd"
        if os.path.exists(self.source_path+"/optimal_params_"+budget+"_"+str(FPR_TPR)+"_iou_"+str(np.min(IOU_THRS))+"_"+str(np.max(IOU_THRS))+self.added_name+".txt"):   
            with open(self.source_path+"/optimal_params_"+budget+"_"+str(FPR_TPR)+"_iou_"+str(np.min(IOU_THRS))+"_"+str(np.max(IOU_THRS))+self.added_name+".txt", "r") as file:
                self.opt_params = [float(x.strip("[]")) for x in file.read().split(",")]
        else:
            self._extract_optimal_params(break_iter)

        return self.opt_params

class MainUncertViz:
    """ Performs different uncertainty analysis and combination steps """
    def __init__(self, model_name, calib=True, per_cls=False, general_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        """ Constructs the necessary attributes to analyse the uncertainty

        Args:
            model_name (str): Model name to read results from and save to
            calib (bool, optional): Select if calibrated or regular uncertainties are to be used
            per_cls (bool, optional): Activates per-class thresholding
            general_path (str, optional): Path to working space
        """
        self.source_path =  general_path + '/results/validation/' + model_name         
        self.imgs_list = []
        self.pred_boxes = []
        self.pred_classes = []
        self.pred_scores = []
        self.gt_boxes = []
        self.gt_classes = []
        self.albox = None
        self.mcbox = None
        self.mcclass = None
        self.entropy = None
        self.calib = calib
        self.per_cls = per_cls

        self.calib_albox = None
        self.calib_entropy = None

        self.ious = []
        self._read_predictions()
        
        if self.albox is not None: self.albox = relativize_uncert(self.pred_boxes, self.albox)
        if self.mcbox is not None: self.mcbox = relativize_uncert(self.pred_boxes, self.mcbox)
        if self.calib_albox is not None: self.calib_albox = relativize_uncert(self.pred_boxes, self.calib_albox)

        # Create correct classification mask  
        tps_indices = np.where(self.pred_classes == self.gt_classes)[0]
        self.tps_class = np.zeros_like(self.pred_classes, dtype=bool)
        self.tps_class[tps_indices] = True
        self.selected_uncerts = []
        
        if calib:
            self.source_path += "/thresholding/calib/"
            if "ENT" in SELECT_UNCERT:
                    self.selected_uncerts.append(self.calib_entropy)
            if "ALBOX" in SELECT_UNCERT:
                self.selected_uncerts.append(np.mean(self.calib_albox, axis=-1))
        else:
            self.source_path += "/thresholding/orig/"
            if "ENT" in SELECT_UNCERT:
                    self.selected_uncerts.append(self.entropy)
            if "ALBOX" in SELECT_UNCERT:
                self.selected_uncerts.append(np.mean(self.albox, axis=-1))

        self.label_map, img_source_path = get_dataset_data(self.source_path)[:2]
        if not os.path.exists(self.source_path): os.makedirs(self.source_path)
        
        all_opt_params = UncertOptimal(self.gt_classes, self.tps_class, self.ious, 
                                       self.selected_uncerts, source_path=self.source_path, added_name="").get_optimal_uncertainty()
        self.opt_uncert=sum(opt_param * uncert for opt_param, uncert in zip(all_opt_params, self.selected_uncerts))
        self._save_thr_performance(added_name="", opt_params=all_opt_params) 

        if self.per_cls:
            perc_opt_params = UncertOptimal(self.gt_classes, self.tps_class, self.ious, self.selected_uncerts, 
                                            source_path=self.source_path, added_name="_clsopt", per_cls=True).get_optimal_uncertainty(break_iter=30)
            # Pred
            collected_uncert = copy.deepcopy(self.selected_uncerts)
            num_classes = int(max(self.gt_classes))
            n_iter = 0
            for i in range(num_classes):
                for j in range(len(self.selected_uncerts)):
                    collected_uncert[j][self.pred_classes==i+1] *= perc_opt_params[n_iter] # Pred or GT for Eval? Assuming perfect classification
                    n_iter+=1
            self.opt_uncert = np.sum(collected_uncert,axis=0)
            self._save_thr_performance(added_name="_clsoptpred") 
            # GT
            collected_uncert = copy.deepcopy(self.selected_uncerts)
            num_classes = int(max(self.gt_classes))
            n_iter = 0
            for i in range(num_classes):
                for j in range(len(self.selected_uncerts)):
                    collected_uncert[j][self.gt_classes==i+1] *= perc_opt_params[n_iter]
                    n_iter+=1
            self.opt_uncert = np.sum(collected_uncert,axis=0)
            self._save_thr_performance(added_name="_clsopt", opt_params=perc_opt_params) 

            # Redo with fix
            collected_uncert = copy.deepcopy(self.selected_uncerts)
            n_iter = 0
            for i in range(num_classes):
                per_cls_uncert = [collected_uncert[j][self.gt_classes==i+1] for j in range(len(self.selected_uncerts))]
                opt_uncert = sum(opt_param * uncert for opt_param, uncert in zip(all_opt_params, per_cls_uncert))
                if len(opt_uncert)>0:
                    all_fdcd = self._save_thr_performance(selected_uncert=per_cls_uncert, opt_uncert=opt_uncert, 
                                                                tps_class=self.tps_class[self.gt_classes==i+1], ious=self.ious[self.gt_classes==i+1], 
                                                                added_name="_clsoptfix")
                    temp_n_iter = n_iter
                    temp_opt_params = []
                    for j in range(len(self.selected_uncerts)):
                        temp_opt_params.append(perc_opt_params[temp_n_iter])
                        temp_n_iter+=1
                    opt_uncert = sum(opt_param * uncert for opt_param, uncert in zip(temp_opt_params, per_cls_uncert))
                    current_fdcd = self._save_thr_performance(selected_uncert=per_cls_uncert, opt_uncert=opt_uncert, 
                                                                tps_class=self.tps_class[self.gt_classes==i+1], ious=self.ious[self.gt_classes==i+1], 
                                                                added_name="_clsoptfix")
                    if current_fdcd < all_fdcd or current_fdcd==100: # If worse than for all
                        temp_n_iter = n_iter
                        for j in range(len(all_opt_params)):
                            perc_opt_params[temp_n_iter] = all_opt_params[j]
                            temp_n_iter+=1
                for j in range(len(self.selected_uncerts)):
                    collected_uncert[j][self.gt_classes==i+1] *= perc_opt_params[n_iter] # Pred or GT for Eval? Assuming perfect classification
                    n_iter+=1
            fixed_opt_uncert = np.sum(collected_uncert,axis=0)
            self._save_thr_performance(opt_uncert=fixed_opt_uncert, added_name="_clsoptfix", opt_params=perc_opt_params) 


        # self.img_source_path = general_path + '/datasets/' + img_source_path
        # self.pred_filters = self._collect_thresh_results(eval_iou_thr=np.max(IOU_THRS))

        # If unwanted, turn off each one
        # self._plot_metricsspider()
        # self._collect_postthresholding()
        # self._plot_validheat()  

    @staticmethod
    def _extract_det(detections, ious, value):
        """ Extracts detection if its detected """
        return np.asarray([detections[i][value] for i in range(len(detections))])[ious>0.0]
    
    def _save_thr_performance(self, selected_uncert=None, opt_uncert=None, tps_class=None, ious=None, added_name="", opt_params=None):
        """ Calculates and saves thresholding performance """
        if selected_uncert is None: selected_uncert=self.selected_uncerts
        if opt_uncert is None: opt_uncert=self.opt_uncert
        if tps_class is None: tps_class=self.tps_class
        if ious is None: ious=self.ious

        def _metrics_iou(uncert, tps_class=tps_class, ious=ious):
            """ Calculates JSD, FPR, AUC """
            jsds = []
            fprs = []
            aucs = []
            for iou_thr in IOU_THRS:
                correct_detections = np.asarray((ious >= iou_thr) * tps_class)
                jsds.append(np.nan_to_num(calc_jsd(uncert[correct_detections], uncert[~correct_detections])))      
                fpr, auc = roc_metrics(uncert, correct_detections)[1:] 
                if np.isnan(fpr):
                    fprs.append(0)
                else: 
                    fprs.append((1-fpr)*100)
                aucs.append(np.nan_to_num(auc)*100)
            return jsds, aucs, fprs
        
        jsds_orig, aucs_orig, fprs_orig = zip(*[_metrics_iou(uncert) for uncert in selected_uncert])
        jsds_opt, aucs_opt, fprs_opt = _metrics_iou(opt_uncert)
        if FIX_CD: 
            budget="cd"
            metric="FD@CD"+str(int(FPR_TPR*100))
        else: 
            budget="fd"
            metric="CD@FD"+str(int(FPR_TPR*100))

        if opt_params is not None:
            num_classes = int(max(self.gt_classes))
            class_jsds = np.zeros((num_classes, len(IOU_THRS)))
            class_aucs = np.zeros((num_classes, len(IOU_THRS)))
            class_fprs = np.zeros((num_classes, len(IOU_THRS)))

            n_iter=0
            for i in range(num_classes):
                selected_inds = np.where(self.gt_classes==i+1)[0]
                if len(selected_inds)>0:
                    selected_tpclass = tps_class[selected_inds]
                    selected_ious = ious[selected_inds]
                    selected_uncert_class = [unc[selected_inds] for unc in selected_uncert]
                    if len(opt_params) == len(selected_uncert):
                        opt_uncert_class = sum(opt_param * uncert for opt_param, uncert in zip(opt_params, selected_uncert_class))
                    else:
                        for j in range(len(selected_uncert_class)):
                            selected_uncert_class[j] *= opt_params[n_iter] # Pred or GT for Eval? Assuming perfect classification
                            n_iter+=1
                        opt_uncert_class = np.sum(selected_uncert_class,axis=0)

                    jsds_class, aucs_class, fprs_class = _metrics_iou(opt_uncert_class, selected_tpclass, selected_ious)
                else: jsds_class = aucs_class = fprs_class = [0]*len(IOU_THRS) 

                class_jsds[i, :] = jsds_class
                class_aucs[i, :] = aucs_class
                class_fprs[i, :] = fprs_class

            iou_range = [f"{i:.2f}" for i in IOU_THRS]
            item_size = int(len(iou_range)/2)
            table_data = [['Classes']+['-']*item_size+['AUC']+['-']*item_size+['*']*item_size+['JSD']+['*']*item_size+['']*item_size+[metric]+['']*item_size, ["IoU Thr."]+iou_range+["Avg."]+iou_range+["Avg."]+iou_range+["Avg."]]

            for i in range(num_classes):
                table_data.append([self.label_map[i+1]]+list(np.round(class_aucs[i],2))+[np.round(np.mean(class_aucs[i]),2)]+list(np.round(class_jsds[i],2))+[np.round(np.mean(class_jsds[i]),2)]+list(np.round(class_fprs[i],2))+[np.round(np.mean(class_fprs[i]),2)])
            avg_row = []
            for i in range(1,len(table_data[0])):
                avg_row.append(np.round(np.mean((np.asarray(table_data)[:,i][2:]).astype("float")),2))
            table_data.append(["Avg. perc"]+avg_row)

            for i in range(len(aucs_orig)): table_data.append(["Orig. U." +str(i)]+list(np.round(aucs_orig[i],2))+[np.round(np.mean(aucs_orig[i]),2)]+list(np.round(jsds_orig[i],2))+[np.round(np.mean(jsds_orig[i]),2)]+list(np.round(fprs_orig[i],2))+[np.round(np.mean(fprs_orig[i]),2)])
            table_data.append(["Opt. Combo"]+list(np.round(aucs_opt,2))+[np.round(np.mean(aucs_opt),2)]+list(np.round(jsds_opt,2))+[np.round(np.mean(jsds_opt),2)]+list(np.round(fprs_opt,2))+[np.round(np.mean(fprs_opt),2)])

            fig, ax = plt.subplots(figsize=(15,5))
            # table = ax.table(loc='center', cellText=table_data, cellLoc='center', colLabels=['Methods'] + list(classes))
            table = ax.table(cellText=table_data, loc='center', cellLoc='center',)
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)
            table.auto_set_column_width([0, 1])
            plt.title(added_name[1:], fontsize=16, pad=-30)

            ax.axis('off')
            plt.tight_layout()
            plt.savefig(self.source_path+"/thr_metrics_"+budget+"_"+str(FPR_TPR)+"_iou_"+str(np.min(IOU_THRS))+"_"+str(np.max(IOU_THRS))+added_name+".png")
            plt.close()

        jsds_orig = np.mean(jsds_orig, axis=-1)
        aucs_orig = np.mean(aucs_orig, axis=-1)
        fprs_orig = np.mean(fprs_orig, axis=-1)
        jsds_opt = np.mean(jsds_opt, axis=0)
        aucs_opt = np.mean(aucs_opt, axis=0)
        fprs_opt = np.mean(fprs_opt, axis=0)
        
        with open(self.source_path+"/thr_metrics_"+budget+"_"+str(FPR_TPR)+"_iou_"+str(np.min(IOU_THRS))+"_"+str(np.max(IOU_THRS))+added_name+".txt", "w") as file:
            file.write("JSD, AUC, " + metric + " of Each Uncertainty:\n")
            for idx, (jsd, auc, fpr) in enumerate(zip(jsds_orig, aucs_orig, fprs_orig), start=1):
                file.write(f"Uncertainty {idx}: JSD = {jsd:.3f}, AUC = {auc:.3f}, {metric} = {fpr:.3f}\n")

            file.write("\nJSD, AUC, " + metric + " of Optimal Uncertainty:\n")
            file.write(f"JSD = {jsds_opt:.3f}, AUC = {aucs_opt:.3f}, {metric} = {fprs_opt:.3f}\n")

        return fprs_opt

    def _collect_thresh_results(self, eval_iou_thr):  
        """ Collects correctly/falsely removed and falsely remaining detections """  
        correct_detections = np.asarray((self.ious >= eval_iou_thr) * self.tps_class, dtype=int)
        opt_thr = roc_metrics(self.opt_uncert, correct_detections)[0]
        correctly_removed = (self.opt_uncert >= opt_thr) * ~correct_detections.astype("bool")
        falsely_removed = (self.opt_uncert >= opt_thr) * correct_detections.astype("bool")
        falsely_remaining = (self.opt_uncert < opt_thr) * ~correct_detections.astype("bool")
        return correctly_removed, falsely_removed, falsely_remaining, opt_thr
    
    def _draw_postthresholding(self, img_name, title):
        """ Draws a figure of 4 subplots including the image, the ground truth on it, the predictions and the predictions post-thresholding """
        correctly_removed, falsely_removed, falsely_remaining = self.pred_filters[:-1]
        det_ind = np.where(img_name == np.asarray(self.imgs_list))[0]

        img_name = img_name.split("/")[-1]
        img_path = self.img_source_path+img_name
        img = Image.open(img_path)
        img_array = np.array(img)
        im = visualize_image(img_array, self.pred_boxes[det_ind], self.pred_classes.astype("int")[det_ind], self.pred_scores[det_ind], self.label_map, min_score_thresh=0.0001)
        gt_im = visualize_image(img_array, self.gt_boxes[det_ind], self.gt_classes.astype("int")[det_ind], np.ones_like(self.pred_scores[det_ind]), self.label_map, min_score_thresh=0.0001)
        
        fig, axs = plt.subplots(2, 2, figsize=(20, 12))
        axs[0, 0].imshow(img_array, cmap='viridis')
        axs[0, 0].axis('off') 
        axs[0, 0].set_title('Image',fontsize=12)
        axs[0, 1].imshow(gt_im, cmap='viridis')
        axs[0, 1].axis('off')
        axs[0, 1].set_title('Ground Truth',fontsize=12)
        axs[1, 0].imshow(im, cmap='viridis')
        axs[1, 0].axis('off')
        axs[1, 0].set_title('Predictions',fontsize=12)
        axs[1, 1].imshow(im, cmap='viridis')
        axs[1, 1].axis('off')
        axs[1, 1].set_title('Post-thresholding',fontsize=12)

        labels = {
            "Correctly removed": 'green',
            "Falsely removed": 'red',
            "Falsely remaining": 'magenta'
        }
        values = {
            "Correctly removed": correctly_removed,
            "Falsely removed":  falsely_removed,
            "Falsely remaining": falsely_remaining
        }

        for label_key, color_value in labels.items():
            l = 0
            for i in det_ind[values[label_key][det_ind]]:
                x, y, w, h = [self.pred_boxes[i][1], self.pred_boxes[i][0], self.pred_boxes[i][3] - self.pred_boxes[i][1], self.pred_boxes[i][2] - self.pred_boxes[i][0]]
                legend_label = label_key if l==0 else "_nolegend_"
                pred_box = Rectangle((x, y), w, h, linewidth=1, edgecolor='none', facecolor=color_value, label=legend_label)
                axs[1, 1].add_patch(pred_box)
                l = 1

        if len(det_ind)>0: axs[1, 1].legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(self.source_path+"/thr_plots/"+title+"/"+img_name.split(".")[0]+"_postthresholding.png", bbox_inches='tight')   
        plt.close() 

    def _collect_postthresholding(self):
        """ Collects and plots 10 images with most correct/false removal, most falsely remaining, and 10 random images with no removals """
        def _process_images(img_indices, label):
            filtered_imgs = self.imgs_list[img_indices]
            item_counts = Counter(filtered_imgs)
            top_10 = [item[0] for item in item_counts.most_common(10)]
            [self._draw_postthresholding(im_name, label) for im_name in top_10]
        folders_names = ["top_correctremove", "top_falserremove", "top_falseremain", "random_noremoval"]
        for f_name in folders_names:
            if os.path.exists(self.source_path+"/thr_plots/"+f_name+"/"):
                shutil.rmtree(self.source_path+"/thr_plots/"+f_name+"/")
            os.makedirs(self.source_path+"/thr_plots/"+f_name+"/")
        correctly_removed, falsely_removed, falsely_remaining, opt_thr = self.pred_filters
        _process_images(correctly_removed, folders_names[0])
        _process_images(falsely_removed, folders_names[1])
        _process_images(falsely_remaining, folders_names[2])
        
        # No removal
        indices_dict = defaultdict(list)
        for index, img_name in enumerate(self.imgs_list):
            indices_dict[img_name].append(index)
        unique_images = np.unique(self.imgs_list)
        inds = np.asarray([np.all(self.opt_uncert[indices_dict[iname]] < opt_thr) for iname in unique_images])
        filtered_imgs = unique_images[inds]
        random_indexes = random.sample(range(len(filtered_imgs)), 10)
        [self._draw_postthresholding(im_name, folders_names[3]) for im_name in filtered_imgs[random_indexes]]
        
    def _read_predictions(self):
        """ Reads and assigns predictions """
        f = open(self.source_path+'/validate_results.txt',"r")
        dets = f.readlines() 
        detections = [ast.literal_eval(d.replace('inf','2e308')) for d in dets] 
        self.gt_boxes = np.asarray([detections[i]["gt_bbox"] for i in range(len(detections))])
        self.pred_boxes = np.asarray([detections[i]["bbox"] for i in range(len(detections))])
        self.ious = calc_iou_np(self.gt_boxes, self.pred_boxes)
        self.gt_boxes = self.gt_boxes[self.ious>0.0]
        self.pred_boxes = self.pred_boxes[self.ious>0.0]
        self.imgs_list = self._extract_det(detections, self.ious, "image_name")
        self.gt_classes = self._extract_det(detections, self.ious, "gt_class")
        self.pred_classes = self._extract_det(detections, self.ious, "class")
        self.pred_scores = self._extract_det(detections, self.ious, "score")

        if "entropy" in detections[0]: self.entropy = self._extract_det(detections, self.ious, "entropy")
        if "iso_percls_entropy" in detections[0]: self.calib_entropy = self._extract_det(detections, self.ious, "iso_percls_entropy")
        if "iso_perclscoo_albox" in detections[0]: self.calib_albox = self._extract_det(detections, self.ious, "iso_perclscoo_albox")
        if "uncalib_albox" in detections[0]: self.albox = self._extract_det(detections, self.ious, "uncalib_albox")
        if "uncalib_mcbox" in detections[0]: self.mcbox = self._extract_det(detections, self.ious, "uncalib_mcbox")
        if "uncalib_mcclass" in detections[0]: self.mcclass = self._extract_det(detections, self.ious, "uncalib_mcclass")

        self.ious=self.ious[self.ious>0.0]
    
    def _plot_validheat(self):
        """ Compare relative uncertainty distribution in all detections as a heatmap"""
        if not os.path.exists(self.source_path+"/heat_plots"):
            os.makedirs(self.source_path+"/heat_plots")

        def _heatmap_plot(pred, value, plot_title, save_title, uncert=True):
            """ Plot heatmap on the whole validation dataset """
            fig = plt.figure(figsize=(12,7))
            mask_size = get_dataset_data(self.source_path)[-2]
            mask = np.zeros(mask_size)
            normalizer = np.zeros(mask_size)
            i = 0
            for bb in pred:
                mask[int(bb[0]):int(bb[2]),int(bb[1]):int(bb[3])] += value[i]
                normalizer[int(bb[0]):int(bb[2]),int(bb[1]):int(bb[3])] += 1
                i+=1
            if uncert: mask = np.nan_to_num(mask/normalizer)
            ax = plt.gca()
            im = ax.imshow(mask, cmap='jet', interpolation='nearest')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            plt.colorbar(im,cax=cax, label=plot_title)
            plt.tight_layout()
            plt.savefig(self.source_path +'/heat_plots/heat_'+save_title)                
            plt.close()

        _heatmap_plot(self.gt_boxes, np.mean(np.ones_like(self.gt_boxes), axis=1), "Location Frequency", 'gt_loc.png', uncert=False)

        if self.albox is not None: 
            _heatmap_plot(self.pred_boxes, np.mean(self.albox, axis=1), "Norm. Mean Aleatoric Uncertainty", 'uncert_albox.png')
            rmse = np.sqrt(np.mean((np.abs(self.gt_boxes-self.pred_boxes) - self.albox)**2, axis=-1))   
            _heatmap_plot(self.gt_boxes, rmse, "RMSE Residuals/Aleatoric Uncertainty", 'rmse_albox.png')

        if self.calib_albox is not None: 
            _heatmap_plot(self.pred_boxes, np.mean(self.calib_albox, axis=1), "Norm. Mean Aleatoric Uncertainty", 'uncert_calib_albox.png')
            rmse = np.sqrt(np.mean((np.abs(self.gt_boxes-self.pred_boxes) - self.calib_albox)**2, axis=-1))   
            _heatmap_plot(self.gt_boxes, rmse, "RMSE Residuals/Aleatoric Uncertainty", 'rmse_calib_albox.png')
            
        if self.entropy is not None: _heatmap_plot(self.pred_boxes, self.entropy, "Entropy", 'entropy.png')       

        if self.calib_entropy is not None: _heatmap_plot(self.pred_boxes, self.calib_entropy, "Entropy", 'calib_entropy.png')

        _heatmap_plot(self.pred_boxes[self.pred_filters[0]], self.opt_uncert[self.pred_filters[0]], "Optimal Uncertainty Correctly Removed", 'opt_uncert_cr.png')
        _heatmap_plot(self.pred_boxes[self.pred_filters[1]], self.opt_uncert[self.pred_filters[1]], "Optimal Uncertainty Falsely Removed", 'opt_uncert_fr.png')
        _heatmap_plot(self.pred_boxes[self.pred_filters[2]], self.opt_uncert[self.pred_filters[2]], "Optimal Uncertainty Falsely Remaining", 'opt_uncert_fk.png')

        # for i in range(self.num_classes):
        #     _heatmap_plot(self.pred_boxes[self.gt_classes==i+1], self.calib_entropy[self.gt_classes==i+1], '_cls_'+str(i+1)+'.png')

    def _plot_metricsspider(self):
        """ PLots different metrics pre- and post-thresholding as a spider plot """        
        if not os.path.exists(self.source_path+"/thr_plots"):
            os.makedirs(self.source_path+"/thr_plots")
        keys = ['mIoU', 'Acc', '%Det.', r'%CD$_t$s', r'%FD$_t$s']   
        metrics_all_data, metrics_removed_data, metrics_remaining_data = [], [], []
        for iou_thr in IOU_THRS:
            correct_detections = np.asarray((self.ious >= iou_thr) * self.tps_class, dtype=int)
            thr = roc_metrics(self.opt_uncert, correct_detections)[0]
            metrics_all_data.append([np.mean(self.ious), np.mean(self.tps_class), 1, 1, 1]) 
            metrics_removed_data.append([np.mean(self.ious[self.opt_uncert >= thr]), 
                                         np.mean(self.tps_class[self.opt_uncert >= thr]),                                         
                                         len(correct_detections[self.opt_uncert >= thr])/len(correct_detections), 
                                         np.sum(correct_detections[self.opt_uncert >= thr])/np.sum(correct_detections),
                                         np.sum(1-correct_detections[self.opt_uncert >= thr])/np.sum(1-correct_detections)])
            
            metrics_remaining_data.append([np.mean(self.ious[self.opt_uncert < thr]), 
                                         np.mean(self.tps_class[self.opt_uncert < thr]),                                         
                                         len(correct_detections[self.opt_uncert < thr])/len(correct_detections), 
                                         np.sum(correct_detections[self.opt_uncert < thr])/np.sum(correct_detections),
                                         np.sum(1-correct_detections[self.opt_uncert < thr])/np.sum(1-correct_detections)])
            
        collected_data = [np.mean(metrics_all_data,axis=0)*100, np.mean(metrics_removed_data,axis=0)*100, np.mean(metrics_remaining_data,axis=0)*100]
        labels = ['All Det.', 'Removed Det.', 'Remaining Det.']
        colors = ['#FD7120', '#00BFFF', '#008000']        
        angles = [i * 2 * np.pi / len(keys) for i in range(len(keys))]
        
        plt.figure(figsize=(3, 3))
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 10,
            'hatch.linewidth': 0.1
        })
        
        plt.clf()
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles, keys, color="black")
        plt.yticks([20, 40, 60, 80], ['20', '40', '60', '80'], color="black")
        plt.ylim(0, 100)
        
        for _, (det, label, color) in enumerate(zip(collected_data, labels, colors)):
            ax.set_rlabel_position(0)
            ax.plot(angles, det, color=color, linestyle='solid', linewidth=1, label=label)
            ax.fill(angles, det, color=color, alpha=0.6)
        
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.savefig(self.source_path +'/thr_plots/spider_metrics.png', bbox_inches='tight')  
        plt.close()

if __name__ == "__main__":
    while True:
        model_name = input("Enter model name, e.g., EXP_KITTI_allclasses_lossattV1: ")
        if os.path.exists(PARENT_DIR + '/results/validation/' + model_name):
            MainUncertViz(model_name)
            break
        else:
            print("Error: The provided path does not exist. Please try again.")