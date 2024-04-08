# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Uncertainty analysis epistemic vs. aleatoric """


import os
import ast
import argparse

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import seaborn as sns
from PIL import Image
from brisque import BRISQUE

from utils_box import relativize_uncert, calc_iou_np
from dataset_data import get_dataset_data

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 15})
plt.rcParams['hatch.linewidth'] = 0.1

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU') 
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class EpistemicVSAleatoric:
    """ Compares aleatoric and epistemic uncertainty and splits them into quadrants"""
    def __init__(self, model_name, im_numbs=10, iou_thr=0.1, uncert="uncalib", ent=False):
        """Sets up the necessary attributes to compare the uncertainties

        Args:
            model_name (str): Model name to find path to validation_results.txt with the val results
            im_numbs (int, optional): Number of images to select, if less available it will be changed accordingly. Defaults to 10.
            iou_thr (float, optional): IoU threshold to filter detections. Defaults to 0.1.
            uncert (str, optional): Name of the uncertainty, such as uncalib, iso_percoo, iso _all. Defaults to "uncalib".
            ent (bool, optional): Selects selects classification entropy to compare with aleatoric, otherwise it is epistemic localization.
        """
        self.ent = ent
        self.source_path = PARENT_DIR + '/results/validation/' + model_name
        self.img_source_path, self.class_names = get_dataset_data(model_name)[1:3]
        self.uncert = uncert
        self.numb = im_numbs
        self.iou_thr = iou_thr
        self.grid_size = 2 # Start at 2x2

    @staticmethod
    def _get_cell_indices(grid_size, uncert_data):
        # Split data into grid cells and assign each data point to a cell
        x_min, x_max = np.min(uncert_data[:,0]), np.max(uncert_data[:,0])
        y_min, y_max = np.min(uncert_data[:,1]), np.max(uncert_data[:,1])
        x_buffer = (x_max - x_min) * 0.05 
        y_buffer = (y_max - y_min) * 0.05
        x_step = (x_max + x_buffer - x_min) / grid_size
        y_step = (y_max + y_buffer - y_min) / grid_size

        cell_indices = []
        for i in range(uncert_data.shape[0]):
            row = int(min((uncert_data[i,1] - y_min) // y_step, grid_size - 1))
            col = int(min((uncert_data[i,0] - x_min) // x_step, grid_size - 1))
            cell_index = row * grid_size + col
            cell_indices.append(cell_index)
        return np.asarray(cell_indices), x_min, x_step, y_min, y_step

    def _save_crops(self, img_names, coords, label):
        # Plot crops on one image
        fig = plt.figure(figsize=(20,12))
        num_images = len(img_names)
        if num_images > 0:
            # Determine the number of rows and columns dynamically
            num_cols = int(np.ceil(np.sqrt(num_images)))
            num_rows = int(np.ceil(num_images / num_cols))
            for i in range(num_images):        
                ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
                im = Image.open(PARENT_DIR + "/datasets/" + self.img_source_path + img_names[i]) 
                im = im.crop((coords[i][1], coords[i][0], coords[i][3], coords[i][2]))
                ax1.imshow(im)
        plt.savefig(self.source_path+"/"+label+"_"+self.uncert+"_"+str(self.iou_thr)+"_"+str(self.numb)+".png")
        plt.close()     

    def plot_unc_metrics(self, data_list, labels, colors):
        """Plots uncertainty metrics

        Args:
            data_list (list): List containing either one, none or two lists for hal_lep and lep_high depending on availability.
            labels (list): List containing either one, none or two labels for the lists in data_list.
            colors (list): List containing either one, none or two colors for the lists in data_list.
        """
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 12))
        for i, (data, label, color) in enumerate(zip(data_list, labels, colors)):
            x = np.arange(len(data[0]))
            if len(x) > 1:
                f = np.poly1d(np.polyfit(x, data[0], 3))
                axes[0, 0].plot(x, f(x), color=color, label=label)

            axes[0, 0].scatter(x, data[0], color=color, alpha=0.2)
            axes[0, 0].set_xlabel("Detection #")
            axes[0, 0].set_ylabel("IoU")
            axes[0, 0].legend(prop={'size': 10}, frameon=False, handletextpad=0.1)

            cp = [self.class_names.index(data[1][j]) for j in range(len(data[1]))]
            y = [np.count_nonzero((np.equal(cp, j))) for j in range(len(self.class_names))]

            axes[0, 1].bar(np.arange(len(self.class_names)) - 0.2, y, width=0.2, align='edge', label="Pred. " + label)

            cp = [self.class_names.index(data[2][j]) for j in range(len(data[2]))]
            y = [np.count_nonzero((np.equal(cp, j))) for j in range(len(self.class_names))]

            axes[0, 1].bar(np.arange(len(self.class_names)) + 0.2, y, width=0.2, align='edge', label="GT " + label)
            axes[0, 1].set_xticks(np.arange(len(self.class_names)))
            axes[0, 1].set_xticklabels(self.class_names, rotation=90)
            axes[0, 1].set_ylabel("Count")
            axes[0, 1].legend(prop={'size': 10}, frameon=False, handletextpad=0.1)

            if len(x) > 1:
                f = np.poly1d(np.polyfit(x, data[3], 3))
                axes[0, 2].plot(x, f(x), color=color)
                
            axes[0, 2].scatter(x, data[3], label=label, color=color, alpha=0.2)
            axes[0, 2].set_xlabel("Detection #")
            axes[0, 2].set_ylabel("Score")
            
            if len(x) > 1:
                f = np.poly1d(np.polyfit(x, data[4], 3))
                axes[1, 0].plot(x, f(x), color=color)

            axes[1, 0].scatter(x, data[4], label=label, color=color, alpha=0.2)
            axes[1, 0].set_xlabel("Detection #")
            axes[1, 0].set_ylabel("Entropy")

            if len(x) > 1:
                f = np.poly1d(np.polyfit(x, data[5], 3))
                axes[1, 1].plot(x, f(x), color=color)

            axes[1, 1].scatter(x, data[5], label=label, color=color, alpha=0.2)
            axes[1, 1].set_xlabel("Detection #")
            axes[1, 1].set_ylabel("Pred. Area")

            if len(x) > 1:
                f = np.poly1d(np.polyfit(x, data[6], 3))
                axes[1, 2].plot(x, f(x), color=color)

            axes[1, 2].scatter(x, data[6], label=label, color=color, alpha=0.2)
            axes[1, 2].set_xlabel("Detection #")
            axes[1, 2].set_ylabel("Pred. Crop BRISQUE")

            if len(x) > 1:
                f = np.poly1d(np.polyfit(x, data[7], 3))
                axes[2, 0].plot(x, f(x), label="EP " + label)
            axes[2, 0].scatter(x, data[7], alpha=0.2,)
            
            if len(x) > 1:
                f = np.poly1d(np.polyfit(x, data[8], 3))
                axes[2, 0].plot(x, f(x), label="AL " + label)

            axes[2, 0].scatter(x, data[8], alpha=0.2)
            axes[2, 0].set_xlabel("Detection #")
            axes[2, 0].set_ylabel("Norm. Uncertainty")
            axes[2, 0].legend(prop={'size': 10}, frameon=False, handletextpad=0.1)


        fig.tight_layout()
        plt.savefig(self.source_path + "/metrics_comp_" + self.uncert + "_" + str(self.iou_thr)+"_"+ str(self.numb) + ".png")
        plt.close()

    def get_metrics_comp(self, filtered_detections, img_names, bbox, gt_bbox):
        # Collect and calculate metrics for hal_lep and lep_hal by receiving filtered detections
        data = []        
        data.append(calc_iou_np(gt_bbox, bbox))
        data.append([self.class_names[int(filtered_detections[i]["class"]-1)] for i in range(len(filtered_detections))])
        data.append([self.class_names[int(filtered_detections[i]["gt_class"]-1)] for i in range(len(filtered_detections))])
        data.append([filtered_detections[i]["score"] for i in range(len(filtered_detections))])
        data.append([filtered_detections[i]["entropy"] for i in range(len(filtered_detections))])
        data.append((np.stack(bbox)[:,3] - np.stack(bbox)[:,1])*(np.stack(bbox)[:,2] - np.stack(bbox)[:,0]))
        data_brisq = []
        for i in range(len(img_names)):    
            obj = BRISQUE(url=False)
            data_brisq.append(np.nan_to_num(obj.score(np.asarray(Image.open(PARENT_DIR + "/datasets/" + self.img_source_path +img_names[i]).crop((bbox[i][1], bbox[i][0], bbox[i][3], bbox[i][2]))))))
        data += [data_brisq]
        return data
    
    def _plot_uncert_dist(self, data, unc_type):
        # Plot the distribution of a an uncertainty
        sns.histplot(data, kde=True, color='skyblue', bins=10)
        plt.xlabel(unc_type+' Uncertainty')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(self.source_path+"/dist_"+unc_type+"_"+self.uncert+"_iou_"+str(self.iou_thr)+".png")
        plt.close()

    def get_plot_grid(self, av_al, av_mc): 
        # Split uncertainty space into grid and plot it      
        uncert_data = np.stack([av_al, av_mc],axis=-1)
        min_samples = 3
        cell_indices, x_min, x_step, y_min, y_step = self._get_cell_indices(self.grid_size, uncert_data)
        top_left_empty = len(cell_indices[cell_indices==self.grid_size**2-self.grid_size]) < min_samples
        bottom_right_empty = len(cell_indices[cell_indices==self.grid_size-1]) < min_samples

        # Decrease grid size if top-left and bottom-right cells are empty
        if top_left_empty == False or bottom_right_empty == False:
            while top_left_empty == False or bottom_right_empty == False:
                self.grid_size += 1
                cell_indices = self._get_cell_indices(self.grid_size, uncert_data)[0]
                top_left_empty = len(cell_indices[cell_indices==self.grid_size**2-self.grid_size]) < min_samples
                bottom_right_empty = len(cell_indices[cell_indices==self.grid_size-1]) < min_samples
            # Go one step back to where it is not empty
            self.grid_size -= 1
            cell_indices, x_min, x_step, y_min, y_step = self._get_cell_indices(self.grid_size, uncert_data)

        print("Found grid size: ", self.grid_size)

        # Define colors for each grid cell
        colors = plt.cm.rainbow(np.linspace(0, 1, self.grid_size**2))
        cell_colors = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_colors.append(colors[i*self.grid_size + j])

        self._plot_uncert_dist(uncert_data[:,0], "AL")
        if self.ent:
            self._plot_uncert_dist(uncert_data[:,1],  "ENT")
        else:
            self._plot_uncert_dist(uncert_data[:,1],  "EP")
        # Plot the data with colors based on grid cell
        fig = plt.figure()
        plt.scatter(uncert_data[:,0], uncert_data[:,1], c=[cell_colors[i] for i in cell_indices])
        # plt.grid(linestyle='--', linewidth=0.5)
        for i in range(1, self.grid_size):
            plt.axvline(x=x_min + i * x_step, linestyle='--', linewidth=0.5)
            plt.axhline(y=y_min + i * y_step, linestyle='--', linewidth=0.5)

        if self.ent: plt.title("Distribution Loc AL vs cls ENT")
        else: plt.title("Distribution Localization AL vs EP")
        plt.xlabel("Aleatoric")
        plt.ylabel("Epistemic")
        plt.tight_layout()
        plt.savefig(self.source_path+"/clust_"+self.uncert+"_iou_"+str(self.iou_thr)+".png")

        from scipy.stats import gaussian_kde

        # Create a 2D KDE (Kernel Density Estimation) of your data
        kde = gaussian_kde([uncert_data[:, 0], uncert_data[:, 1]])
        # Define the boundaries of the quadrants
        x_min, x_max = uncert_data[:, 0].min(), uncert_data[:, 0].max()
        y_min, y_max = uncert_data[:, 1].min(), uncert_data[:, 1].max()
        # Create a grid of points to evaluate the KDE
        x, y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        positions = np.vstack([x.ravel(), y.ravel()])
        values = kde(positions)
        # Reshape the values to match the grid shape
        values = values.reshape(x.shape)
        # Plot the PDF using contourf and color based on quadrants
        plt.contourf(x, y, values, cmap='jet', alpha=0.7)

        plt.colorbar(label='PDF')
        plt.tight_layout()
        plt.savefig(self.source_path+"/clust_pdf_"+self.uncert+"_iou_"+str(self.iou_thr)+".png")
        plt.close()

        return cell_indices

    def save_highlow(self,detections,gt_bbox,bbox,hal_lep_ind,lal_hep_ind):
        # Save hal_lep and hep_lal detections info and crops
        if os.path.exists(self.source_path+'/hal_lep_'+self.uncert+'_iou_'+str(self.iou_thr)+'.txt'):
            os.remove(self.source_path+'/hal_lep_'+self.uncert+'_iou_'+str(self.iou_thr)+'.txt')
        with open(self.source_path+'/hal_lep_'+self.uncert+'_iou_'+str(self.iou_thr)+'.txt', 'w') as f:
            for itm in detections[hal_lep_ind]:
                f.write(str(itm)+"\n")

        if os.path.exists(self.source_path+'/lal_hep_'+self.uncert+'_iou_'+str(self.iou_thr)+'.txt'):
            os.remove(self.source_path+'/lal_hep_'+self.uncert+'_iou_'+str(self.iou_thr)+'.txt')
        with open(self.source_path+'/lal_hep_'+self.uncert+'_iou_'+str(self.iou_thr)+'.txt', 'w') as f:
            for itm in detections[lal_hep_ind]:
                f.write(str(itm)+"\n")
        # Save crops
        self._save_crops(self.imgs_hal_lep, gt_bbox[hal_lep_ind], "gt_im_hal_lep")
        self._save_crops(self.imgs_hal_lep, bbox[hal_lep_ind], "pred_im_hal_lep")
        self._save_crops(self.imgs_lal_hep, gt_bbox[lal_hep_ind], "gt_im_lal_hep")
        self._save_crops(self.imgs_lal_hep, bbox[lal_hep_ind], "pred_im_lal_hep")
    
    def compare_epal(self):
        # Main function for the class
        # Read uncertainties
        f = open(self.source_path+'/validate_results.txt',"r")
        dets = f.readlines() 
        detections = [ast.literal_eval(det.replace('inf','2e308')) for det in dets] 
        bbox = [det["bbox"] for det in detections]
        gt_bbox = [det["gt_bbox"] for det in detections]  
        ious = calc_iou_np(gt_bbox, bbox)
        detections = np.asarray(detections)[ious>self.iou_thr]
        bbox = np.asarray(bbox)[ious>self.iou_thr]
        gt_bbox = np.asarray(gt_bbox)[ious>self.iou_thr]

        uncert_albox = np.nan_to_num([det[self.uncert+"_albox"] for det in detections])
        uncert_albox = relativize_uncert(bbox, uncert_albox)
        av_al = np.mean(uncert_albox, axis=-1)

        # Select which uncertainty along aleatoric
        if self.ent:
            print("No epistemic uncertainty, using classification entropy instead")
            av_mc = np.asarray([det["entropy"] for det in detections]) 
            self.source_path +="/compare_epal/entropy/"
        else:
            uncert_mcbox = np.nan_to_num([det[self.uncert+"_mcbox"] for det in detections])
            uncert_mcbox = relativize_uncert(bbox, uncert_mcbox)      
            av_mc = np.mean(uncert_mcbox, axis=-1)  
            self.source_path +="/compare_epal/mcdropout/"

        if not os.path.exists(self.source_path):
            os.makedirs(self.source_path) 

        # Get grid split
        cell_indices = self.get_plot_grid(av_al, av_mc)

        hal_lep_ind = np.where(cell_indices==self.grid_size-1)[0][:self.numb]
        lal_hep_ind = np.where(cell_indices==self.grid_size**2-self.grid_size)[0][:self.numb]

        av_al = (av_al - np.min(av_al)) / (np.max(av_al) - np.min(av_al))
        av_mc = (av_mc - np.min(av_mc)) / (np.max(av_mc) - np.min(av_mc))

        # Gather image names
        self.imgs_hal_lep = [detections[hal_lep_ind][i]["image_name"] for i in range(len(detections[hal_lep_ind]))]
        self.imgs_lal_hep = [detections[lal_hep_ind][i]["image_name"] for i in range(len(detections[lal_hep_ind]))]

        # Save crops and highlow txt
        self.save_highlow(detections,gt_bbox,bbox,hal_lep_ind,lal_hep_ind)

        # Retrieve metrics        
        available_metrics = []
        available_labels = []
        available_colors = []
        if len(hal_lep_ind) > 0:
            metrics_hal_lep = self.get_metrics_comp(detections[hal_lep_ind], self.imgs_hal_lep, bbox[hal_lep_ind], gt_bbox[hal_lep_ind])
            available_metrics+=[metrics_hal_lep+[av_mc[hal_lep_ind]]+[av_al[hal_lep_ind]]]
            available_labels+=["High Al-Low EP"]
            available_colors+=["green"]
        if len(lal_hep_ind) != 0:
            metrics_lal_hep = self.get_metrics_comp(detections[lal_hep_ind], self.imgs_lal_hep, bbox[lal_hep_ind], gt_bbox[lal_hep_ind])
            available_metrics+=[metrics_lal_hep+[av_mc[lal_hep_ind]]+[av_al[lal_hep_ind]]]
            available_labels+=["Low Al-High EP"]
            available_colors+=["red"]

        self.plot_unc_metrics(available_metrics, available_labels, available_colors)
        print("Comparison complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command-line tool to compare Epistemic vs Aleatoric uncertainty")

    parser.add_argument("--model_name", type=str, help="Model name", default=None)
    parser.add_argument("--im_numbs", type=int, help="Number of images", default=10)
    parser.add_argument("--iou_thr", type=float, help="IoU threshold", default=0.1)
    parser.add_argument("--uncert", type=str, help="Uncertainty type", default="uncalib")
    parser.add_argument("--ent", type=bool, help="Select entropy", default=False)

    args = parser.parse_args()
    if args.model_name is not None:
        EpistemicVSAleatoric(args.model_name, args.im_numbs, args.iou_thr, args.uncert, args.ent).compare_epal()
    else:
        print("Command-line arguments not provided. Asking for user input.") 
        try:  
          model_name = str(input("Enter model name, e.g., EXP_KITTI_allclasses_mcdropout_lossattVhead.3_ckpt20: "))
          im_numbs = int(input("Enter limit on number of images, default is 10: ") or 10)
          iou_thr = float(input("Enter iou threshold for filtering, default is 0.1: ") or 0.1)
          uncert = str(input("Enter uncert type, default is 'uncalib': ") or "uncalib")
          ent = bool(input("Enter if entropy to be used, default is False: ") or False)
          EpistemicVSAleatoric(model_name, im_numbs, iou_thr, uncert, ent).compare_epal()
        except ValueError:
          print("Invalid input. Please enter valid values.")











