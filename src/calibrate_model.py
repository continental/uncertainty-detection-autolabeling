# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Calibrate model on validation set """


import os
import shutil

import tensorflow as tf
import numpy as np

from calibrate_classification import ClassificationCalib
from calibrate_regression import RegressionCalib
from utils_box import calc_iou_np
from utils_extra import update_arrays, gt_box_assigner
from dataset_data import get_dataset_data


class Calibrate:
    """ Class for the validation on a dataset """
    def __init__(self, model_params, img_names, driver, gt_classes, gt_coords, save_dir, general_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        """  Constructs all the necessary attributes for the validation 
        
        Args:
        model_params (dict): Dictionary of model parameters
        img_names (list): Image names
        driver (object): Driver for serving single or batch images
        gt_classes (array): Ground truth classes
        gt_coords (array): Ground truth coordinates
        save_dir (str): Path with model name, to save calibration models
        general_path (str): Path to working space

        """  
        
        self.saving_path = general_path+"/results/calibration/"
        self.model_params = model_params
        self.img_names = img_names
        self.driver = driver
        self.gt_classes = gt_classes
        self.gt_coords = gt_coords
        self.save_dir = save_dir
        self.filtered_boxes = np.array([])
        self.filtered_scores = np.array([])
        self.filtered_classes = np.array([])
        if model_params["enable_softmax"]:
            self.filtered_classes_logits = np.array([])
            if  model_params['mc_classheadrate'] or model_params['mc_dropoutrate']:  self.filtered_classes_mccls = np.array([])
            else: self.filtered_classes_mccls = None
            
        if model_params["loss_attenuation"]: self.filtered_uncert_albox = np.array([])
        else:  self.filtered_uncert_albox = None

        if  model_params['mc_boxheadrate'] or model_params['mc_dropoutrate']: self.filtered_uncert_mcbox= np.array([])
        else: self.filtered_uncert_mcbox = None

        self.filtered_gt_classes = np.array([])
        self.filtered_gt_coords = np.array([])

    def gather_detections(self):
        """ Predict on every image and gather results for calibration """

        ind = 0
        for im_name in self.img_names:
            image_npath = get_dataset_data(self.save_dir, im_name=im_name)[-1]
            image_file = tf.io.read_file(self.save_dir.split("models")[0]+"/datasets/"+ image_npath)
            im_name = tf.io.decode_image(image_file, channels=3, expand_animations=False)
            im_name = tf.expand_dims(im_name, axis=0)   
            detections_bs = self.driver.serve(im_name)
            if self.model_params["enable_softmax"]:
                box, scores, classe, _, logits = tf.nest.map_structure(np.array, detections_bs) 
            else: 
                box, scores, classe, _ = tf.nest.map_structure(np.array, detections_bs) 
                   
            if (self.model_params['mc_boxheadrate'] or self.model_params['mc_dropoutrate']) and not self.model_params["loss_attenuation"]: 
                mc_boxhead = box[:,:,4:]
                al_boxhead = None
            elif (self.model_params['mc_boxheadrate'] or self.model_params['mc_dropoutrate']) and self.model_params["loss_attenuation"]: 
                al_boxhead = box[:,:,4:8]
                mc_boxhead = box[:,:,8:]
            elif not (self.model_params['mc_boxheadrate'] or self.model_params['mc_dropoutrate']) and self.model_params["loss_attenuation"]:
                al_boxhead = box[:,:,4:] 
                mc_boxhead = None
            else:
                al_boxhead = None
                mc_boxhead = None
            if self.model_params['mc_classheadrate'] or self.model_params['mc_dropoutrate']: 
                mc_classhead = classe[:,:,1:]
                classe = classe[:,:,0]
            else:
                mc_classhead = None
            
            if mc_boxhead is not None or al_boxhead is not None: box = box[:,:,:4]       
            
            for i in range(box[0].shape[0]):
                # remove -1 background, 100 NMS output
                if self.gt_classes[ind][i] >= 0:
                    self.filtered_gt_classes = np.append(self.filtered_gt_classes, self.gt_classes[ind][i]-1)
                    self.filtered_gt_coords = update_arrays(self.filtered_gt_coords, self.gt_coords[ind], i) 
                    correct_index = gt_box_assigner(self.model_params['assign_gt_box'], self.gt_coords[ind], box[0], i)
                    self.filtered_scores = np.append(self.filtered_scores, scores[0][correct_index])
                    self.filtered_classes = np.append(self.filtered_classes, classe[0][correct_index])
                    self.filtered_boxes = update_arrays(self.filtered_boxes, box[0], correct_index)

                    if self.model_params["calibrate_regression"]:
                        if self.model_params["loss_attenuation"]:
                            self.filtered_uncert_albox = update_arrays(self.filtered_uncert_albox, al_boxhead[0], correct_index)
                        if self.model_params['mc_boxheadrate'] or self.model_params['mc_dropoutrate']:
                            self.filtered_uncert_mcbox = update_arrays(self.filtered_uncert_mcbox, mc_boxhead[0], correct_index)
                    if self.model_params["calibrate_classification"] and self.model_params["enable_softmax"]:
                        self.filtered_classes_logits = update_arrays(self.filtered_classes_logits, logits[0], correct_index)
                        if self.model_params['mc_classheadrate'] or self.model_params['mc_dropoutrate']:
                            self.filtered_classes_mccls = update_arrays(self.filtered_classes_mccls, mc_classhead[0], correct_index)
            ind += 1
            
    def calibrate_regclas(self):   
        """ Run classification and regression calibration """       

        if os.path.exists(self.saving_path+self.save_dir.split("/")[-1]):
            shutil.rmtree(self.saving_path+self.save_dir.split("/")[-1])
        os.makedirs(self.saving_path+self.save_dir.split("/")[-1])
        self.gather_detections()

        ious = calc_iou_np(self.filtered_gt_coords, self.filtered_boxes)    
        # min_score = self.model_params["nms_configs"]["score_thresh"] or 0.4        
        # filterd = [(self.filtered_scores>min_score)*(ious > 0.5)]  
        filterd = ious > 0.0
        self.filtered_gt_classes = self.filtered_gt_classes[filterd]
        self.filtered_classes_logits = self.filtered_classes_logits[filterd]
        
        self.filtered_gt_coords = self.filtered_gt_coords[filterd]        
        self.filtered_boxes = self.filtered_boxes[filterd]

        if self.filtered_classes_mccls is not None: self.filtered_classes_mccls = self.filtered_classes_mccls[filterd]
        if self.filtered_uncert_albox is not None: self.filtered_uncert_albox = self.filtered_uncert_albox[filterd]
        if self.filtered_uncert_mcbox is not None: self.filtered_uncert_mcbox = self.filtered_uncert_mcbox[filterd]

        if self.model_params["calibrate_classification"] and self.model_params["enable_softmax"] and self.model_params["num_classes"]>1:
            os.makedirs(self.saving_path+self.save_dir.split("/")[-1]+ '/classification/')
            ClassificationCalib(self.filtered_gt_classes, self.filtered_classes_logits, self.filtered_classes_mccls, self.save_dir, self.saving_path).class_calibration()
            
        if self.model_params["calibrate_regression"]:
            if self.filtered_uncert_albox is not None:                
                os.makedirs(self.saving_path+self.save_dir.split("/")[-1]+ '/regression/aleatoric/')
                RegressionCalib(self.filtered_gt_coords, self.filtered_boxes, np.nan_to_num(self.filtered_uncert_albox), self.filtered_gt_classes, self.save_dir.split("/")[-1]+ '/regression/aleatoric/', self.saving_path).box_uncert_calibration()
            if self.filtered_uncert_mcbox is not None: 
                os.makedirs(self.saving_path+self.save_dir.split("/")[-1]+ '/regression/mcdropout/')
                RegressionCalib(self.filtered_gt_coords, self.filtered_boxes, np.nan_to_num(self.filtered_uncert_mcbox), self.filtered_gt_classes, self.save_dir.split("/")[-1]+ '/regression/mcdropout/', self.saving_path).box_uncert_calibration()
        print("Calibration is finished on all the validation dataset")