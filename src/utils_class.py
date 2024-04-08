# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Utils for classification functions """


import os
from absl import logging

import numpy as np
from matplotlib import pyplot as plt
import pickle
import tensorflow_probability as tfp
import tensorflow as tf

import label_util

def label_map_extractor(label_map_path):
  """ Extract a dictionary with class labels and IDs from txt file """
  ids = []
  names = []
  label_map = {}
  with open(label_map_path, 'r') as file:
      lines = file.readlines()
      for line in lines:
          if 'name' in line:
            names.append(line.split(":")[1].strip().strip("'"))             
          elif 'id' in line:
            ids.append(int(line.split(":")[1].strip()))
  for i in range(len(ids)): label_map[names[i]] = ids[i]
  return label_map

def stable_softmax(logits):
  """ Applies numerically stable softmax on logits """
  softmax_logits = []
  for x in logits:
      softmax_logits.append(np.exp(x - max(x))/np.sum(np.exp(x - max(x))))
  return np.asarray(softmax_logits)

class CalibrateClass:
  """ Class to calibrate probabilities, entropy and class uncertainty during inference on images """
  def __init__(self, logits, model_name, calib_method= "ts_all", uncert=None, y_true=None, general_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    """ Constructs all the necessary attributes for recalibration

    Args:
        logits (array): Predicted logits
        model_name (str): Model name for path
        calib_method (str, optional): Calibration method to select for further analysis. Defaults to "ts_all".
        uncert (array, optional): Predicted classification uncertainty. Defaults to None.
        y_true (array, optional): Ground truth classes. Defaults to None.
        general_path (str): Path to working space
    """
    
    self.model_name = model_name
    self.general_path = general_path
    self.calibrators = {}
    self.calib_methods = ["classification_ts_all", "classification_ts_percls", "classification_iso_all", "classification_iso_percls", 
               "unc_classification_ts_all", "unc_classification_ts_percls", "unc_classification_iso_all", "unc_classification_iso_percls"]
    self._import_calib_class()
    self.calib_method = calib_method
    self.logits = logits
    if uncert is not None: self.uncert = uncert[0] # Since it is contained in a list
    else: self.uncert = uncert
    
    if y_true is not None: print("Original Misclassification Error:", len(np.where(np.argmax(logits,axis=-1)!=y_true)[0])/len(y_true))
    self.y_true = y_true
    self.available_calib = ["ts_all", "ts_percls", "iso_all", "iso_percls"]

  def _import_calib_class(self):
      """ Import calibration models for classification calibration """
      for method in self.calib_methods:
        path = self.general_path + '/results/calibration/'+ self.model_name +'/classification/'+ method
        if os.path.exists(path):
          with open(path,'rb') as read_file:
              self.calibrators[method] = pickle.load(read_file)

  def _perform_class_calib(self, calib_method):
    """ Perform calibration on classification data based on user-selected method

    Args:
        calib_method (str): Calibration method selection

    Returns:
        Calibrated values
    """    
    samp_logits = self.logits
    if self.uncert is not None:
      calib_method = "unc_classification_"+ calib_method    
      n_y = tfp.distributions.Normal(loc=self.logits, scale=self.uncert)
      samples = n_y.sample(10)
      samp_logits = np.asarray(samples).reshape([-1,samples.shape[-1]])
      if self.y_true is not None: samp_y_true = np.stack([self.y_true]*10,axis=0).flatten()
      if self.y_true is not None: print("Post-sampling Misclassification Error:", len(np.where(np.argmax(samp_logits,axis=-1)!=samp_y_true)[0])/len(samp_y_true))
    else:
      calib_method = "classification_"+ calib_method

    if "ts" in calib_method: 
      calib_logits = samp_logits/self.calibrators[calib_method]
      new_probab = stable_softmax(calib_logits)
      
    elif "iso" in calib_method:
      calib_logits = stable_softmax(samp_logits)
      iso = self.calibrators[calib_method]
      if "all" in calib_method:
        calib_logits_post = iso.predict(calib_logits.flatten()).reshape(calib_logits.shape)
      else:
        calib_logits_post = [iso[i].predict(calib_logits[:,i]) for i in range(self.logits.shape[-1])]
        calib_logits_post = np.stack(calib_logits_post,axis=1)
        # necessary to recalibrate into sum 1
      new_probab = calib_logits_post/np.stack([np.sum(calib_logits_post, axis=-1)]*self.logits.shape[-1],axis=-1)
      if self.y_true is not None: print("Post-calibration Misclassification Error:", len(np.where(np.argmax(new_probab,axis=-1)!=samp_y_true)[0])/len(samp_y_true))
    else: 
      print("Unknown calibration method") 

    if self.uncert is not None:
      new_probab = new_probab.reshape([10,-1, self.logits.shape[-1]])
      new_uncert = np.std(new_probab,axis=0) # it needs to come before it
      new_probab = np.mean(new_probab,axis=0)
      if self.y_true is not None: print("Post-averaging Misclassification Error:", len(np.where(np.argmax(new_probab,axis=-1)!=self.y_true)[0])/len(self.y_true))
      new_entropy = -np.sum(new_probab*np.nan_to_num(np.log2(np.maximum(new_probab,10**-7))), axis=1)  
      return new_entropy, new_probab, new_uncert
    else:
      new_entropy = -np.sum(new_probab*np.nan_to_num(np.log2(np.maximum(new_probab,10**-7))), axis=1)  
      return new_entropy, new_probab
    
  def calibrate_class(self):
    """ Main function to perform calibration """

    if "classification_"+self.available_calib[0] in self.calibrators: calib_class_tsall = self._perform_class_calib(self.available_calib[0])
    else: calib_class_tsall = [np.array([]),np.array([]),np.array([])]
    
    if "classification_"+self.available_calib[1] in self.calibrators: calib_class_tspcl = self._perform_class_calib(self.available_calib[1]) 
    else: calib_class_tspcl = [np.array([]),np.array([]),np.array([])]
    
    if "classification_"+self.available_calib[2] in self.calibrators: calib_class_isoall = self._perform_class_calib(self.available_calib[2]) 
    else: calib_class_isoall = [np.array([]),np.array([]),np.array([])]
    
    if "classification_"+self.available_calib[3] in self.calibrators: calib_class_isopcl = self._perform_class_calib(self.available_calib[3])   
    else: calib_class_isopcl = [np.array([]),np.array([]),np.array([])] 
  
    if self.uncert is not None: select_uncert_mcclass = np.array([])
    select_entropy = np.array([])
    if self.calib_method == self.available_calib[0] and "classification"+self.available_calib[0] in self.calibrators:
      if self.uncert is not None: select_uncert_mcclass = calib_class_tsall[2]
      select_entropy = calib_class_tsall[0]
    elif self.calib_method == self.available_calib[1] and "classification"+self.available_calib[1] in self.calibrators:
      if self.uncert is not None: select_uncert_mcclass = calib_class_tspcl[2]
      select_entropy = calib_class_tspcl[0]
    elif self.calib_method == self.available_calib[2] and "classification"+self.available_calib[2] in self.calibrators:
      if self.uncert is not None: select_uncert_mcclass = calib_class_isoall[2]
      select_entropy = calib_class_isoall[0]
    elif self.calib_method == self.available_calib[3] and "classification"+self.available_calib[3] in self.calibrators:
      if self.uncert is not None: select_uncert_mcclass = calib_class_isopcl[2]
      select_entropy = calib_class_isopcl[0]

    if self.uncert is not None: 
      return select_uncert_mcclass, select_entropy, calib_class_tsall[1], calib_class_tsall[2], calib_class_tsall[0], calib_class_tspcl[1], calib_class_tspcl[2], calib_class_tspcl[0], calib_class_isoall[1], calib_class_isoall[2], calib_class_isoall[0], calib_class_isopcl[1], calib_class_isopcl[2], calib_class_isopcl[0]
    else:
      return select_entropy, calib_class_tsall[1], calib_class_tsall[0], calib_class_tspcl[1], calib_class_tspcl[0], calib_class_isoall[1], calib_class_isoall[0], calib_class_isopcl[1], calib_class_isopcl[0]

class Count_class_instances:
    """ Class to count the distribution of the classes in the dataset """
    def __init__(self, train_data, val_data, config):
      """ Constructs all the necessary attributes for counting class instances per dataset
      
      Args:
        train_data (tensor): Dataset containing the training data
        val_data (tensor): Dataset containing the validation data
        config (dict): Model config
      """

      # val_labels = list(map(lambda x: x[1], val_dataset.take(1)))
      # train_labels = list(map(lambda x: x[1], train_dataset.take(1)))
      # val_dataset =  val_dataset.take(1)
      # for (images, labels) in val_dataset:
      #     print(images.shape)
      # cv2.imwrite("val_im.jpg", np.asarray(images[0]).astype(np.uint8))

      self.train_data = train_data
      self.val_data = val_data
      self.config = config
      label_map = label_util.get_label_map(config.label_map)
      class_names = []
      for i in range(1, 1+config.num_classes):
          class_names.append(label_map[i])    
      self.class_names = class_names
      self.collect_perclass = [0]*config.num_classes
      self.count_train_labels = 0
      self.count_val_labels = 0
      self.count_mean_train = 0
      self.count_mean_val = 0

    def _collect_instances(self, labels): 
      """ Collects class instances """

      # logging.info("# GT labels:" + str(len(labels)))
      # logging.info("Batch shape:" + str(labels[0].shape))
      for i in range(len(labels)):      
          temp_labels = labels[i][:,-1] # [y1, x1, y2, x2, is_crowd, area, class]
          occurances = temp_labels[temp_labels>=0] 
          self.collect_perclass = [self.collect_perclass[i]+len(occurances[occurances==i+1]) for i in range(self.config.num_classes)]

    def _plot_class_dist(self):       
        # Existing code for saving the plot
        save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "misc", f"classes_distribution_{self.config.label_map}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure()
        x_axis = np.arange(len(self.class_names))
        train_bar = plt.bar(x_axis -0.2, self.count_train_labels, color = 'b', width=0.4, label = 'Train')
        val_bar = plt.bar(x_axis +0.2, self.count_val_labels, color = 'g', width=0.4, label = 'Val')
        plt.xticks(x_axis, self.class_names)
        plt.ylabel("Count")
        plt.legend(loc='upper right')

        i = 0
        for p in train_bar:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            plt.text(x+width/2,
                    y+height*1.01,
                    str(self.count_mean_train[i])+'%',
                    ha='center',
                    weight='bold')
            i+=1

        i = 0
        for p in val_bar:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            plt.text(x+width/2,
                    y+height*1.01,
                    str(self.count_mean_val[i])+'%',
                    ha='center',
                    weight='bold')
            i+=1
        
        plt.savefig(save_path)
        plt.close()

    def count_dataset(self, num_batches, plot_dist=False):
      """ Count the number of instances for each class

      Args:
        num_batches (int): Number of batches per epoch
        plot_dist (bool): Draw histogram of the distribution
      """

      strategy = tf.distribute.get_strategy()      
      # Count for validation dataset
      val_count = self.config.eval_samples //  self.config.batch_size      
      if val_count !=  self.config.eval_samples / self.config.batch_size:
          logging.info("Class instances: Number of validation samples not directly dividable by batch size, slight variation might occur. Difference is " + str(abs(val_count -  self.config.eval_samples / self.config.batch_size)*self.config.batch_size) + " images.")
      val_dataset =  self.val_data.take(val_count)
      val_dataset = strategy.experimental_distribute_dataset(val_dataset)
      for (_, labels) in val_dataset: 
          strategy.run(self._collect_instances, [labels['groundtruth_data']])
      
      # Save values for validation
      self.count_val_labels = self.collect_perclass.copy()
      # Reset
      self.collect_perclass = [0]*self.config.num_classes

      # Count for training dataset
      train_count =  self.config.steps_per_epoch
      if train_count != num_batches/self.config.batch_size:
          logging.info("Class instances: Number of training samples not directly dividable by batch size, slight variation might occur. Difference is " + str(abs(train_count- num_batches/self.config.batch_size)*self.config.batch_size) + " images.")

      train_dataset =  self.train_data.take(train_count)
      train_dataset = strategy.experimental_distribute_dataset(train_dataset)
      for (_, labels) in train_dataset: # batches are distributed on number of gpus
          strategy.run(self._collect_instances, [labels['groundtruth_data']])
      # Save values for training
      self.count_train_labels = self.collect_perclass.copy()
      # Reset
      self.collect_perclass = [0]*self.config.num_classes
      
      self.count_mean_train = np.round(np.asarray(self.count_train_labels)/sum(self.count_train_labels)*100, 2)
      self.count_mean_val = np.round(np.asarray(self.count_val_labels)/sum(self.count_val_labels)*100, 2)
      for i in range(self.config.num_classes):
          logging.info('Amount of objects of class {} in training data is {}, ({}%)'.format(self.class_names[i], self.count_train_labels[i], self.count_mean_train[i]))
          logging.info('Amount of objects of class {} in validation data is {}, ({}%)'.format(self.class_names[i], self.count_val_labels[i], self.count_mean_val[i]))

      if plot_dist: self._plot_class_dist()
