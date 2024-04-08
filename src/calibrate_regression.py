# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Calibrate localization uncertainty """


import os

import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy import stats
import tensorflow as tf
from tensorflow.python.training import gradient_descent
from sklearn.isotonic import IsotonicRegression

from utils_box import calc_nll, calc_ece, calc_rmse


class RegressionCalib:
    """ Class to perform regression calibration """
    def __init__(self, gt_boxes, pred_boxes, box_uncert, gt_classes, model_name, general_path):
        """ Constructs the necessary attributes for the regression calibration

        Args:
            gt_boxes (array): Ground truth coodinates
            pred_boxes (array): Predicted bounding boxes
            box_uncert (array): Predicted uncertainty per coordinate
            gt_classes (array): Ground truth classes for per-class calibration
            model_name (str): Model name in order to save plots and files under it
            general_path (str): Path to working space

        """
        self.saving_path = general_path
        self.gt_boxes = gt_boxes
        self.pred_boxes = pred_boxes
        self.box_uncert = box_uncert
        self.model_name = model_name
        self.gt_classes = gt_classes
        
    @staticmethod
    def _relative_calc_ece(norm_residuals, norm_box_uncert):
        """ Calculate expected calibration error based on normalized sigma with box width and height """
        n_intervals = 100
        p_m = np.linspace(0,1,n_intervals)
        emp_conf = [0]*n_intervals
        for i in range(n_intervals): 
            interval_fit = np.less_equal(norm_residuals, np.abs(norm_box_uncert*stats.norm.ppf((1-p_m[i])/2))) 
            emp_conf[i] = np.mean(interval_fit,axis=0)
        if len(norm_residuals.shape) == 1: #for one coordinate
            ece = np.mean(np.abs(emp_conf - p_m))
        else:
            ece = np.mean(np.abs(emp_conf - np.swapaxes([p_m]*4,0,1)))
        return ece
    
    @staticmethod
    def _plot_boxcalib(gt_boxes, pred_boxes, box_uncert, label):
        """ Plot calibration curve """
        n_intervals = 100
        p_m = np.linspace(0,1,n_intervals)
        emp_conf = [0]*n_intervals

        for i in range(n_intervals): 
            interval_fit = np.less_equal(np.abs(pred_boxes-gt_boxes), np.abs(box_uncert*stats.norm.ppf((1-p_m[i])/2))) 
            emp_conf[i] = np.mean(interval_fit)
        plt.plot(p_m,emp_conf,"s-", label=label)
        plt.legend()
        # ece = np.mean(np.abs(emp_conf- p_m))   

    def _plot_stdvsdiff(self, residuals, box_uncert, iso_reg, split=0, label="ymin"):
        """ Plot the residuals vs the uncertainty with the isotonic regression calibration curve

        Args:
            residuals (array): Absolute difference between predicted and ground truth bounding boxes
            box_uncert (array): Predicted uncertainty per coordinate
            split (float): Split value to plot subset of the results
            label (str): Label to save the plot under
        """
        res_dict = {'x_range': (float(iso_reg.X_min_), float(iso_reg.X_max_)),

        'x_thresholds': [float(val) for val in iso_reg.X_thresholds_],

        'y_thresholds': [float(val) for val in iso_reg.y_thresholds_]}

        plt.figure()
        plt.scatter(box_uncert[split:].flatten(), residuals[split:].flatten(), alpha=0.3)
        plt.plot(res_dict['x_thresholds'], res_dict['y_thresholds'], '-o', color="tab:red", label='Iso Curve')
        plt.xlabel("Predicted STD")
        plt.ylabel("Residuals")
        plt.legend()
        plt.grid("minor")
        plt.tight_layout() 
        plt.legend(loc = "upper center")
        plt.savefig(self.saving_path+self.model_name+'/regres_plots/regression_calibration_'+label+'_std.png',bbox_inches='tight')    
        plt.close()
     
    def _plot_intervals(self, gt_boxes, pred_boxes, box_uncert, calib_box_uncert, split, label, calib=False):
        """ Plot the ground truth and the predicted values +- their uncertainty intervals
        
        Args:
            gt_boxes (array): Ground truth coordinates
            pred_boxes (array): Predicted coordinates
            box_uncert (array): Predicted uncertainty per coordinate
            calib_box_uncert (array): Calibrated predicted uncertainty
            split (float): Train/val split
            label (str): Label to save the plot with
            calib (bool): Selects if calibrated uncertainty should be used
        """


        if (len(gt_boxes)-split) < 100:
            n_samples = len(gt_boxes)-split
        else: n_samples = 100

        x = np.arange(n_samples)
        cut_gt_boxes = gt_boxes[split:split+n_samples]
        cut_pred_boxes = pred_boxes[split:split+n_samples]
        if calib:
            select_uncert = calib_box_uncert
            cut_select_uncert = calib_box_uncert[:n_samples] # its already split
        else:
            select_uncert = box_uncert[split:]
            cut_select_uncert = box_uncert[split:split+n_samples]

        upper_int = cut_pred_boxes+cut_select_uncert
        lower_int = cut_pred_boxes-cut_select_uncert

        correct_all = 0
        for i in range(len(calib_box_uncert)):
            if pred_boxes[split:][i]-select_uncert[i]<=gt_boxes[split:][i]<=pred_boxes[split:][i]+select_uncert[i]: correct_all+=1
        correct_all /= len(calib_box_uncert)
        correct_all *= 100
        
        plt.figure()
        plt.scatter(x,cut_gt_boxes, label="Ground truth")
        plt.fill_between(x, lower_int, upper_int, color='grey', alpha=0.5)
        plt.xlabel('# Bounding Box')
        plt.ylabel(label + '_coordinate')
        plt.title("% of true values in interval \u00B1 \u03C3 of predicted values \n out of total detections: "+str(round(correct_all,1))+ "%")
        if calib:
            plt.plot(x,cut_pred_boxes+cut_select_uncert, label="Pred. upper limit, calib.", color='black', alpha=0.5)
            plt.plot(x,cut_pred_boxes-cut_select_uncert, label="Pred. lower limit, calib.", color='red', alpha=0.5)
            plt.legend()
            plt.savefig(self.saving_path+self.model_name+'/regres_plots/regression_calibration_'+label+'_coord_calib.png')
        else:
            plt.plot(x,cut_pred_boxes+cut_select_uncert, label="Pred. upper limit", color='black', alpha=0.5)
            plt.plot(x,cut_pred_boxes-cut_select_uncert, label="Pred. lower limit", color='red', alpha=0.5)
            plt.legend()
            plt.savefig(self.saving_path+self.model_name+'/regres_plots/regression_calibration_'+label+'_coord.png')        
        plt.close()        

    def _conf_all(self, gt_boxes, pred_boxes, box_uncert, calib_box_uncert, split=0, calib=False, method="TS All"):
        """ Calculate the percentage of true values in +- sigma, ece and nll, and writes the results to a text file.
        
        Args:
            gt_boxes (array): Ground truth coordinates
            pred_boxes (array): Predicted coordinates
            box_uncert (array): Predicted uncertainty per coordinate
            calib_box_uncert (array): Calibrated predicted uncertainty
            split (float): Train/val split
            calib (bool): Selects if calibrated uncertainty should be used
            method (str): String of the used method to save along the results in order to recognize it
        """

        if calib:
            select_uncert = calib_box_uncert # Already on val set
        else:
            select_uncert = box_uncert[split:]
        
        correct_all = 0
        for i in range(len(calib_box_uncert)):
            if (pred_boxes[split:][i]-select_uncert[i]<=gt_boxes[split:][i]).all() and (gt_boxes[split:][i]<=pred_boxes[split:][i]+select_uncert[i]).all(): correct_all+=1
        correct_all /= len(calib_box_uncert)
        correct_all *= 100

        residuals = np.abs(gt_boxes[split:]-pred_boxes[split:])
        if calib:
            log_percent = "% of true values in interval \u00B1 \u03C3 of predicted values for all coordinates after calibration with "+method+": " + str(round(correct_all,1))
            ece = np.round(calc_ece(gt_boxes[split:].flatten(), pred_boxes[split:].flatten(), calib_box_uncert.flatten()),4)
            log_ece = "Calibrated ECE " + str(ece)      
            nll = np.round(calc_nll(residuals.flatten(),calib_box_uncert.flatten()),4)
            log_nll = "Calibrated NLL " + str(nll)      
            rmsu = np.round(calc_rmse(residuals.flatten().flatten(), calib_box_uncert.flatten()),4)
            log_rmsu = "Calibrated RMSUE " + str(rmsu)  
            sharp = np.round(np.sqrt(np.mean(np.square(calib_box_uncert.astype(np.float32).flatten()))),4)
            log_sharp = "Calibrated Sharp. " + str(sharp)


        else:
            log_percent = "% of true values in interval \u00B1 \u03C3 of predicted values for all coordinates: " + str(round(correct_all,1))
            ece = np.round(calc_ece(gt_boxes[split:].flatten(), pred_boxes[split:].flatten(), box_uncert[split:].flatten()),4)
            log_ece = "Uncalibrated ECE " + str(ece) 
            nll = np.round(calc_nll(np.abs(gt_boxes[split:]-pred_boxes[split:]).flatten(),box_uncert[split:].flatten()),4)
            log_nll = "Uncalibrated NLL " + str(nll)           
            rmsu = np.round(calc_rmse(residuals.flatten().flatten(), box_uncert[split:].flatten()),4)
            log_rmsu = "Uncalibrated RMSUE " + str(rmsu)  
            sharp = np.round(np.sqrt(np.mean(np.square(box_uncert[split:].astype(np.float32).flatten()))),4)
            log_sharp = "Uncalibrated Sharp. " + str(sharp)

        with open(self.saving_path+self.model_name+"/regression_logging.txt", "a") as f: 
            f.write(log_percent + " \n" +  log_ece + ", " + log_nll + ", " + log_rmsu + ", " + log_sharp + " \n")
      
    def _iso_regression(self, gt_boxes, pred_boxes, box_uncert, split, label):
        """ Perform isotonic regression on the predicted uncertainty
        
        Args:
            gt_boxes (array): Ground truth coordinates
            pred_boxes (array): Predicted coordinates
            box_uncert (array): Predicted uncertainty per coordinate
            split (float): Train/val split
            label (str): Label to save the plot with
        Returns:
            Isotonic regression model
        """
        with open(self.saving_path+self.model_name+"/regression_logging.txt", "a") as f: f.write("Isotonic regression: " + label + " \n")     

        residuals = np.abs(pred_boxes - gt_boxes)
        if len(gt_boxes.shape)==1:
            iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(box_uncert[:split], residuals[:split])
            calib_box_uncert= iso_reg.predict(box_uncert[split:])

            plt.figure()
            plt.plot([0, 1], [0, 1], "k:", label="Ideal")
            self._plot_boxcalib(gt_boxes[split:],pred_boxes[split:],box_uncert[split:], "Original")
            self._plot_boxcalib(gt_boxes[split:],pred_boxes[split:],calib_box_uncert, "Calibrated")            
            plt.xlabel('Measured Confidence Levels')
            plt.ylabel('Expected Confidence Levels')
            plt.title(label)
            plt.savefig(self.saving_path+self.model_name+'/regres_plots/regression_calibration_'+label+'.png')              
            plt.close()

            self._plot_intervals(gt_boxes, pred_boxes, box_uncert, calib_box_uncert, split, label)
            self._plot_intervals(gt_boxes, pred_boxes, box_uncert, calib_box_uncert, split, label, True)
            self._conf_all(gt_boxes, pred_boxes, box_uncert, calib_box_uncert, split)
            self._conf_all(gt_boxes, pred_boxes, box_uncert, calib_box_uncert, split, True, method="IS-"+label)

            iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(box_uncert, residuals) # On all data
            self._plot_stdvsdiff(residuals, box_uncert, iso_reg, 0, label)
        else:
            iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(box_uncert[:split].flatten(), residuals[:split].flatten())
            calib_box_uncert= iso_reg.predict(box_uncert[split:].flatten()).reshape([-1,4])

            self._conf_all(gt_boxes, pred_boxes, box_uncert, calib_box_uncert, split)
            self._conf_all(gt_boxes, pred_boxes, box_uncert, calib_box_uncert, split, True, method="IS ALL")
            iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(box_uncert.flatten(), residuals.flatten())

        # return fitted on all data
        return iso_reg

    def _temp_scaling(self, gt_boxes, pred_boxes, box_uncert, split, label):
        """ Apply temperature scaling on the predicted uncertainty 

        Args:
            gt_boxes (array): Ground truth coordinates
            pred_boxes (array): Predicted coordinates
            box_uncert (array): Predicted uncertainty per coordinate
            split (float): Train/val split
            label (str): Label to save the plot with

        Returns:
            Optimized temperature
        """
        
        with open(self.saving_path+self.model_name+"/regression_logging.txt", "a") as f: f.write("Temperature scaling: " + label + " \n")     

        x = tf.Variable(1.0, trainable=True, dtype=tf.float32)
        @tf.function
        def f_x():
            """ Loss function to be reduced with gradient descent """
            # MAE
            # return tf.reduce_mean(tf.math.abs(tf.cast(tf.math.abs(gt_boxes[:split]-pred_boxes[:split]), dtype=tf.float32) - tf.math.divide_no_nan(tf.cast(box_uncert[:split], dtype=tf.float32) , tf.abs(x)))) 

            # MSE
            # return tf.reduce_mean(tf.math.square(tf.cast(tf.math.abs(gt_boxes[:split]-pred_boxes[:split]), dtype=tf.float32) - tf.math.divide_no_nan(tf.cast(box_uncert[:split], dtype=tf.float32) , tf.abs(x)))) 

            # RMSE
            return tf.math.sqrt(tf.reduce_mean(tf.math.square(tf.cast(tf.math.abs(gt_boxes[:split]-pred_boxes[:split]), dtype=tf.float32) - tf.math.divide_no_nan(tf.cast(box_uncert[:split], dtype=tf.float32) , tf.abs(x)))))

        for _ in range(100):
            gradient_descent.GradientDescentOptimizer(0.1).minimize(f_x)
            # print([x.numpy(), f_x().numpy()])
        temp = np.abs(x.numpy())
        with open(self.saving_path+self.model_name+"/regression_logging.txt", "a") as f: f.write("Temperature: "+ str(temp) + " \n")   
        
        if len(gt_boxes.shape) == 1:
            self._plot_intervals(gt_boxes, pred_boxes, box_uncert, box_uncert[split:] / temp, split, label + "_temp", True)
            self._conf_all(gt_boxes, pred_boxes, box_uncert, box_uncert[split:] / temp, split, True, method="TS-"+label)
        else:
            self._conf_all(gt_boxes, pred_boxes, box_uncert, box_uncert[split:] / temp, split, True)

        # Fit on all data        
        y = tf.Variable(1.0, trainable=True, dtype=tf.float32)
        @tf.function
        def f_y():
            """defines the loss function to be reduced with gradient descent
            """
            # MAE
            return tf.reduce_mean(tf.math.abs(tf.cast(tf.math.abs(gt_boxes-pred_boxes), dtype=tf.float32) - tf.math.divide_no_nan(tf.cast(box_uncert, dtype=tf.float32) , tf.abs(y)))) 

            # MSE
            # return tf.reduce_mean(tf.math.square(tf.cast(tf.math.abs(gt_boxes-pred_boxes), dtype=tf.float32) - tf.math.divide_no_nan(tf.cast(box_uncert, dtype=tf.float32) , tf.abs(y)))) 

            # RMSE
            # return tf.math.sqrt(tf.reduce_mean(tf.math.square(tf.cast(tf.math.abs(gt_boxes-pred_boxes), dtype=tf.float32) - tf.math.divide_no_nan(tf.cast(box_uncert, dtype=tf.float32) , tf.abs(y)))))

        for _ in range(100):
            gradient_descent.GradientDescentOptimizer(0.1).minimize(f_y)
        temp = np.abs(y.numpy())
        return temp

    def _rel_temp_scaling(self, norm_box_uncert, norm_residuals, norm):
        """ Apply temperature scaling on the normalized predicted uncertainty
        
        Args:
            norm_box_uncert (array): Predicted uncertainty normalized by the width and height
            norm_residuals (array): Absolute difference normalized by the width and height
            norm (array): Normalizing array containing width and height

        Returns:
            Optimized temperature
        """
        x = tf.Variable(1.0, trainable=True, dtype=tf.float16)
        @tf.function
        def f_x():
            """ Loss function to be reduced with gradient descent """
            
            # MAE
            return tf.reduce_mean(tf.math.abs(norm_residuals - tf.math.divide_no_nan(norm_box_uncert , tf.abs(x)))) 

            # MSE
            # return tf.reduce_mean(tf.math.square(norm_residuals - tf.math.divide_no_nan(norm_box_uncert , tf.abs(x)))) 

            # RMSE
            # return tf.math.sqrt(tf.reduce_mean(tf.math.square(norm_residuals - tf.math.divide_no_nan(norm_box_uncert , tf.abs(x)))))


        for _ in range(100):
            gradient_descent.GradientDescentOptimizer(0.1).minimize(f_x)
            # print([x.numpy(), f_x().numpy()])
        temp = np.abs(x.numpy())

        return temp

    def _absolute_calibration(self, split):
        """  Perform isotonic regression and temperature scaling on the predicted uncertainty

        Args:
            split (float): Train/val split
        """
        with open(self.saving_path+self.model_name+"/regression_logging.txt", "a") as f: f.write("Absolute calibration \n")     
        ymin_calib = self._iso_regression(self.gt_boxes[:,0],self.pred_boxes[:,0],self.box_uncert[:,0], split, label="ymin")
        xmin_calib = self._iso_regression(self.gt_boxes[:,1],self.pred_boxes[:,1],self.box_uncert[:,1], split, label="xmin")
        ymax_calib = self._iso_regression(self.gt_boxes[:,2],self.pred_boxes[:,2],self.box_uncert[:,2], split, label="ymax")
        xmax_calib = self._iso_regression(self.gt_boxes[:,3],self.pred_boxes[:,3],self.box_uncert[:,3], split, label="xmax")
        
        with open(self.saving_path+self.model_name+"/regres_models/regression_calib_iso_pcoo", "wb") as fp:   
            pickle.dump(ymin_calib,fp)
            pickle.dump(xmin_calib,fp)
            pickle.dump(ymax_calib,fp)
            pickle.dump(xmax_calib,fp)

        temp_ymin = self._temp_scaling(self.gt_boxes[:,0],self.pred_boxes[:,0],self.box_uncert[:,0], split, label="ymin")
        temp_xmin = self._temp_scaling(self.gt_boxes[:,1],self.pred_boxes[:,1],self.box_uncert[:,1], split, label="xmin")
        temp_ymax = self._temp_scaling(self.gt_boxes[:,2],self.pred_boxes[:,2],self.box_uncert[:,2], split, label="ymax")
        temp_xmax = self._temp_scaling(self.gt_boxes[:,3],self.pred_boxes[:,3],self.box_uncert[:,3], split, label="xmax")
        
        with open(self.saving_path+self.model_name+"/regres_models/regression_calib_ts_pcoo", "wb") as fp:
                pickle.dump(temp_ymin,fp)
                pickle.dump(temp_xmin,fp)
                pickle.dump(temp_ymax,fp)
                pickle.dump(temp_xmax,fp)

        iso_all = self._iso_regression(self.gt_boxes,self.pred_boxes,self.box_uncert, split, label="all")
        temp_all = self._temp_scaling(self.gt_boxes,self.pred_boxes,self.box_uncert, split, label="all")

        with open(self.saving_path+self.model_name+"/regres_models/regression_calib_ts_all", "wb") as fp: pickle.dump(temp_all,fp)
        with open(self.saving_path+self.model_name+"/regres_models/regression_calib_iso_all", "wb") as fp: pickle.dump(iso_all,fp)

        with open(self.saving_path+self.model_name+"/regression_logging.txt", "a") as f: f.write("################################################## End absolute calibration ##################################################" + " \n\n")   
    
    def _relative_calibration(self, split):     
        """  Perform isotonic regression and temperature scaling on the predicted normalized uncertainty

        Args:
            split (float): Train/val split
        """

        with open(self.saving_path+self.model_name+"/regression_logging.txt", "a") as f: f.write("Relative calibration \n")   
        width = np.asarray(self.pred_boxes[:,3]-self.pred_boxes[:,1])
        height = np.asarray(self.pred_boxes[:,2]-self.pred_boxes[:,0])
        norm = np.swapaxes([height,width,height,width], 0, 1)

        norm_residuals = np.divide(np.abs(self.gt_boxes-self.pred_boxes), norm, out=np.zeros_like(np.abs(self.gt_boxes-self.pred_boxes)), where=norm!=0, dtype=np.float16)    
        norm_box_uncert = np.divide(self.box_uncert, norm, out=np.zeros_like(self.box_uncert), where=norm!=0, dtype=np.float16)
        
        # Relative isotonic regression
        with open(self.saving_path+self.model_name+"/regression_logging.txt", "a") as f: f.write("Relative isotonic regression: all \n")    
        norm_iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(norm_box_uncert[:split].flatten(), norm_residuals[:split].flatten())
        norm_calib_box_uncert = norm_iso_reg.predict(norm_box_uncert[split:].flatten()).reshape([-1,4])
        self._conf_all(self.gt_boxes[split:], self.pred_boxes[split:], norm_box_uncert[split:]*norm[split:], norm_calib_box_uncert*norm[split:], 0)
        self._conf_all(self.gt_boxes[split:], self.pred_boxes[split:], norm_box_uncert[split:]*norm[split:], norm_calib_box_uncert*norm[split:], 0, True, method = "relative IS ALL")

        norm_iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(norm_box_uncert.flatten(), norm_residuals.flatten())
        with open(self.saving_path+ self.model_name +"/regres_models/regression_calib_iso_all_relative", "wb") as fp: pickle.dump(norm_iso_reg,fp)

        labels = ["ymin","xmin","ymax","xmax"]
        with open(self.saving_path+ self.model_name +"/regres_models/regression_calib_iso_pcoo_relative", "wb") as fp:
            for i in range(4):
                with open(self.saving_path+self.model_name+"/regression_logging.txt", "a") as f: f.write("Relative isotonic regression: " + labels[i] + " \n")    
                norm_iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(norm_box_uncert[:split][:,i], norm_residuals[:split][:,i])
                norm_calib_box_uncert = norm_iso_reg.predict(norm_box_uncert[split:][:,i])        
                self._conf_all(self.gt_boxes[split:][:,i], self.pred_boxes[split:][:,i], norm_box_uncert[split:][:,i]*norm[split:][:,i], norm_calib_box_uncert*norm[split:][:,i], 0)
                self._conf_all(self.gt_boxes[split:][:,i], self.pred_boxes[split:][:,i], norm_box_uncert[split:][:,i]*norm[split:][:,i], norm_calib_box_uncert*norm[split:][:,i], 0, True, method="relative IS-"+ labels[i])
                norm_iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(self.box_uncert[:,i], norm_residuals[:,i])
                pickle.dump(norm_iso_reg, fp)

        # Relative temperature scaling
        with open(self.saving_path+self.model_name+"/regression_logging.txt", "a") as f: f.write("Relative temperature scaling: all \n")     
        norm_temp = self._rel_temp_scaling(norm_box_uncert[:split], norm_residuals[:split], norm[:split])
        with open(self.saving_path+self.model_name+"/regression_logging.txt", "a") as f: f.write("Temperature: "+ str(norm_temp) + " \n") 
        self._conf_all(self.gt_boxes[split:], self.pred_boxes[split:], norm_box_uncert[split:]*norm[split:], norm_box_uncert[split:]/norm_temp*norm[split:], 0, True, method="relative TS All")      

        norm_temp = self._rel_temp_scaling(norm_box_uncert, norm_residuals, norm)
        with open(self.saving_path+ self.model_name +"/regres_models/regression_calib_ts_all_relative", "wb") as fp: pickle.dump(norm_temp, fp)       
    
        with open(self.saving_path+ self.model_name +"/regres_models/regression_calib_ts_pcoo_relative", "wb") as fp:
            for i in range(4):
                with open(self.saving_path+self.model_name+"/regression_logging.txt", "a") as f: f.write("Relative temperature scaling: " + labels[i] + " \n")     
                norm_temp = self._rel_temp_scaling(norm_box_uncert[:split][:,i], norm_residuals[:split][:,i], norm[:split][:,i])
                with open(self.saving_path+self.model_name+"/regression_logging.txt", "a") as f: f.write("Temperature: "+ str(norm_temp) + " \n") 
                self._conf_all(self.gt_boxes[split:][:,i], self.pred_boxes[split:][:,i], norm_box_uncert[split:][:,i]*norm[split:][:,i], norm_box_uncert[split:][:,i]/norm_temp*norm[split:][:,i], 0, True, method="Relative TS-"+ labels[i])
                norm_temp = self._rel_temp_scaling(norm_box_uncert[:,i], norm_residuals[:,i], norm[:,i])
                pickle.dump(norm_temp,fp)

        with open(self.saving_path+self.model_name+"/regression_logging.txt", "a") as f: f.write("################################################## End relative calibration ##################################################" + " \n\n")   

    def _per_class_calibration(self, split):
        """  Perform per-class isotonic regression and temperature scaling on the predicted uncertainty and normalized uncertainty

        Args:
            split (float): Train/val split
        """
        with open(self.saving_path+self.model_name+"/regression_logging.txt", "a") as f: f.write("Per-class calibration \n")   

        # Calibrate per-class per-coordinate
        if np.min(self.gt_classes) == 0: num_classes = int(np.max(self.gt_classes)) +1
        else: 
            num_classes = int(np.max(self.gt_classes))
            self.gt_classes-=1
        calibrators = []
        for i in range(num_classes):
            for j in range(4):
                residuals = np.abs(self.gt_boxes[:,j] - self.pred_boxes[:,j])[:split][self.gt_classes[:split]==i]
                calibrators.append(IsotonicRegression(increasing=True, out_of_bounds='clip').fit(self.box_uncert[:,j][:split][self.gt_classes[:split]==i].flatten(), residuals.flatten()))
        
        calibrators = np.asarray(calibrators).reshape(num_classes,4)
        calib_box_uncert = np.zeros_like(self.box_uncert[split:])
        lbls = ["ymin","xmin","ymax","xmax"]
        for i in range(num_classes):
            for j in range(4):
                calib_box_uncert[:,j][self.gt_classes[split:]==i]=(calibrators[i,j].predict(self.box_uncert[:,j][split:][self.gt_classes[split:]==i]))
                self._conf_all(self.gt_boxes[:,j][split:][self.gt_classes[split:]==i], self.pred_boxes[:,j][split:][self.gt_classes[split:]==i], self.box_uncert[:,j][split:][self.gt_classes[split:]==i], calib_box_uncert[:,j][self.gt_classes[split:]==i], 0)
                self._conf_all(self.gt_boxes[:,j][split:][self.gt_classes[split:]==i], self.pred_boxes[:,j][split:][self.gt_classes[split:]==i], self.box_uncert[:,j][split:][self.gt_classes[split:]==i], calib_box_uncert[:,j][self.gt_classes[split:]==i], 0, True, method="per class IS-"+lbls[j]+" for class "+str(i))

        # Fit on all the data
        calibrators = []
        for i in range(num_classes):
            for j in range(4):
                residuals = np.abs(self.gt_boxes[:,j] - self.pred_boxes[:,j])[self.gt_classes==i]
                calibrators.append(IsotonicRegression(increasing=True, out_of_bounds='clip').fit(self.box_uncert[:,j][self.gt_classes==i].flatten(), residuals.flatten()))
        
        with open(self.saving_path+ self.model_name +"/regres_models/regression_calib_iso_perclscoo", "wb") as fp:   pickle.dump(calibrators, fp)
        

        # Relative per-class per-coordinate calibration
        width = np.asarray(self.pred_boxes[:,3]-self.pred_boxes[:,1])
        height = np.asarray(self.pred_boxes[:,2]-self.pred_boxes[:,0])
        norm = np.swapaxes([height,width,height,width], 0, 1)

        norm_residuals = np.divide(np.abs(self.gt_boxes-self.pred_boxes), norm, out=np.zeros_like(np.abs(self.gt_boxes-self.pred_boxes)), where=norm!=0, dtype=np.float16)    
        norm_box_uncert = np.divide(self.box_uncert, norm, out=np.zeros_like(self.box_uncert), where=norm!=0, dtype=np.float16)
        calibrators = []
        for i in range(num_classes):
            for j in range(4):
                calibrators.append(IsotonicRegression(increasing=True, out_of_bounds='clip').fit(norm_box_uncert[:,j][:split][self.gt_classes[:split]==i].flatten(), norm_residuals[:,j][:split][self.gt_classes[:split]==i].flatten()))
        
        calibrators = np.asarray(calibrators).reshape(num_classes,4)
        calib_norm_box_uncert = np.zeros_like(norm_box_uncert[split:])
        lbls = ["ymin","xmin","ymax","xmax"]
        for i in range(num_classes):
            for j in range(4):
                calib_norm_box_uncert[:,j][self.gt_classes[split:]==i]=(calibrators[i,j].predict(norm_box_uncert[:,j][split:][self.gt_classes[split:]==i]))
                self._conf_all(self.gt_boxes[:,j][split:][self.gt_classes[split:]==i], self.pred_boxes[:,j][split:][self.gt_classes[split:]==i], norm_box_uncert[:,j][split:][self.gt_classes[split:]==i]*norm[:,j][split:][self.gt_classes[split:]==i], calib_norm_box_uncert[:,j][self.gt_classes[split:]==i]*norm[:,j][split:][self.gt_classes[split:]==i], 0)
                self._conf_all(self.gt_boxes[:,j][split:][self.gt_classes[split:]==i], self.pred_boxes[:,j][split:][self.gt_classes[split:]==i], norm_box_uncert[:,j][split:][self.gt_classes[split:]==i]*norm[:,j][split:][self.gt_classes[split:]==i], calib_norm_box_uncert[:,j][self.gt_classes[split:]==i]*norm[:,j][split:][self.gt_classes[split:]==i], 0, True, method="per class relative IS-"+lbls[j]+" for class "+str(i))

        # Fit on all the data
        calibrators = []
        for i in range(num_classes):
            for j in range(4):
                calibrators.append(IsotonicRegression(increasing=True, out_of_bounds='clip').fit(norm_box_uncert[:,j][self.gt_classes==i].flatten(), norm_residuals[:,j][self.gt_classes==i].flatten()))

        with open(self.saving_path+ self.model_name +"/regres_models/regression_calib_iso_perclscoo_relative", "wb") as fp:   pickle.dump(calibrators, fp)

        with open(self.saving_path+self.model_name+"/regression_logging.txt", "a") as f: f.write("################################################## End per-class calibration ##################################################" + " \n\n")  

    def box_uncert_calibration(self):  
        """ Main function to calibrate predicted box uncertainty and plot calibration results """  

        with open(self.saving_path+self.model_name+"/regression_logging.txt", "a") as f: f.write("Measuring box uncertainty calibration" + " \n")   

        if not os.path.exists(self.saving_path+ self.model_name +"/regres_models"): os.makedirs(self.saving_path+ self.model_name +"/regres_models")  
        if not os.path.exists(self.saving_path+ self.model_name +"/regres_plots"): os.makedirs(self.saving_path+ self.model_name +"/regres_plots")  

        split_factor = 0.8
        split = int(len(self.gt_boxes)*split_factor)
        self._absolute_calibration(split)
        self._relative_calibration(split)
        self._per_class_calibration(split)
        print("Plotted regression calibration on 20% of the validation dataset, the rest is used for fiting")