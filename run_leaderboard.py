#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow
import keras_leaderboard 

import datetime as dt
import os
import warnings

np.random.seed(21)
tensorflow.set_random_seed(21)

class make_leaderboard(object):
    def __init__(self, 
                 X_train, y_train, X_val, y_val, 
                 data_name, 
                 augs, 
                 width, 
                 height, 
                 channels, 
                 optimizer, 
                 loss, 
                 metrics, 
                 classes, 
                 n_classes, #same as classes
                 batch_size, 
                 model_network_name, 
                 epochs, 
                 patience, 
                 logfile,
                 save_folder=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.data_name = data_name
        self.augs = augs
        self.width = width
        self.height = height
        self.channels = channels
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.classes = classes
        self.n_classes = n_classes #the same value
        self.batch_size = batch_size
        self.model_network_name = model_network_name
        self.epochs = epochs
        self.patience = patience
        self.logfile = logfile
        
        self.dir, self.plot_dir, self.csv_dir= self.make_dirs(save_folder)

        
    def register_the_data(self): 
        data_object = keras_leaderboard.Register_Data(self.X_train, self.y_train, self.X_val, self.y_val, self.data_name, self.augs)
        xtr_shape, ytr_shape, xval_shape, yval_shape = data_object.register_shapes()
        data_augs, train_datagen = data_object.register_augs()
        
        return data_object, xtr_shape, ytr_shape, xval_shape, yval_shape, data_augs, train_datagen
    
    
    '''
    model_network_name: str        
    default model_type is cnn; change to 'vgg' to call vgg
    build as many CNN's as you like                   
    '''
    
    def build_and_compile_CNN(self, model_type='cnn'):
        model_object = keras_leaderboard.LB_Model(self.width, self.height, self.channels, self.optimizer, self.loss, self.metrics, self.classes, self.batch_size, self.model_network_name)
        if model_type == 'cnn':
            network = model_object.basic_cnn()
        elif model_type == 'vgg':
            network = model_object.basic_vgg()
        return model_object, network
    
    
    '''
    logfile_name: str
    model_network: object
    datagen: object
    default fit_type is no early stopping; change to 'early' if you want early stopping
    '''       
    
    def fit_model(self, model_network, train_datagen, fit_type = 'nostop'):
        fitter_object = keras_leaderboard.LB_Fit(self.X_train, self.y_train, self.X_val, self.y_val, self.batch_size, self.epochs, self.patience, self.logfile, 
                                                self.model_network_name, model_network, train_datagen)
        if fit_type == 'nostop':
            fitter_history, fitter_log = fitter_object.fit_nostop(self.dir)
        elif fit_type == 'early':
            fitter_history, fitter_log = fitter_object.fit_early(self.dir)
        return fitter_object, fitter_history, fitter_log
    
    
    def evaluate_model(self, history, metrics_name, model_network, save=True):
        metrics_object = keras_leaderboard.LB_Model_Metrics(history, metrics_name, model_network, self.X_val, self.y_val, self.n_classes, self.batch_size)
        
        # Save output values
        pred_values, pred_classes = metrics_object.softmax_predict()
        acc, prec, recall = metrics_object.scores(pred_classes)

        # Create plots
        acc, loss = metrics_object.losscurves()
        confusion_matrix = metrics_object.confusion_matrix_maker(pred_classes)
        rocs, roc_all = metrics_object.roc_auc(pred_values)
        
        # Save plots
        if save:
            
            np.save(os.path.join(self.dir, "predicted_values.npy"), pred_values) # Save prediction to numpy
            
            acc.savefig(os.path.join(self.plot_dir, f"{metrics_object.name}_accuracy.png"))
            print('Accuracy plot saved.')
            
            loss.savefig(os.path.join(self.plot_dir, f"{metrics_object.name}_loss.png"))
            print('Loss curve saved.')
            
            confusion_matrix.savefig(os.path.join(self.plot_dir, f"{metrics_object.name}_confusion_matrix.png"))
            print('Confusion matrix saved.')
            
            for k, plot in enumerate(rocs):
                plot.savefig(os.path.join(self.plot_dir, f"{metrics_object.name}_{k}_roc_curve.png"))    
            roc_all.savefig(os.path.join(self.plot_dir, f"{metrics_object.name}_roc_all_classes.png"))
            
            print('ROC curves saved.')
        
        # Show plots
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acc.show()
            loss.show()
            
            for plot in rocs: 
                plot.show()
            
        roc_all.show()
        
        return metrics_object, pred_values, pred_classes, acc, prec, recall 
    
       
    def model_configs(self, data_object, model_object, model_network, xtr_shape, ytr_shape, xval_shape, yval_shape, fitter_object, logfile, acc, prec, rec):
        model_configs_object = keras_leaderboard.leaderboard(data_object, model_object, model_network, self.augs, xtr_shape, ytr_shape, xval_shape, yval_shape, fitter_object, logfile, acc, prec, rec)
        
        model_configs_object.generate_configs() # Generate configuration and model outputs
        model_configs_object.save_configs(self.csv_dir) # Save configuration and model outputs to a csv
        
        return model_configs_object
    
    
    def make_dirs(self, save_folder):
        '''Make directory to save graphs and outputs'''
        id = dt.datetime.now()
        
        # Check if aggregate folder to save to
        if save_folder==None:
            dir_ = f"{self.model_network_name}_{id.month}-{id.day}-{id.hour}-{id.minute}"
        else:
            if os.path.isdir(save_folder)==False: # Make aggregate folder if it doesn't exist
                os.mkdir(save_folder)
                
            dir_ = os.path.join(save_folder, f"{self.model_network_name}_{id.month}-{id.day}-{id.hour}-{id.minute}")
            
        plot_dir = os.path.join(dir_, 'plots')
        csv_dir = os.path.join(dir_, 'csvs')
        
        os.mkdir(dir_) # Make directory
        os.mkdir(plot_dir) # Make directory for plots
        os.mkdir(csv_dir) # Make directory for output csvs
        
        with open(os.path.join(dir_,'model_parameters.txt'),'w') as f: # Save model params
            f.write(f"Model Name: {self.model_network_name}\n") # Record model name
            f.write(f"Batch Size: {self.batch_size}\n") # Record batch size parameter
            f.write(f"Epochs: {self.epochs}\n") # Record epoch parameter
            f.write(f"Optimizer: {self.optimizer}\n") # Record optimizer parameter
            
        return dir_, plot_dir, csv_dir