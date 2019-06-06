#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import tensorflow
import keras
from keras import layers
from keras import models
from keras import optimizers 
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input, Dense
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
import re
import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score
from scipy import interp
from itertools import cycle
import seaborn as sns 


class Register_Data(object):
    def __init__(self, X_train, y_train, X_val, y_val, name, augs):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.name = name
        self.augs = augs
        
    def register_shapes(self):
        xtr_shape = "%s" "_" "%s" "_" "%s" % (self.name, "X_train", self.X_train.shape)
        ytr_shape = "%s" "_" "%s" "_" "%s" % (self.name, "y_train", self.y_train.shape)
        xval_shape = "%s" "_" "%s" "_" "%s" % (self.name, "X_val", self.X_val.shape)
        yval_shape = "%s" "_" "%s" "_" "%s" % (self.name, "y_val", self.y_val.shape)
        return(xtr_shape, ytr_shape, xval_shape, yval_shape)
    
    def register_augs(self):
        #project_name enter as string using ''
        data_augs = "%s" "_" "%s" "_" "%s" % (self.name, "image_augs", self.augs) #string
        #instantiate augmenter
        train_datagen = ImageDataGenerator(data_augs)
        #fit X_train
        train_datagen.fit(self.X_train)
        return data_augs, train_datagen

class LB_Model(object):
    def __init__(self, width, height, channels, optimizer, loss, metrics, classes, batch_size, name):
        self.width = width
        self.height = height
        self.channels = channels
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.classes = classes
        self.batch_size = batch_size
        self.name = name
        
        
    def basic_cnn(self):
        main_input = Input(shape=(self.width, self.height, self.channels), name='main_input')
        x = Conv2D(32, (5, 5), activation='relu')(main_input)
        x = MaxPooling2D((2,2))(x)

        x = Conv2D(64, (5, 5), activation='relu')(x)
        x = MaxPooling2D((2,2))(x)

        x = Conv2D(128, (5, 5), activation='relu')(x)
        x = MaxPooling2D((2,2))(x)

        x = Flatten()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        main_output = Dense(self.classes, activation='softmax')(x)
        # This creates a model 
        model = Model(inputs=main_input, outputs=main_output, name='basic_cnn')

        model.compile(optimizer=self.optimizer,loss=self.loss, metrics=[self.metrics])
        
        print("Generated Basic CNN for:"+self.name)
        return model
    
    def basic_vgg(self):
        conv_base = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape = (self.width, self.height, self.channels))
        conv_base.trainable = False

        main_input = Input(shape=(self.width, self.height, self.channels), name='main_input')
        x = (conv_base)(main_input)
        x = Dense(256, activation='relu')(x)
        x = Dense(120, activation='relu')(x)
        main_output = Dense(self.classes, activation='softmax')(x)
        # This creates a model that includes
        # the conv_base and 3 dense layers
        model = Model(inputs=main_input, outputs=main_output)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metrics])

        print("Generated VGG_16 for:"+self.name)
        return model

class LB_Fit(object):
    def __init__(self, X_train, y_train, X_val, y_val, batch_size, epochs, patience, logfile, name, model, train_datagen):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size
        self.epochs = epochs
        self.logfile = logfile
        self.name = name
        self.model = model
        self.train_datagen = train_datagen
        self.patience = patience
        
    def fit_nostop(self, out_dir):
        train_generator = self.train_datagen.flow(self.X_train, self.y_train, self.batch_size)
        csv_logger = CSVLogger(os.path.join(out_dir, f"{self.logfile}.csv"), append=True, separator=';')
        history = self.model.fit_generator(train_generator,
                              steps_per_epoch=len(self.X_train) / self.batch_size,
                              epochs=self.epochs,
                              validation_data = (self.X_val, self.y_val),
                                      callbacks=[csv_logger])
        #save model
        self.model.save(os.path.join(out_dir, f"{self.name}.h5")) 
        log = pd.read_csv(os.path.join(out_dir, f"{self.logfile}.csv"), sep=';')
        score = self.model.evaluate(self.X_val, self.y_val, batch_size=self.batch_size)
        print("Accuracy is:", score)
        return history, log 
    
    def fit_early(self, out_dir):
        '''Fit with early stop'''
        
        # Make directory for models
        model_dir = os.path.join(out_dir,'models')
        os.mkdir(model_dir)
        
        train_generator = self.train_datagen.flow(self.X_train, self.y_train, batch_size=self.batch_size)
        csv_logger = CSVLogger(os.path.join(out_dir, f"{self.logfile}.csv"), append=True, separator=';')
        
        checkpointer = ModelCheckpoint(filepath=os.path.join(model_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), 
                                       verbose=1, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=1)

        history = self.model.fit_generator(train_generator,
                              steps_per_epoch=len(self.X_train) / self.batch_size,
                              epochs=self.epochs,
                              validation_data = (self.X_val, self.y_val),
                             callbacks=[csv_logger, checkpointer, earlystopper])
        log = pd.read_csv(os.path.join(out_dir, f"{self.logfile}.csv"), sep=';')
        score = self.model.evaluate(self.X_val, self.y_val, batch_size=self.batch_size)
        print("Accuracy is:", score)
        return history, log 

class LB_Model_Metrics(object): 
    def __init__(self, history, name, model, X_val, y_val, n_classes, batch_size):
        self.history = history
        self.name = name
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.n_classes = n_classes
        self.batch_size = batch_size
    
    def losscurves(self):
        #model_name is the string name of your model
        fig_acc, acc = plt.subplots()
        acc.plot(self.history.history['acc'])
        acc.plot(self.history.history['val_acc'])
        acc.set_title('model accuracy')
        acc.set_ylabel('accuracy')
        acc.set_xlabel('epoch')
        acc.legend(['train', 'test'], loc='upper left')

        # summarize history for loss
        fig_loss, loss = plt.subplots()
        loss.plot(self.history.history['loss'])
        loss.plot(self.history.history['val_loss'])
        loss.set_title('model loss')
        loss.set_ylabel('loss')
        loss.set_xlabel('epoch')
        loss.legend(['train', 'test'], loc='upper left')
        
        return fig_acc, fig_loss
        
    
    #returns probabilities and probability classes
    def softmax_predict(self):
        #yhat are the predictions need as argument for confusion matrix and roc
        yhat = self.model.predict(self.X_val, verbose=1)
        yhat_classes = yhat.argmax(axis=-1)
        
        #convert to the class
        return yhat, yhat_classes
    
    def confusion_matrix_maker(self, prediction_classes):
        actuals_classes = self.y_val.argmax(axis=-1)
    
        cm = confusion_matrix(prediction_classes, actuals_classes)

        sns.set_style('ticks')
        fig, ax = plt.subplots()

        # size of inches
        fig.set_size_inches(11, 8)
    
        if self.n_classes!=0: 
            xticks = []
            yticks = []

            for i in range(0, self.n_classes):
                ix = i
                xticks.append(str(ix))
                yticks.append(str(ix))
        
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=xticks, yticklabels=yticks)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            print(sklearn.metrics.classification_report(prediction_classes, actuals_classes, target_names = xticks))
            
            return plt
        
        elif self.n_classes ==0:
            print("No classes")

    def roc_auc(self, prediction_probas):
        y_test = self.y_val
        y_score = prediction_probas

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        print(roc_auc["micro"])
    
        #print individual classes
        lw = 2
        rocs = []
        for i in range (0, self.n_classes):
            fig, ax = plt.subplots()
            ax.plot(fpr[i], tpr[i], color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
            ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver operating characteristic example')
            ax.legend(loc="lower right")
            
            rocs.append(fig)
    
        #Plot together in one graph
    
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= self.n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves

        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]), 
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]), 
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'black'])
    
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        
        
        return rocs, plt
    
    
    def scores(self, prediction_classes):
        #note, prediction_classes is yhat_classes
        actuals_classes = self.y_val.argmax(axis=-1)
        accuracy = (self.model.evaluate(self.X_val, self.y_val, batch_size=self.batch_size))[1]
        precision = precision_score(actuals_classes, prediction_classes, average=None)  
        recall = recall_score(actuals_classes, prediction_classes, average=None)  
        return accuracy, precision, recall
        
class leaderboard(object):
    def __init__(self, data, model, network, augs, xtr_shape, ytr_shape, xval_shape, yval_shape, fitter, logfile, accuracy, precision, recall):
        self.data = data
        self.model = model
        self.network = network
        self.augs = augs
        self.xtr_shape = xtr_shape
        self.ytr_shape = ytr_shape
        self.xval_shape = xval_shape
        self.yval_shape = yval_shape
        self.fitter = fitter
        self.logfile = logfile
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        
    #model outputs    
    def get_model_outputs(self):
        try: 
            inp = self.network.input                                           
            outputs = [layer.output for layer in self.network.layers] #output list
            #regex out the shapes
            shapes = []
            for i in range (0, len(outputs)):
                mstr = str(outputs[i])
                start = 'shape=\('
                end = '\),'
                result = re.search('%s(.*)%s' % (start, end), mstr).group(1)
                shapes.append(result)
            #regex out the tensors
            tensors = []
            for i in range (0, len(outputs)):
                mstr = str(outputs[i])
                start = '\("'
                end = '"'
                result = re.search('%s(.*)%s' % (start, end), mstr).group(1)
                tensors.append(result)
           #make dfs
            sh = pd.DataFrame(shapes)
            te = pd.DataFrame(tensors)
            outputs = pd.concat([te, sh], axis=1)
            outputs.columns=['tensor', 'shape']
            return outputs
        
        except:
            print("Something went wrong, here's an empty dataframe")
            empty = pd.DataFrame()
            
            return empty


    #returns layers for each model
    def get_model_layers(self):
        layer_dict= dict([(layer.name, layer) for layer in self.network.layers])
        layers = pd.DataFrame.from_dict(layer_dict, orient='index')
        layers.columns=['object']
        
        return layers
    
    #returns configs for each layer
    def get_layer_configs(self):
        configs = self.network.get_config() #get all the model configs
        configs = configs['layers'] #just the configs for the layers
        c_df = pd.DataFrame.from_dict(configs)

        return c_df
    
    
    def make_csv(self):
        params = {'myaugs': [self.augs], 
                  'xtrain_shape': self.xtr_shape, #str
                  'ytrain_shape': self.ytr_shape, #str
                  'xval_shape': self.xval_shape, #str
                  'yval_shape': self.yval_shape, #str
                  'image_width': self.model.width, #scalar
                  'image_height': self.model.height, #scalar
                  'image_channels': self.model.channels, #scalar
                  'model_name': self.model.name, #str
                  'model optimizer': self.model.optimizer, #str
                  'model_loss_metric': self.model.loss, #str
                  'num_classes': self.model.classes, #scalar
                  'model_batchsize': self.model.batch_size, #scalar
                  'num_epochs': self.fitter.epochs, #scalar
                  'model_training_log': [self.logfile], #pandas df
                  'model_val_acc': self.accuracy, #scalar
                  'model_prec': [self.precision], #list
                  'model_recall': [self.recall], #list
                  'data_name':self.data.name}
        df = pd.DataFrame.from_dict(params)           
        df1 = df.reindex_axis(['data_name', 
                               'model_name',
                               'model_val_acc',
                               'model_prec',
                               'model_recall',
                               'myaugs', 
                               'xtrain_shape',
                           'ytrain_shape',
                           'xval_shape',
                       'yval_shape',
                       'image_width',
                       'image_height',
                       'image_channels',
                       'model optimizer',
                       'model_loss_metric',
                       'num_classes',
                       'model_batchsize',
                       'num_epochs',
                      'model_training_log'
                              ], axis=1)

        return df1   
    
    def generate_configs(self):
        self.model_outputs = self.get_model_outputs()
        self.model_layers = self.get_model_layers()
        self.model_configs = self.get_layer_configs()
        self.model_df = self.make_csv()
    
    def save_configs(self, output_directory):
        self.model_outputs.to_csv(os.path.join(output_directory, f"{self.model.name}_outputs.csv"))
        self.model_layers.to_csv(os.path.join(output_directory, f"{self.model.name}_layers.csv"))
        self.model_configs.to_csv(os.path.join(output_directory, f"{self.model.name}_configs.csv"))
        self.model_df.to_csv(os.path.join(output_directory, f"{self.model.name}_leaderboard.csv"), index = 0)
        