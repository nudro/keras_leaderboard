#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import skimage.io
import skimage.transform
import tensorflow
import run_leaderboard

np.random.seed(21)
tensorflow.set_random_seed(21)

if __name__ == '__main__':

    ####DATA PROCESSING####
    
    print("Processing data...")
    
    # Parameters
    training_dataset_path = "dataset_updated/training_set"
    test_dataset_path = "dataset_updated/validation_set"
    wild_test_dataset_path = 'wild_test'
    
    # categories to use
    categories = ['drawings', 'engraving', 'iconography', 'painting', 'sculpture']
    n_categories = len(categories)
    category_embeddings = {
        'drawings': 0,
        'engraving': 1,
        'iconography': 2,
        'painting': 3,
        'sculpture': 4
    }
    
    # set width, height, RGB
    width = 100 
    height = 100 
    n_channels = 3
    
    training_data = []
    for cat in categories:
        files = os.listdir(os.path.join(training_dataset_path, cat))
        for file in files:
            training_data += [(os.path.join(cat, file), cat)]
    
    test_data = []
    for cat in categories:
        files = os.listdir(os.path.join(test_dataset_path, cat))
        for file in files:
            test_data += [(os.path.join(cat, file), cat)]
            
    print("Loading images...")
            
    # Load all images to the same format (takes some time)
    def load_dataset(tuples_list, dataset_path):
        indexes = np.arange(len(tuples_list))
        np.random.shuffle(indexes)
        
        X = []
        y = []
        n_samples = len(indexes)
        cpt = 0
        for i in range(n_samples):
            t = tuples_list[indexes[i]]
            try:
                img = skimage.io.imread(os.path.join(dataset_path, t[0]))
                img = skimage.transform.resize(img, (width, height, n_channels), mode='reflect')
                X += [img]
                y_tmp = [0 for _ in range(n_categories)]
                y_tmp[category_embeddings[t[1]]] = 1
                y += [y_tmp]
            except OSError:
                pass
            
            cpt += 1
            
            if cpt % 1000 == 0:
                print("Processed {} images".format(cpt))
    
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    X_train, y_train = load_dataset(training_data, training_dataset_path)
    X_val, y_val = load_dataset(test_data, test_dataset_path)
    
    print("Your train and val shapes...")
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    
    print("I'm starting to register the data, build the model and leaderboard.")
    ###declare your augmentations
    
    myaugs = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
    
    
    #function to create any number of CNN's based on batch_size and epochs
    def make_cnns(batch_size, epochs, optimizer):

            cnn_object = run_leaderboard.make_leaderboard(X_train = X_train, 
                                    y_train = y_train, 
                                    X_val = X_val, 
                                    y_val = y_val, 
                                    data_name = 'art_data', 
                                    augs = myaugs, 
                                    width =100, 
                                    height = 100, 
                                    channels = 3, 
                                    optimizer = optimizer, 
                                    loss = 'categorical_crossentropy', 
                                    metrics = 'accuracy', 
                                    classes = 5, 
                                    n_classes = 5, #same as classes
                                    batch_size = batch_size, 
                                    model_network_name = 'art_cnn',
                                    epochs = epochs, 
                                    patience = 3, 
                                    logfile = 'cnn_log',
                                    save_folder = 'test')

            art_data, xtr_shape, ytr_shape, xval_shape, yval_shape, data_augs, art_train_datagen = cnn_object.register_the_data()
            art_model_cnn, art_cnn = cnn_object.build_and_compile_CNN(model_type='cnn')
            art_cnn_fitter, art_cnn_history, art_cnn_log = cnn_object.fit_model(model_network = art_cnn, train_datagen = art_train_datagen, fit_type = 'nostop') 
            art_cnn_metrics, yhat_predictions_art, yhat_classes_art, cnn_accuracy, cnn_precision, cnn_recall = cnn_object.evaluate_model(history=art_cnn_history, 
                                                                                                                                 metrics_name='art_cnn_metrics', 
                                                                                                                                 model_network = art_cnn)
            cnn_configs = cnn_object.model_configs(data_object = art_data,
                                                    model_object = art_model_cnn,
                                                    model_network = art_cnn,
                                                    xtr_shape = xtr_shape, 
                                                    ytr_shape = ytr_shape, 
                                                    xval_shape = xval_shape, 
                                                    yval_shape = yval_shape, 
                                                    fitter_object = art_cnn_fitter, 
                                                    logfile = art_cnn_log, 
                                                    acc = cnn_accuracy, 
                                                    prec = cnn_precision, 
                                                    rec = cnn_recall)
            return cnn_configs.model_df

    
    #MAKE VGGs
    def make_vggs(batch_size, epochs, optimizer):
        vgg_object = run_leaderboard.make_leaderboard(X_train = X_train, 
                                y_train = y_train, 
                                X_val = X_val, 
                                y_val = y_val, 
                                data_name = 'art_data', 
                                augs = myaugs, 
                                width =100, 
                                height = 100, 
                                channels = 3, 
                                optimizer = optimizer, 
                                loss = 'categorical_crossentropy', 
                                metrics = 'accuracy', 
                                classes = 5, 
                                n_classes = 5, #same as classes
                                batch_size = batch_size, 
                                model_network_name = 'art_vgg',
                                epochs = epochs, 
                                patience = 3,
                                logfile = 'vgg_log',
                                save_folder='test')    

        art_data, xtr_shape, ytr_shape, xval_shape, yval_shape, data_augs, art_train_datagen = vgg_object.register_the_data()
        art_model_vgg, art_vgg = vgg_object.build_and_compile_CNN(model_type='vgg')
        art_vgg_fitter, art_vgg_history, art_vgg_log = vgg_object.fit_model(art_vgg, art_train_datagen, fit_type = 'early')
        art_vgg_metrics, yhat_predictions_art, yhat_classes_art, vgg_accuracy, vgg_precision, vgg_recall = vgg_object.evaluate_model(history=art_vgg_history, 
                                                                                                                             metrics_name='art_vgg_metrics', 
                                                                                                                             model_network = art_vgg)
        
        vgg_configs = vgg_object.model_configs(data_object = art_data,
                                                model_object = art_model_vgg,
                                                model_network = art_vgg,
                                                xtr_shape = xtr_shape, 
                                                ytr_shape = ytr_shape, 
                                                xval_shape = xval_shape, 
                                                yval_shape = yval_shape, 
                                                fitter_object = art_vgg_fitter, 
                                                logfile = art_vgg_log, 
                                                acc = vgg_accuracy, 
                                                prec = vgg_precision, 
                                                rec = vgg_recall)
        return vgg_configs.model_df




    #MAKE MODELS
    optimizer = 'adam'
            
    cnn1 = make_cnns(32, 5, optimizer)
    cnn2 = make_cnns(32, 10, optimizer)
    cnn3 = make_cnns(64, 10, optimizer)
    
    vgg1 = make_vggs(32, 5, optimizer)
    vgg2 = make_vggs(32, 10, optimizer)
    vgg3 = make_vggs(64, 10, optimizer)



   ##append into a larger leaderboard
    
    final_lb = cnn1.append([cnn2, cnn3, vgg1, vgg2, vgg3])
    final_lb.to_csv('final_lb.csv')
    
    print("Your leaderboard CSV is done.")
