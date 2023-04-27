import tensorflow as tf
import keras
import numpy as np
from keras.layers import Input, Dense, Dropout, concatenate, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from PIL import Image
from skimage.color import gray2rgb
from tensorflow import keras
import os
import cv2
import matplotlib.image as imag
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import random

from help_functions_register_data import *
from load_images import *
from image_model import *
from help_functions_prob_fusion import plt_roc_patient_based_fusion, get_im_and_labels_as_array_for_fusion

def ff_image_model(model_path='image_model_patients_extended_alt_lr/fold_4'):
    reconstructed_model = keras.models.load_model(model_path)
    reconstructed_model.trainable = False
    feature_map = reconstructed_model.layers[4].input
    feat_extract_model = keras.Model(reconstructed_model.input, feature_map)

    return feat_extract_model

def Feature_Fusion_model(lr=1e-5, pat=50, im_model_path = 'image_model_patients_extended_alt_lr/fold_4', num_reg_param = 7):

    image_model = ff_image_model(im_model_path)
    print('image model loaded')

    image_input = keras.Input(shape=(224, 224, 3))
    register_input = keras.Input(shape=(num_reg_param,))

    image_output = image_model(image_input, training=False)
    image_norm_layer = LayerNormalization(axis=-1, scale = False, center = False, name='img_norm')
    image_output = image_norm_layer(image_output)

    reg_norm_layer = LayerNormalization(axis=-1, scale = False, center = False, name='reg_norm')
    register_output = reg_norm_layer(register_input)

    combined = concatenate([image_output, register_output])
    output = Dense(1024, activation='relu', name='dense_shallow1')(combined)
    output = Dropout(0.5, name='dropout_shallow1')(output)
    output = Dense(512, activation='relu', name='dense_shallow2')(output)
    output = Dropout(0.5, name='dropout_shallow2')(output)
    output = Dense(256, activation='relu', name='dense_shallow3')(output)
    output = Dropout(0.5, name='dropout_shallow3')(output)
    output = Dense(1, activation='sigmoid', name='dense_shallow4')(output)

    model = keras.Model(inputs=[image_input, register_input], outputs=output)

    optimizer = Adam(learning_rate=lr)
    loss = 'binary_crossentropy'
    metric = [BinaryAccuracy(),
                keras.metrics.AUC(name='auc', curve='ROC'),
                ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metric)

    callback_metric = EarlyStopping(monitor='val_loss', patience=pat, mode='min', restore_best_weights=True, verbose=2)

    #Metric for learning-rate-LearningRateScheduler
    callback_lr = LearningRateScheduler(scheduler_ff, verbose = 1)

    callback = [callback_metric, callback_lr]

    return model, callback, optimizer


def Feature_Fusion_train(model, X_train, Y_train, X_valid, Y_valid, callback, class_weights, optimizer, num_epochs):

    history = model.fit(X_train, Y_train, epochs=num_epochs, validation_data=(X_valid, Y_valid), shuffle=True, callbacks=callback, class_weight=class_weights, verbose=2)

    return history

def scheduler_ff(epoch, lr):

    if lr > 1e-5:
        return lr * tf.math.exp(-0.01)
    else:
        return lr

def crossvalidation_FF_model(n_folds=5, learning_rate=1e-4, patience=50, epochs=1000, save_csv=False, model_num=1, folder='csv_feature_fusion', variable_selection_func = load_registerdata_7_nonbinary_fold, only_age_sex = False, all_register_parameters = False, train_metrics_path='Plots_for_feature_fusion_2_param', save_model_path='feature_fusion_2_param_saved_models', repetition=1):

    num_reg_param = 7

    if only_age_sex:
        num_reg_param = 2

    if all_register_parameters:
        num_reg_param = 45

    if save_csv == True:
        #Clear csv file
        path_plot = folder + '/results_cross_val_model' + str(model_num) + '_repetition_' + str(repetition) + '.csv'
        f = open(path_plot,'w+')
        f.close()
        path_mean = folder + '/roc_mean_model' + str(model_num) + '_repetition_' + str(repetition) + '.csv'
        f = open(path_mean,'w+')
        f.close()

    folder_for_numpys = 'numpys_for_crossval/'
    registerdata_path_fold1 = folder_for_numpys + 'registerdata_fold_1.npy'
    registerdata_path_fold2 = folder_for_numpys + 'registerdata_fold_2.npy'
    registerdata_path_fold3 = folder_for_numpys + 'registerdata_fold_3.npy'
    registerdata_path_fold4 = folder_for_numpys + 'registerdata_fold_4.npy'
    registerdata_path_fold5 = folder_for_numpys + 'registerdata_fold_5.npy'
    registerdata_path_fold6 = folder_for_numpys + 'registerdata_fold_6.npy'

    registerdata_lst_paths = []
    registerdata_lst_paths.append(registerdata_path_fold1)
    registerdata_lst_paths.append(registerdata_path_fold2)
    registerdata_lst_paths.append(registerdata_path_fold3)
    registerdata_lst_paths.append(registerdata_path_fold4)
    registerdata_lst_paths.append(registerdata_path_fold5)
    registerdata_lst_paths.append(registerdata_path_fold6)

    im_path_fold1 = folder_for_numpys + 'im_fold_1.npy'
    im_path_fold2 = folder_for_numpys + 'im_fold_2.npy'
    im_path_fold3 = folder_for_numpys + 'im_fold_3.npy'
    im_path_fold4 = folder_for_numpys + 'im_fold_4.npy'
    im_path_fold5 = folder_for_numpys + 'im_fold_5.npy'
    im_path_fold6 = folder_for_numpys + 'im_fold_6.npy'

    im_lst_paths = []
    im_lst_paths.append(im_path_fold1)
    im_lst_paths.append(im_path_fold2)
    im_lst_paths.append(im_path_fold3)
    im_lst_paths.append(im_path_fold4)
    im_lst_paths.append(im_path_fold5)
    im_lst_paths.append(im_path_fold6)

    labels_path_fold1 = folder_for_numpys + 'labels_fold_1.npy'
    labels_path_fold2 = folder_for_numpys + 'labels_fold_2.npy'
    labels_path_fold3 = folder_for_numpys + 'labels_fold_3.npy'
    labels_path_fold4 = folder_for_numpys + 'labels_fold_4.npy'
    labels_path_fold5 = folder_for_numpys + 'labels_fold_5.npy'
    labels_path_fold6 = folder_for_numpys + 'labels_fold_6.npy'

    labels_lst_paths = []
    labels_lst_paths.append(labels_path_fold1)
    labels_lst_paths.append(labels_path_fold2)
    labels_lst_paths.append(labels_path_fold3)
    labels_lst_paths.append(labels_path_fold4)
    labels_lst_paths.append(labels_path_fold5)
    labels_lst_paths.append(labels_path_fold6)

    registerdata_lst_np = []
    im_lst_np = []
    labels_lst_np = []

    for i in range(0, len(registerdata_lst_paths)):
        if i == (len(registerdata_lst_paths) - 1):
            registerdata_test = variable_selection_func(registerdata_lst_paths[i])
            im_test = np.load(im_lst_paths[i])
            patient_IDs_test = np.load(folder_for_numpys + 'patients_fold_6.npy')
            labels_test = np.load(labels_lst_paths[i])
        else:
            registerdata_lst_np.append(variable_selection_func(registerdata_lst_paths[i]))
            im_lst_np.append(np.load(im_lst_paths[i]))
            labels_lst_np.append(np.load(labels_lst_paths[i]))

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    lst_acc = []
    lst_sens = []
    lst_spec = []
    lst_mcc = []

    # Compensate for unbalanced dataset, via class weights
    n_samples = np.vstack(registerdata_lst_np).shape[0]
    n_classes = len(set(np.hstack(labels_lst_np)))
    class_weights = n_samples / (n_classes * np.bincount(np.hstack(labels_lst_np)))
    class_weights = {0: class_weights[0], 1: class_weights[1]}

    for i in range(0, len(registerdata_lst_np)):
        copy_folds_registerdata = registerdata_lst_np.copy()
        copy_folds_im = im_lst_np.copy()
        copy_folds_labels = labels_lst_np.copy()

        registerdata_valid = copy_folds_registerdata[i]
        im_valid = copy_folds_im[i]
        labels_valid = copy_folds_labels[i]

        del copy_folds_registerdata[i]
        del copy_folds_im[i]
        del copy_folds_labels[i]

        registerdata_train = np.vstack(copy_folds_registerdata)
        im_train = np.vstack(copy_folds_im)
        labels_train = np.hstack(copy_folds_labels)

        labels_train_non_zero = np.count_nonzero(labels_train)
        labels_test_non_zero = np.count_nonzero(labels_test)
        labels_valid_non_zero = np.count_nonzero(labels_valid)

        print('% AFF train: ', labels_train_non_zero/len(labels_train))
        print('% AFF test: ', labels_test_non_zero/len(labels_test))
        print('% AFF valid: ', labels_valid_non_zero/len(labels_valid))

        print("\n -------------------- FF fold", i+1, "---------------------------")

        #load model and train
        feature_fusion_model, callback, optimizer = Feature_Fusion_model(lr=learning_rate, pat=patience, im_model_path = 'image_model_weights/fold_' + str(i) + '.h5', num_reg_param = num_reg_param)

        history = Feature_Fusion_train(feature_fusion_model, X_train=[im_train, registerdata_train], Y_train=labels_train, X_valid=[im_valid, registerdata_valid], Y_valid=labels_valid, callback=callback, class_weights=class_weights, optimizer=optimizer, num_epochs=epochs)

        Y_pred = feature_fusion_model.predict([im_test, registerdata_test])

        new_labels_pred, new_patient_IDs, new_labels_test = pred_patient(Y_pred, patient_IDs_test, labels_test)

        mean_tpr, mean_fpr, lst_acc, lst_sens, lst_spec, lst_mcc = Evaluation_CrossVal_patients(history, new_labels_test, new_labels_pred, feature_fusion_model, save_csv, i, model_num, mean_tpr, mean_fpr, lst_acc, lst_sens, lst_spec, lst_mcc, folder=folder, repetition=repetition)

        plt_acc(path_plot, fold=i, path_save=train_metrics_path, repetition=repetition)
        plt_loss(path_plot, fold=i, path_save=train_metrics_path, repetition=repetition)

        feature_fusion_model.save(save_model_path + '/fold_' + str(i+1) + '.h5')

        keras.backend.clear_session()

    mean_tpr /= n_folds
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)

    #Calculate mean and std of accuracy and sensitivity, across all folds
    array_acc = np.asarray(lst_acc)
    array_sens = np.asarray(lst_sens)
    array_spec = np.asarray(lst_spec)
    array_mcc = np.asarray(lst_mcc)

    mean_acc = np.mean(array_acc)
    std_acc = np.std(array_acc)
    mean_sens = np.mean(array_sens)
    std_sens = np.std(array_sens)
    mean_spec = np.mean(array_spec)
    std_spec = np.std(array_spec)
    mean_mcc = np.mean(array_mcc)
    std_mcc = np.std(array_mcc)

    # Save mean roc
    if save_csv:
        path = folder + '/roc_mean_model' + str(model_num) + '_repetition_' + str(repetition) + '.csv'
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([mean_tpr, mean_fpr, mean_auc, mean_acc, std_acc, mean_sens, std_sens, mean_spec, std_spec, mean_mcc, std_mcc])
            file.close()

            
