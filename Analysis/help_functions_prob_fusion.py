### help functions for fusion ###

import tensorflow as tf
import keras
import numpy as np
from keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from PIL import Image
from skimage.color import gray2rgb
from tensorflow import keras
from sklearn.metrics import matthews_corrcoef
import os
import cv2
import matplotlib.image as imag
import matplotlib.pyplot as plt
import math
import random

from help_functions_register_data import *
from load_images import *
from image_model import *


def prepare_data_for_fusion(path = '/local/data1/andek67/AFFfusion/Data/images', img_size=224, folder_save = 'numpys_for_crossval', norm=False):

    registerdata = pd.read_csv('registerdata_all.csv')

    registerdata = np.asarray(registerdata.values)

    data = read_im_and_label_to_dict_fusion(path, registerdata, img_size=img_size, norm=norm)

    CrossValidation_to_divide_data(data, folder_save = folder_save, n_folds=5)


def read_im_and_label_to_dict_fusion(path, registerdata, img_size=224, norm=False):

    counter = 0
    data = {}

    assert len(os.listdir(path)) != 0, "directory is empty"

    for x in os.listdir(path):
        if x.endswith(".png"):

            print('Image counter: ', counter)

            file_name = str(x)

            path_im = path + '/' + file_name

            file_name_words = file_name.split('_')
            patient_number = int(file_name_words[1])

            img = cv2.imread(path_im)

            shape = np.shape(img)
            height = shape[0]
            h_c = height/2
            width = shape[1]
            w_c = width/2
            pad_size = max(height, width)
            p_c = pad_size/2
            pad_im = np.zeros((pad_size, pad_size, 3))

            pad_im[math.ceil(p_c-h_c):math.ceil(p_c+h_c),math.ceil(p_c-w_c):math.ceil(p_c+w_c)] = img

            pad_im_resize = cv2.resize(pad_im, (img_size, img_size), interpolation = cv2.INTER_AREA)
            if norm:
                pad_im_resize = normalize_im(pad_im_resize)
                print('normalize im!')
            registerdata_for_a_patient = None

            for i in range(0, registerdata.shape[0]):
                if registerdata[i,-2] == patient_number:

                    registerdata_for_a_patient = registerdata[i,:-1]

            if "_AFF_" in path_im:
                patient_label = 1
            else:
                patient_label = 0

            data_keys = data.keys()

            if not patient_number in data_keys:
                data[patient_number] = {"images" : [pad_im_resize],
                "label" : patient_label,
                "registerdata" : registerdata_for_a_patient}
            else:
                data[patient_number]["images"].append(pad_im_resize)

        counter+=1

    return data

def CrossValidation_to_divide_data(data_dict, folder_save, n_folds=5):
    n_folds = n_folds + 1
    folder = folder_save

    num_patients_per_fold = math.ceil(len(data_dict) / n_folds)
    folds_im = []
    folds_labels = []
    folds_registerdata = []

    im_test = np.array([])
    label_test = np.array([])
    registerdata_test = np.array([])

    data_dict_list = list(data_dict.items())

    for i in range(0,n_folds):
        index1 = i*num_patients_per_fold
        index2 = (i+1)*num_patients_per_fold

        temp_dict_list = data_dict_list[index1:index2]
        temp_images, temp_labels, temp_patients, temp_registerdata = get_im_and_labels_as_array_for_fusion(temp_dict_list)

        numpy_im_title = folder + '/im_fold_'+  str(i+1) + '.npy'
        np.save(numpy_im_title, temp_images)

        numpy_labels_title = folder + '/labels_fold_'+  str(i+1) + '.npy'
        np.save(numpy_labels_title, temp_labels)

        numpy_patients_title = folder + '/patients_fold_'+  str(i+1) + '.npy'
        np.save(numpy_patients_title, temp_patients)

        numpy_registerdata_title = folder + '/registerdata_fold_'+  str(i+1) + '.npy'
        np.save(numpy_registerdata_title, temp_registerdata)
        print('Saved numpy arrays!')

def load_pretrained_register_data_model_pf(checkpoint_path='RegisterDataModels/Model_fold_3.cp.ckpt', restored_weights=True, num_dim=7):
    activation='relu'
    model_num = 2

    model = Sequential()

    if model_num == 1:
        model.add(Dense(50, input_dim=num_dim, activation=activation))
        model.add(Dropout(0.25))
        print('------------ model_num == 1 ------------')
    if model_num == 2:
        model.add(Dense(512, input_dim=num_dim, activation=activation, name='dense1_rd'))
        model.add(Dropout(0.25, name='dropout1_rd'))
        model.add(Dense(256, activation=activation, name='dense2_rd'))
        model.add(Dropout(0.25, name='dropout2_rd'))
        model.add(Dense(128, activation = activation, name='dense3_rd'))
        model.add(Dropout(0.25, name='dropout3_rd'))
        print('------------ model_num == 2 ------------')
    if model_num == 3:
        model.add(Dense(1024, input_dim=num_dim, activation=activation))
        model.add(Dense(512, activation=activation))
        model.add(Dropout(0.25))

        print('------------ model_num == 3 ------------')
    if model_num == 4:
        model.add(Dense(256, input_dim=num_dim, activation=activation))
        model.add(Dense(256, activation=activation))
        model.add(Dense(128, activation=activation))
        model.add(Dense(64, activation=activation))
        model.add(Dense(32, activation=activation))
        print('------------ model_num == 4 ------------')

    model.add(Dense(1, activation='sigmoid', name='last_dense_in_rd_model'))

    # Configurate model
    metric = [BinaryAccuracy(),
                keras.metrics.AUC(name='auc', curve='ROC'),
                ]
    optimizer = Adam(learning_rate=1e-4)
    loss = 'binary_crossentropy'

    model.compile(optimizer=optimizer, loss=loss, metrics=metric)

    if restored_weights:
        model.load_weights(checkpoint_path)
        print('Loaded weights')
    else:
        print('Did not load weights')

    return model, optimizer, loss, metric

def get_im_and_labels_as_array_for_fusion(dataset_list):

    #Create a list of images and a corresponding list of labels, for each dataset
    ds_images = []
    ds_labels = []
    ds_patients = []
    ds_registerdata = []

    for patient in dataset_list:
        #ds_patients.append(patient[0])
        inner_dict = patient[1]
        images = inner_dict["images"]
        label = inner_dict["label"]
        registerdata = inner_dict["registerdata"]
        for image in images:
            ds_images.append(image)
            ds_labels.append(label)
            ds_patients.append(patient[0])
            ds_registerdata.append(registerdata)

    ds_images = np.asarray(ds_images)
    ds_labels = np.asarray(ds_labels)
    ds_patients = np.asarray(ds_patients)
    ds_registerdata = np.asarray(ds_registerdata)

    return ds_images, ds_labels, ds_patients, ds_registerdata

def crossvalidation_PF_model(n_folds=5, learning_rate=1e-4, patience=50, epochs=1000, save_csv=False, model_num=1, folder='csv_prob_fusion', variable_selection_func = load_registerdata_7_nonbinary_fold, only_age_sex = False, all_register_parameters = False, train_metrics_path = None, save_model_path = None, repetition=1):

    registerdata_weights = 'RegisterDataModels'

    if only_age_sex:
        num_reg_var = 2
        #registerdata_weights = 'RegisterDataModels_age_sex'
        print("Only age and sex")
    else:
        num_reg_var = 7
        
    if all_register_parameters:
        num_reg_var = 45

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
    print('class_weights: ', class_weights)

    model_number = model_num[0]

    # Loop over folds
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

        print("\n -------------------- PF fold", i+1, "---------------------------")

        #load model and train
        prob_fusion_model, callback, optimizer = alt_Probability_Fusion_model(image_model_path='image_model_weights/fold_' + str(i) + '.h5', registerdata_weights=registerdata_weights + '/Model_fold_' + str(i+1) + '_cp.ckpt', lr=learning_rate, pat=patience, num_reg_var=num_reg_var)

        history = Probability_Fusion_train(prob_fusion_model, X_train=[im_train, registerdata_train], Y_train=labels_train, X_valid=[im_valid, registerdata_valid], Y_valid=labels_valid, callback=callback, class_weights=class_weights, optimizer=optimizer, num_epochs=epochs)

        Y_pred = prob_fusion_model.predict([im_test, registerdata_test])

        new_labels_pred, new_patient_IDs, new_labels_test = pred_patient(Y_pred, patient_IDs_test, labels_test)

        mean_tpr, mean_fpr, lst_acc, lst_sens, lst_spec, lst_mcc = Evaluation_CrossVal_patients(history, new_labels_test, new_labels_pred, prob_fusion_model, save_csv, i, model_num, mean_tpr, mean_fpr, lst_acc, lst_sens, lst_spec, lst_mcc, folder=folder, repetition=repetition)

        plt_acc(path_plot, fold=i, path_save=train_metrics_path, repetition=repetition)
        plt_loss(path_plot, fold=i, path_save=train_metrics_path, repetition=repetition)

        prob_fusion_model.save(save_model_path + '/fold_' + str(i+1) + '.h5')

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



def load_pretrained_image_model_pf(model_path='image_model_patients_extended_alt_lrl/fold_4'):
    reconstructed_model = keras.models.load_model(model_path)
    reconstructed_model.trainable = False

    metric = [keras.metrics.BinaryAccuracy(), keras.metrics.AUC(name='auc', curve="ROC"),]
    learning_rate = 1e-4
    opt = keras.optimizers.Adam(learning_rate)
    reconstructed_model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = metric)

    return reconstructed_model



def scheduler_pf(epoch, lr):

    if lr > 1e-4:
        return lr * tf.math.exp(-0.01)
    else:
        return lr

def alt_Probability_Fusion_model(image_model_path='image_model_patients_model/fold_4', registerdata_weights='RegisterDataModels/Model_fold_2.cp.ckpt', lr=1e-4, pat=50, num_reg_var = 7):

    image_model = load_pretrained_image_model_pf(model_path=image_model_path)
    image_model.summary()
    print('Image model loaded')

    registerdata_model, opt, loss, metrics = load_pretrained_register_data_model_pf(checkpoint_path=registerdata_weights, restored_weights=True, num_dim=num_reg_var)
    registerdata_model.trainable = False
    registerdata_model.compile(optimizer=opt, loss=loss, metrics=metrics)
    registerdata_model.summary()
    print('Registerdata model loaded')

    image_input = keras.Input(shape=(224, 224, 3))
    register_input = keras.Input(shape=(num_reg_var,))

    image_output = image_model(image_input, training=False)
    register_output = registerdata_model(register_input, training=False)

    combined = concatenate([image_output, register_output])

    output = Dense(16, activation='relu', name='dense_shallow1')(combined)
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
    callback_lr = LearningRateScheduler(scheduler_pf, verbose = 1)

    callback = [callback_metric, callback_lr]

    model.summary()

    return model, callback, optimizer

def Probability_Fusion_train(model, X_train, Y_train, X_valid, Y_valid, callback, class_weights, optimizer, num_epochs):

    history = model.fit(X_train, Y_train, epochs=num_epochs, validation_data=(X_valid, Y_valid), shuffle=True, callbacks=callback, class_weight=class_weights, verbose=2)

    return history

def plt_roc_patient_based_fusion(filename='results_cross_val_model', filename_mean_roc='roc_mean_model', save_or_show_fig='show', model_num=1, path_to_place_plots_in='Plots_for_prob_fusion', model_type = None, repetition=1):
    num_folds = 0
    sum_roc_auc = 0
    array_roc_auc = np.array([])
    filename = filename + str(model_num) + '_repetition_' + str(repetition) + '.csv'
    filename_mean_roc  = filename_mean_roc + str(model_num) + '_repetition_' + str(repetition) + '.csv'
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file, delimiter=',')
        c = 0
        for row in reader:
            if c == 0:
                header = row
                c = c + 1
            else:
                c = c + 1
                model_nr = row[0]
                tpr_temp = row[5]
                fpr_temp = row[6]
                roc_auc = row[7]
                epochs = row[8]
                num_folds = num_folds + 1

                tpr_temp = make_plot_lst(tpr_temp, mode=" ")
                fpr_temp = make_plot_lst(fpr_temp, mode=" ")
                tpr_temp = np.asarray(tpr_temp)
                fpr_temp = np.asarray(fpr_temp)

                plt.plot(fpr_temp, tpr_temp, lw=1, label='Fold {:d}, Epoch {:d}, AUC = {:.3f}'.format(int(model_nr), int(epochs), float(roc_auc)))

    with open(filename_mean_roc, 'r', newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            tpr_mean = row[0]
            fpr_mean = row[1]
            roc_auc_mean = row[2]

            tpr_mean = make_plot_lst(tpr_mean, mode=" ")
            fpr_mean = make_plot_lst(fpr_mean, mode=" ")

            plt.plot(fpr_mean, tpr_mean, 'k--', label=r'Mean ROC, AUC = {:.3f}'.format(float(roc_auc_mean)))
    plt.plot([0,1], [0,1], '--', color=(0.6,0.6,0.6), label='Random Classifier, AUC = 0.5')
    model_num_title = str(model_num).split('_')
    model_num_title = model_num_title[0]
    plt.title(model_type + ': ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    if save_or_show_fig == 'show':
        plt.show()
    else:
        plt.savefig(path_to_place_plots_in + '/ROC_model' + str(model_num) + '_repetition_' + str(repetition) + '.eps')
        plt.savefig(path_to_place_plots_in + '/ROC_model' + str(model_num) + '_repetition_' + str(repetition) + '.png')

