##############################################
## help functions for register data network ##
##############################################

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from read_csv_for_plts import plt_acc, plt_loss
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import class_weight
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from scipy import interp
import csv
import math
import os

def find_patients_with_images(path = "/local/data1/andek67/AFFfusion/Data/images"):

    patients_lst = []
    counter_different_patients = 0

    assert len(os.listdir(path)) != 0, "directory is empty"

    for x in os.listdir(path):
        if x.endswith(".png"):

            file_name = str(x)
            path_im = path + '/' + file_name
            file_name_words = file_name.split('_')
            patient_number = int(file_name_words[1])

            if not patient_number in patients_lst:
                patients_lst.append(patient_number)
                counter_different_patients = counter_different_patients + 1
                print('counter_different_patients: ', counter_different_patients)

    patients_lst.sort()

    return patients_lst

def control_removed_data(fold = 1, folder_original_data='numpys_for_crossval', folder_new_data='numpys_for_crossval_norm_im'):

    original_patients = np.load(folder_original_data + '/patients_fold_' + str(fold) + '.npy')
    original_labels = np.load(folder_original_data + '/labels_fold_' + str(fold) + '.npy')
    original_registerdata = np.load(folder_original_data + '/registerdata_fold_' + str(fold) + '.npy')

    new_patients = np.load(folder_new_data + '/patients_fold_' + str(fold) + '.npy')
    new_labels = np.load(folder_new_data + '/labels_fold_' + str(fold) + '.npy')
    new_registerdata = np.load(folder_new_data + '/registerdata_fold_' + str(fold) + '.npy')

    print("Original patients, fold", fold, ":", original_patients)
    print("Original labels, fold", fold, ":", original_labels)

    print("New patients, fold", fold, ":", new_patients)
    print("New labels, fold", fold, ":", new_labels)   


def remove_duplicates_for_registerdata(path_register, path_label):

    folder = 'numpys_for_crossval_registerdatamodel/'
    tot_num_patients = 0

    for i in range(0, len(path_register)):

        original_register = np.load(path_register[i])
        original_labels = np.load(path_label[i])

        patient_numbers = original_register[:,-1]
        print(patient_numbers)
        print("len(patients_numbers)", np.size(patient_numbers))

        index_to_remove = []
        for j in range(1, len(patient_numbers)):
            if patient_numbers[j] == patient_numbers[j-1]:
                index_to_remove.append(j)

        patient_numbers_new = np.delete(patient_numbers, index_to_remove)
        new_labels = np.delete(original_labels, index_to_remove)
        new_registerdata = np.delete(original_register, index_to_remove, axis = 0)

        tot_num_patients += np.size(patient_numbers_new)

        numpy_labels_title = folder + 'labels_fold_'+  str(i+1) + '.npy'
        np.save(numpy_labels_title, new_labels)

        numpy_registerdata_title = folder + 'registerdata_fold_'+  str(i+1) + '.npy'
        np.save(numpy_registerdata_title, new_registerdata)

        numpy_patients_title = folder + 'patients_fold_'+  str(i+1) + '.npy'
        np.save(numpy_patients_title, patient_numbers_new)

        print('Saved numpy arrays!')

    print("Total number of patients-------------", tot_num_patients)

def extract_registerdata_for_patients_with_images(patients_with_im_path, registerdata_file="registerdata_all.csv", filename_new_csv="registerdata_for_images.csv"):

    patients_with_im = np.load(patients_with_im_path, allow_pickle=True)
    registerdata = pd.read_csv(registerdata_file)
    registerdata_pat = registerdata.values[:,-2]
    registerdata_pat_lst = registerdata_pat.tolist()
    patients_with_im_lst = patients_with_im.tolist()
    header = registerdata.columns
    header = list(header)

    counter = 0

    if os.stat(filename_new_csv).st_size != 0:
        f = open(filename_new_csv,'w+')
        f.close()

    temp_lst = []

    with open(filename_new_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        for patient_num in patients_with_im_lst:
            index = registerdata_pat_lst.index(patient_num)
            registerdata_for_a_patient = registerdata.values[index, :]
            temp_lst.append(registerdata_for_a_patient.tolist())
            counter = counter + 1
        writer.writerow(header)
        for row in temp_lst:
            writer.writerow(row)

        file.close()

def load_registerdata_age_sex_fold(numpy_filename='numpys_for_crossval_registerdatamodel/registerdata_fold_1.npy', csv_file_path='registerdata_all.csv'):
    csv_file = pd.read_csv(csv_file_path)
    header = csv_file.columns
    header_lst = list(header)

    idx_age = header_lst.index('age')
    idx_sex = header_lst.index('sex')

    registerdata_fold = np.load(numpy_filename)
    new_registerdata = registerdata_fold[:,idx_sex:(idx_age+1)]

    return new_registerdata

def load_registerdata_7_binary_fold(numpy_filename='numpys_for_crossval_registerdatamodel/registerdata_fold_1.npy', csv_file_path='registerdata_all.csv'):
    csv_file = pd.read_csv(csv_file_path)
    header = csv_file.columns
    header_lst = list(header)

    # Some variable names in Swedish
    idx_age = header_lst.index('age')
    idx_sex = header_lst.index('sex')
    idx_osteopor = header_lst.index('osteopor')
    idx_ra = header_lst.index('ra')
    idx_kort = header_lst.index('kort')
    idx_proton = header_lst.index('proton')
    idx_bisfo = header_lst.index('bisfo')

    registerdata_fold = np.load(numpy_filename)
    rows = np.shape(registerdata_fold)[0]
    new_registerdata = np.zeros((rows,7))

    new_registerdata[:,0] = registerdata_fold[:, idx_age]
    new_registerdata[:,1] = registerdata_fold[:, idx_sex]
    new_registerdata[:,2] = registerdata_fold[:, idx_osteopor]
    new_registerdata[:,3] = registerdata_fold[:, idx_ra]
    new_registerdata[:,4] = registerdata_fold[:, idx_kort]
    new_registerdata[:,5] = registerdata_fold[:, idx_proton]
    new_registerdata[:,6] = registerdata_fold[:, idx_bisfo]

    return new_registerdata

def load_registerdata_7_nonbinary_fold(numpy_filename='numpys_for_crossval_registerdatamodel/registerdata_fold_1.npy', csv_file_path='registerdata_all.csv'):
    csv_file = pd.read_csv(csv_file_path)
    header = csv_file.columns
    header_lst = list(header)

    # Some variable names in Swedish
    idx_age = header_lst.index('age')
    idx_sex = header_lst.index('sex')
    idx_osteopor = header_lst.index('osteopor_d')
    idx_ra = header_lst.index('ra_d')
    idx_kort = header_lst.index('kort')
    idx_proton = header_lst.index('proton')
    idx_bisfo = header_lst.index('bisfo_d')

    registerdata_fold = np.load(numpy_filename)
    rows = np.shape(registerdata_fold)[0]
    new_registerdata = np.zeros((rows,7))

    new_registerdata[:,0] = registerdata_fold[:, idx_age]
    new_registerdata[:,1] = registerdata_fold[:, idx_sex]
    new_registerdata[:,2] = registerdata_fold[:, idx_osteopor]
    new_registerdata[:,3] = registerdata_fold[:, idx_ra]
    new_registerdata[:,4] = registerdata_fold[:, idx_kort]
    new_registerdata[:,5] = registerdata_fold[:, idx_proton]
    new_registerdata[:,6] = registerdata_fold[:, idx_bisfo]

    return new_registerdata

def load_registerdata_all_binary_fold(numpy_filename='numpys_for_crossval_registerdatamodel/registerdata_fold_1.npy', csv_file_path='registerdata_all.csv'):
    csv_file = pd.read_csv(csv_file_path)
    header = csv_file.columns
    header_lst = list(header)

    idx_lst = []

    # Some variable names in Swedish
    idx_age = header_lst.index('age')
    idx_lst.append(idx_age)
    idx_sex = header_lst.index('sex')
    idx_lst.append(idx_sex)
    idx_blod = header_lst.index('blod')
    idx_lst.append(idx_blod)
    idx_cervsl = header_lst.index('cervsl')
    idx_lst.append(idx_cervsl)
    idx_cvd = header_lst.index('cvd')
    idx_lst.append(idx_cvd)
    idx_endokrin = header_lst.index('endokrin')
    idx_lst.append(idx_endokrin)
    idx_fraktsl = header_lst.index('fraktsl')
    idx_lst.append(idx_fraktsl)
    idx_hipsl = header_lst.index('hipsl')
    idx_lst.append(idx_hipsl)
    idx_hud = header_lst.index('hud')
    idx_lst.append(idx_hud)
    idx_infekt = header_lst.index('infekt')
    idx_lst.append(idx_infekt)
    idx_malign = header_lst.index('malign')
    idx_lst.append(idx_malign)
    idx_muskskel = header_lst.index('muskskel')
    idx_lst.append(idx_muskskel)
    idx_nerv = header_lst.index('nerv')
    idx_lst.append(idx_nerv)
    idx_psyke = header_lst.index('psyke')
    idx_lst.append(idx_psyke)
    idx_resp = header_lst.index('resp')
    idx_lst.append(idx_resp)
    idx_trochsl = header_lst.index('trochsl')
    idx_lst.append(idx_trochsl)
    idx_urin = header_lst.index('urin')
    idx_lst.append(idx_urin)
    idx_ihd = header_lst.index('ihd')
    idx_lst.append(idx_ihd)
    idx_kardiomy = header_lst.index('kardiomy')
    idx_lst.append(idx_kardiomy)
    idx_hjartsv = header_lst.index('hjartsv')
    idx_lst.append(idx_hjartsv)
    idx_cvl = header_lst.index('cvl')
    idx_lst.append(idx_cvl)
    idx_atherosc = header_lst.index('atherosc')
    idx_lst.append(idx_atherosc)
    idx_hyperton = header_lst.index('hyperton')
    idx_lst.append(idx_hyperton)
    idx_mage = header_lst.index('mage')
    idx_lst.append(idx_mage)
    idx_diabetes = header_lst.index('diabetes')
    idx_lst.append(idx_diabetes)
    idx_lipid = header_lst.index('lipid')
    idx_lst.append(idx_lipid)
    idx_ami = header_lst.index('ami')
    idx_lst.append(idx_ami)
    idx_cvlhem = header_lst.index('cvlhem')
    idx_lst.append(idx_cvlhem)
    idx_cvlisch = header_lst.index('cvlisch')
    idx_lst.append(idx_cvlisch)
    idx_opfraktsl = header_lst.index('opfraktsl')
    idx_lst.append(idx_opfraktsl)
    idx_kotasl = header_lst.index('kotasl')
    idx_lst.append(idx_kotasl)
    idx_gica = header_lst.index('gica')
    idx_lst.append(idx_gica)
    idx_gica2 = header_lst.index('gica2')
    idx_lst.append(idx_gica2)
    idx_galla = header_lst.index('galla')
    idx_lst.append(idx_galla)
    idx_pancreas = header_lst.index('pancreas')
    idx_lst.append(idx_pancreas)
    idx_ra = header_lst.index('ra')
    idx_lst.append(idx_ra)
    idx_osteopor = header_lst.index('osteopor')
    idx_lst.append(idx_osteopor)
    idx_index = header_lst.index('index')
    idx_lst.append(idx_index)
    idx_kort = header_lst.index('kort')
    idx_lst.append(idx_kort)
    idx_proton = header_lst.index('proton')
    idx_lst.append(idx_proton)
    idx_hrt = header_lst.index('hrt')
    idx_lst.append(idx_hrt)
    idx_serm = header_lst.index('serm')
    idx_lst.append(idx_serm)
    idx_antiep = header_lst.index('antiep')
    idx_lst.append(idx_antiep)
    idx_antidepr = header_lst.index('antidepr')
    idx_lst.append(idx_antidepr)
    idx_bisfo = header_lst.index('bisfo')
    idx_lst.append(idx_bisfo)

    registerdata_fold = np.load(numpy_filename)
    rows = np.shape(registerdata_fold)[0]
    new_registerdata = np.zeros((rows,len(idx_lst)))

    col = 0
    for index in idx_lst:
        new_registerdata[:, col] = registerdata_fold[:, index]
        col += 1

    return new_registerdata

def load_registerdata_all_nonbinary_fold(numpy_filename='numpys_for_crossval_registerdatamodel/registerdata_fold_1.npy', csv_file_path='registerdata_all.csv'):
    csv_file = pd.read_csv(csv_file_path)
    header = csv_file.columns
    header_lst = list(header)

    idx_lst = []

    # Some variable names in Swedish
    idx_age = header_lst.index('age')
    idx_lst.append(idx_age)
    idx_sex = header_lst.index('sex')
    idx_lst.append(idx_sex)
    idx_blod = header_lst.index('blod_d')
    idx_lst.append(idx_blod)
    idx_cervsl = header_lst.index('cervsl_d')
    idx_lst.append(idx_cervsl)
    idx_cvd = header_lst.index('cvd_d')
    idx_lst.append(idx_cvd)
    idx_endokrin = header_lst.index('endokrin_d')
    idx_lst.append(idx_endokrin)
    idx_fraktsl = header_lst.index('fraktsl_d')
    idx_lst.append(idx_fraktsl)
    idx_hipsl = header_lst.index('hipsl_d')
    idx_lst.append(idx_hipsl)
    idx_hud = header_lst.index('hud_d')
    idx_lst.append(idx_hud)
    idx_infekt = header_lst.index('infekt_d')
    idx_lst.append(idx_infekt)
    idx_malign = header_lst.index('malign_d')
    idx_lst.append(idx_malign)
    idx_muskskel = header_lst.index('muskskel_d')
    idx_lst.append(idx_muskskel)
    idx_nerv = header_lst.index('nerv_d')
    idx_lst.append(idx_nerv)
    idx_psyke = header_lst.index('psyke_d')
    idx_lst.append(idx_psyke)
    idx_resp = header_lst.index('resp_d')
    idx_lst.append(idx_resp)
    idx_trochsl = header_lst.index('trochsl_d')
    idx_lst.append(idx_trochsl)
    idx_urin = header_lst.index('urin_d')
    idx_lst.append(idx_urin)
    idx_ihd = header_lst.index('ihd_d')
    idx_lst.append(idx_ihd)
    idx_kardiomy = header_lst.index('kardiomy_d')
    idx_lst.append(idx_kardiomy)
    idx_hjartsv = header_lst.index('hjartsv_d')
    idx_lst.append(idx_hjartsv)
    idx_cvl = header_lst.index('cvl_d')
    idx_lst.append(idx_cvl)
    idx_atherosc = header_lst.index('atherosc_d')
    idx_lst.append(idx_atherosc)
    idx_hyperton = header_lst.index('hyperton_d')
    idx_lst.append(idx_hyperton)
    idx_mage = header_lst.index('mage_d')
    idx_lst.append(idx_mage)
    idx_diabetes = header_lst.index('diabetes_d')
    idx_lst.append(idx_diabetes)
    idx_lipid = header_lst.index('lipid_d')
    idx_lst.append(idx_lipid)
    idx_ami = header_lst.index('ami_d')
    idx_lst.append(idx_ami)
    idx_cvlhem = header_lst.index('cvlhem_d')
    idx_lst.append(idx_cvlhem)
    idx_cvlisch = header_lst.index('cvlisch_d')
    idx_lst.append(idx_cvlisch)
    idx_opfraktsl = header_lst.index('opfraktsl_d')
    idx_lst.append(idx_opfraktsl)
    idx_kotasl = header_lst.index('kotasl_d')
    idx_lst.append(idx_kotasl)
    idx_gica = header_lst.index('gica_d')
    idx_lst.append(idx_gica)
    idx_gica2 = header_lst.index('gica2_d')
    idx_lst.append(idx_gica2)
    idx_galla = header_lst.index('galla_d')
    idx_lst.append(idx_galla)
    idx_pancreas = header_lst.index('pancreas_d')
    idx_lst.append(idx_pancreas)
    idx_ra = header_lst.index('ra_d')
    idx_lst.append(idx_ra)
    idx_osteopor = header_lst.index('osteopor_d')
    idx_lst.append(idx_osteopor)
    idx_index = header_lst.index('index')
    idx_lst.append(idx_index)
    idx_kort = header_lst.index('kort')
    idx_lst.append(idx_kort)
    idx_proton = header_lst.index('proton')
    idx_lst.append(idx_proton)
    idx_hrt = header_lst.index('hrt')
    idx_lst.append(idx_hrt)
    idx_serm = header_lst.index('serm')
    idx_lst.append(idx_serm)
    idx_antiep = header_lst.index('antiep')
    idx_lst.append(idx_antiep)
    idx_antidepr = header_lst.index('antidepr')
    idx_lst.append(idx_antidepr)
    idx_bisfo = header_lst.index('bisfo_d')
    idx_lst.append(idx_bisfo)

    registerdata_fold = np.load(numpy_filename)
    rows = np.shape(registerdata_fold)[0]
    new_registerdata = np.zeros((rows,len(idx_lst)))

    col = 0
    for index in idx_lst:
        new_registerdata[:, col] = registerdata_fold[:, index]
        col += 1

    return new_registerdata

def crossvalidation_registerdata(num_epochs=50000, model_num='1_age_sex', lr = 1e-4, save_csv = False, folder='csv_2', variable_selection_func = None, path_to_place_plots_in='Plots_for_images', repetition=1):

    if save_csv == True:
        #Clear csv file
        path_plot = folder + '/results_cross_val_model' + str(model_num) + '_repetition_' + str(repetition) + '.csv'
        f = open(path_plot,'w+')
        f.close()
        path_mean = folder + '/roc_mean_model' + str(model_num) + '_repetition_' + str(repetition) + '.csv'
        f = open(path_mean,'w+')
        f.close()

    folder_for_numpys = 'numpys_for_crossval_registerdatamodel/'
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
    labels_lst_np = []

    for i in range(0, len(registerdata_lst_paths)):
        if i == (len(registerdata_lst_paths) - 1):
            registerdata_test = variable_selection_func(registerdata_lst_paths[i])
            labels_test = np.load(labels_lst_paths[i])
        else:
            registerdata_lst_np.append(variable_selection_func(registerdata_lst_paths[i]))
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

    for i in range(0, len(registerdata_lst_np)):
        copy_folds_registerdata = registerdata_lst_np.copy()
        copy_folds_labels = labels_lst_np.copy()

        registerdata_valid = copy_folds_registerdata[i]
        labels_valid = copy_folds_labels[i]

        del copy_folds_registerdata[i]
        del copy_folds_labels[i]

        registerdata_train = np.vstack(copy_folds_registerdata)
        labels_train = np.hstack(copy_folds_labels)

        labels_train_non_zero = np.count_nonzero(labels_train)
        labels_test_non_zero = np.count_nonzero(labels_test)
        labels_valid_non_zero = np.count_nonzero(labels_valid)

        print('% AFF train: ', labels_train_non_zero/len(labels_train))
        print('% AFF test: ', labels_test_non_zero/len(labels_test))
        print('% AFF valid: ', labels_valid_non_zero/len(labels_valid))

        model, optimizer, callback = MakeRegisterDataModel(registerdata_train, labels_train, model_num=model_number, num_epochs=num_epochs, fold=i, learning_rate=lr)

        history = train_registerdata_model(model, registerdata_train, labels_train, registerdata_valid, labels_valid, callback, class_weights, optimizer=optimizer, num_epochs=num_epochs)

        model.summary()
        labels_pred = model.predict(registerdata_test)

        mean_tpr, mean_fpr, lst_acc, lst_sens, lst_spec, lst_mcc = Evaluation_CrossVal_register(history, registerdata_test, labels_test, labels_pred, model, save_csv, i, model_num, mean_tpr, mean_fpr, lst_acc, lst_sens, lst_spec, lst_mcc, folder, repetition)
        plt_acc(path_csv=path_plot, fold=i, path_save=path_to_place_plots_in)
        plt_loss(path_csv=path_plot, fold=i, path_save=path_to_place_plots_in)
        keras.backend.clear_session()

    n_folds = 6

    mean_tpr /= (n_folds - 1)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    mean_acc = sum(lst_acc) / (n_folds - 1)
    std_acc = np.std(np.asarray(lst_acc))
    mean_sens = sum(lst_sens) / (n_folds - 1)
    std_sens = np.std(np.asarray(lst_sens))

    array_spec = np.asarray(lst_spec)
    mean_spec = np.mean(array_spec)
    std_spec = np.std(array_spec)

    array_mcc = np.asarray(lst_mcc)
    mean_mcc = np.mean(array_mcc)
    std_mcc = np.std(array_mcc)

    # Save mean roc
    if save_csv:
        path = folder + '/roc_mean_model' + str(model_num) + '_repetition_' + str(repetition) + '.csv'
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["mean_tpr", "mean_fpr", "mean_auc", "mean_acc", "std_acc", "mean_sens", "std_sens", "mean_spec", "std_spec", "mean_mcc", "std_mcc"])
            writer.writerow([mean_tpr, mean_fpr, mean_auc, mean_acc, std_acc, mean_sens, std_sens, mean_spec, std_spec, mean_mcc, std_mcc])
            file.close()

def Evaluation_CrossVal_register(history, X_test, Y_test, Y_pred, model, save_csv, iteration, model_num, mean_tpr, mean_fpr, lst_acc, lst_sens, lst_spec, lst_mcc, folder, repetition):
    path = folder + '/results_cross_val_model' + str(model_num) + '_repetition_' + str(repetition) + '.csv'

    if os.stat(path).st_size == 0:
        model_nr = 1
        mode = 'w'
    else:
        mode = 'a'
        with open(path, "r", encoding="utf-8", errors="ignore") as scraped:
            model_nr = iteration + 1

    epochs = len(history.history['binary_accuracy'])
    x_axis = range(1, epochs + 1)
    results = model.evaluate(X_test, Y_test)
    loss = results[0]
    bin_acc = results[1]
    auc = results[2]
    print('results (loss, binary accuracy, auc): ', results)

    # Sensitivity
    TP = keras.metrics.TruePositives()
    TP.update_state(Y_test, Y_pred)
    TP = TP.result().numpy()
    FN = keras.metrics.FalseNegatives()
    FN.update_state(Y_test, Y_pred)
    FN = FN.result().numpy()
    sensitivity = TP / (TP + FN)

    print('TP: ', TP)
    print('FN: ', FN)
    print('Sensitivity: ', sensitivity)

    #Specificity
    TN = keras.metrics.TrueNegatives()
    TN.update_state(Y_test, Y_pred)
    TN = TN.result().numpy()
    FP = keras.metrics.FalsePositives()
    FP.update_state(Y_test, Y_pred)
    FP = FP.result().numpy()
    specificity = TN / (TN + FP)

    lst_spec.append(specificity)

    # MCC
    MCC = matthews_corrcoef(Y_test, Y_pred.round())
    lst_mcc.append(MCC)
    print('MCC: ', MCC)

    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = metrics.auc(fpr,tpr)
    lst_acc.append(bin_acc)
    lst_sens.append(sensitivity)

    print('Roc auc: ', roc_auc)

    if save_csv:
        with open(path, mode, newline='') as file:
            writer = csv.writer(file)
            if os.stat(path).st_size == 0:
                writer.writerow(["Model", "BinAcc", "ValBinAcc", "Loss", "ValLoss", "tpr_temp", "fpr_temp", "Roc AUC", "Epochs", "Sensitivity", "BinaryAccuracy", "Specificity", "MCC"])
            writer.writerow([model_nr, history.history['binary_accuracy'], history.history['val_binary_accuracy'], history.history['loss'], history.history['val_loss'], tpr, fpr, roc_auc, epochs, sensitivity, bin_acc, specificity, MCC])
            file.close()
    return mean_tpr, mean_fpr, lst_acc, lst_sens, lst_spec, lst_mcc


def MakeRegisterDataModel(X_train, Y_train, model_num, activation='relu', loss='binary_crossentropy', num_epochs=1000, fold=0, learning_rate=1e-4):
    num_dim = X_train.shape[1]
    model_num = int(model_num)

    model = Sequential()

    if model_num == 1:
        model.add(Dense(50, input_dim=num_dim, activation=activation))
        model.add(Dropout(0.25))
    if model_num == 2:
        model.add(Dense(512, input_dim=num_dim, activation=activation))
        model.add(Dropout(0.25))
        model.add(Dense(256, activation=activation))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation = activation))
        model.add(Dropout(0.25))
    if model_num == 3:
        model.add(Dense(1024, input_dim=num_dim, activation=activation))
        model.add(Dense(512, activation=activation))
        model.add(Dropout(0.25))
    if model_num == 4:
        model.add(Dense(256, input_dim=num_dim, activation=activation))
        model.add(Dense(256, activation=activation))
        model.add(Dense(128, activation=activation))
        model.add(Dense(64, activation=activation))
        model.add(Dense(32, activation=activation))
    model.add(Dense(1, activation='sigmoid'))

    # Configure model
    metric = [BinaryAccuracy(),
                keras.metrics.AUC(name='auc', curve='ROC'),
                ]
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=metric)

    # Early Stopping
    callback_metric = EarlyStopping(monitor='val_loss', verbose=2, patience=100, restore_best_weights=True)

    # Save best model
    #path_save = 'RegisterDataModels_age_sex/Model_fold_' + str(fold + 1) + '_cp.ckpt'
    path_save = 'RegisterDataModels/Model_fold_' + str(fold + 1) + '_cp.ckpt'
    callback_save = ModelCheckpoint(path_save, monitor='val_loss', verbose=0, save_best_only=True, mode='min', save_weights_only=True)

    callback = [callback_metric, callback_save]

    return model, optimizer, callback

def train_registerdata_model(model, X_train, Y_train, X_valid, Y_valid, callback, class_weights, activation='relu', optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', num_epochs=1000):
    history = model.fit(X_train, Y_train, epochs=num_epochs, verbose=1, validation_data=(X_valid,Y_valid), shuffle=True, callbacks=callback, class_weight=class_weights)
    return history
