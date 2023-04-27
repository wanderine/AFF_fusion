import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from load_images import load_saved_data, load_saved_data_not_split
#from test import Evaluation_no_save_CSV
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from load_images import get_im_and_labels_as_array, normalize_im
from read_csv_for_plts import make_plot_lst
from predict_patient import pred_patient
from read_csv_for_plts import plt_acc, plt_loss
import numpy as np
import math
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from numpy import interp
import csv
import os
import matplotlib.pyplot as plt

print("Num GPUs available:", len(tf.config.list_physical_devices("GPU")))

gpus = tf.config.list_physical_devices("GPU")
print(gpus)

if gpus:
    print("GPU detected")
    tf.config.experimental.set_memory_growth(gpus[0], True) 

def scheduler(epoch, lr):
    return lr
"""
    if epoch > 50 and lr > 1e-5:
        return lr * tf.math.exp(-0.05)
    else:
        return lr
"""

def build_im_model(learning_rate=1e-4, patience=50, aug_setting=3, fold_num=5, img_res = 224, model_setting = 0):

    #base_model = ResNet50(include_top=False, weights='imagenet', input_shape = (img_res, img_res, 3))
    base_model = ResNet50(include_top=False, weights='imagenetweights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape = (img_res, img_res, 3))

    #base_model.summary()
    #create augmentation layers
    flip_bool = False
    rot_factor = 0.3
    trans_factor = 0.1
    zoom_factor = 0.1
    contrast_factor = 0.1

    if aug_setting == 3:
        flip_bool = True
        contrast_factor = 0.3 

    if aug_setting == 4:
        flip_bool = True
        rot_factor = 0.5
        contrast_factor = 0.3

    if aug_setting == 5:
        rot_factor = 0.5
        contrast_factor = 0.3

    if aug_setting == 6:
        rot_factor = 1.0
        contrast_factor = 0.3


    if flip_bool:

        data_augmentation = keras.Sequential(
        [keras.layers.RandomFlip(mode='horizontal_and_vertical'),
        keras.layers.RandomRotation(factor=rot_factor, fill_mode='constant', fill_value=0.0),
        keras.layers.RandomTranslation(height_factor=trans_factor, width_factor=trans_factor, fill_mode='constant', fill_value = 0.0),
        keras.layers.RandomZoom(height_factor=zoom_factor, width_factor =zoom_factor, fill_mode='constant', fill_value=0.0),
        keras.layers.RandomContrast(factor=contrast_factor),
        ]
        )

    else:
        data_augmentation = keras.Sequential(
        [keras.layers.RandomRotation(factor=rot_factor, fill_mode='constant', fill_value=0.0),
        keras.layers.RandomTranslation(height_factor=trans_factor, width_factor=trans_factor, fill_mode='constant', fill_value = 0.0),
        keras.layers.RandomZoom(height_factor=zoom_factor, width_factor =zoom_factor, fill_mode='constant', fill_value=0.0),
        keras.layers.RandomContrast(factor=contrast_factor),
        ]
        )

    if img_res == 224:
        train_layers_index = 38

        print("Name of layer:", base_model.layers[train_layers_index].name)

        print("Name of next layer:", base_model.layers[train_layers_index+1].name)

        print("\n Total number of layers is:", len(base_model.layers))

        for layer in base_model.layers[:train_layers_index]:
            layer.trainable = False
    else:
        print("------------------------")
        print("All layers are trainable")
        print("------------------------")

    #create new model on top
    inputs = keras.Input(shape=(img_res, img_res, 3))

    #apply data augmentation
    aug_inputs = data_augmentation(inputs)

    #aug_inputs = inputs

    x = base_model(aug_inputs)


    #x = keras.layers.GlobalAveragePooling2D()(x)
    #x = keras.layers.Conv2D(2048, 5, strides = (2,2), activation = 'relu')(x)

    if model_setting == 0:
        x = keras.layers.GlobalMaxPool2D()(x)
        x = keras.layers.Dense(units = 1024, activation = "relu")(x)
        x = keras.layers.Dropout(rate = 0.5)(x)

        x = keras.layers.Dense(units = 512, activation = "relu")(x)
        x = keras.layers.Dropout(rate = 0.5)(x)

        x = keras.layers.Dense(units = 256, activation = "relu")(x)
        x = keras.layers.Dropout(rate = 0.5)(x)

        print("--------------------Same fully connected layer as  before-------------------")

    elif model_setting == 1:
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.25)(x)

        print("---------------------Only AvgPool and Dropout---------------------")
    else:
        print("Something wrong with model top...")

    output = keras.layers.Dense(units = 1, activation = "sigmoid")(x)

    model = keras.Model(inputs, output)

    metric = [keras.metrics.BinaryAccuracy(), keras.metrics.AUC(name='auc', curve="ROC"),]

    opt = keras.optimizers.Adam(learning_rate)

    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = metric)

    callback_metric = EarlyStopping(monitor='val_loss', patience=patience, mode = "min", restore_best_weights=True, verbose = 2)

    #We also need a callback for saving the best_model
    #path_save = 'image_model_patients_weights/fold_' + str(fold_num) + '.ckpt'
    #callback_save = ModelCheckpoint(path_save, monitor='val_loss', verbose = 2, save_best_only=True, mode='min', save_weights_only = True)

    #Metric for learning-rate-LearningRateScheduler
    callback_lr = LearningRateScheduler(scheduler, verbose = 1)

    callback = [callback_metric, callback_lr]

    model.summary()


    return model, opt, callback#, metric


def train_im_model(model, X_train, Y_train, X_valid, Y_valid, callback, class_weights, activation='relu', optimizer=None, loss='binary_crossentropy', num_epochs=100):
    history = model.fit(X_train, Y_train, epochs=num_epochs, validation_data=(X_valid,Y_valid), shuffle=True, callbacks=callback, class_weight=class_weights, verbose=2,)
    return history

def Evaluation_CrossVal(history, X_test, Y_test, Y_pred, model, save_csv, iteration, model_num, mean_tpr, mean_fpr, lst_acc, lst_sens, lst_spec, folder='csv_images'):
    #csv for cross validation
    path =  folder + '/results_cross_val_model' + str(model_num) + '.csv'

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

    lst_acc.append(bin_acc)
    lst_sens.append(sensitivity)

    print('TP: ', TP)
    print('FN: ', FN)
    print('Sensitivity: ', sensitivity)

    #Specificity
    TN = keras.metrics.TrueNegatives()
    TN.update_state(Y_test, Y_pred)
    TN.result().numpy()
    FP = keras.metrics.FalsePositives()
    FP.update_state(Y_test, Y_pred)
    FP = FP.result().numpy()
    specificity = TN / (TN + FP)

    lst_spec.append(specificity)

    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = metrics.auc(fpr,tpr)

    print('Roc auc: ', roc_auc)

    if save_csv:
        with open(path, mode, newline='') as file:
            writer = csv.writer(file)
            if os.stat(path).st_size == 0:
                writer.writerow(["Fold", "BinAcc", "ValBinAcc", "Loss", "ValLoss", "tpr_temp", "fpr_temp", "Roc AUC", "Epochs", "Sensitivity", "BinaryAccuracy", "Specificity"])
            writer.writerow([model_nr, history.history['binary_accuracy'], history.history['val_binary_accuracy'], history.history['loss'], history.history['val_loss'], tpr, fpr, roc_auc, epochs, sensitivity, bin_acc, specificity])
            file.close()

    return mean_tpr, mean_fpr, lst_acc, lst_sens, lst_spec

def Evaluation_CrossVal_patients(history, Y_test, Y_pred, model, save_csv, iteration, model_num, mean_tpr, mean_fpr, lst_acc, lst_sens, lst_spec, lst_mcc, folder, repetition):
    #csv for cross validation
    path =  folder + '/results_cross_val_model' + str(model_num) + '_repetition_' + str(repetition) + '.csv'

    if os.stat(path).st_size == 0:
        model_nr = 1
        mode = 'w'
    else:
        mode = 'a'
        with open(path, "r", encoding="utf-8", errors="ignore") as scraped:
            model_nr = iteration + 1

    epochs = len(history.history['binary_accuracy'])
    x_axis = range(1, epochs + 1)

    #print('Y_pred: ', Y_pred)
    round_Y_pred = Y_pred.round()
    #print('round_Y_pred: ', round_Y_pred)
    bin_acc = metrics.accuracy_score(Y_test, round_Y_pred)
    #print('bin_acc: ', bin_acc)


    # Sensitivity
    TP = keras.metrics.TruePositives()
    TP.update_state(Y_test, Y_pred)
    TP = TP.result().numpy()
    FN = keras.metrics.FalseNegatives()
    FN.update_state(Y_test, Y_pred)
    FN = FN.result().numpy()
    sensitivity = TP / (TP + FN)

    lst_acc.append(bin_acc)
    lst_sens.append(sensitivity)

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

    #Naive classifier, AUC

    """
    naive_pred = np.zeros(np.size(Y_test))
    print("Test labels ---------------------", Y_test, "\n")
    print("Naive prediictions ----------------------", naive_pred, "\n")
    naive_fpr, naive_tpr, naive_thresholds = metrics.roc_curve(Y_test, naive_pred)
    print("naive fpr ---------------------", naive_fpr, "\n")
    print("naive tpr ----------------------", naive_tpr, "\n")
    naive_auc = metrics.auc(naive_fpr, naive_tpr)
    print("naive auc ----------------------", naive_auc, "\n")

    lst_naive_auc.append(naive_auc)

    """


    print('Roc auc: ', roc_auc)

    if save_csv:
        with open(path, mode, newline='') as file:
            writer = csv.writer(file)
            if os.stat(path).st_size == 0:
                writer.writerow(["Fold", "BinAcc", "ValBinAcc", "Loss", "ValLoss", "tpr_temp", "fpr_temp", "Roc AUC", "Epochs", "Sensitivity", "BinaryAccuracy", "Specificity", "MCC"])
            writer.writerow([model_nr, history.history['binary_accuracy'], history.history['val_binary_accuracy'], history.history['loss'], history.history['val_loss'], tpr, fpr, roc_auc, epochs, sensitivity, bin_acc, specificity, MCC])
            file.close()

    return mean_tpr, mean_fpr, lst_acc, lst_sens, lst_spec, lst_mcc


def crossval_images(shuffled_X, shuffled_Y, n_folds=5, num_epochs=100, save_csv=False, model_num=1, learning_rate=1e-5, patience=50, aug_setting=None):
    n_folds = n_folds + 1
    num_patients_per_fold = math.ceil(len(shuffled_X) / n_folds)
    num_features = shuffled_X.shape[1]
    folds_X = []
    folds_Y = []
    X_test = np.array([])
    Y_test = np.array([])
    print('num_patients_per_fold: ', num_patients_per_fold)

    if save_csv == True:
        #Clear csv file
        path_plot = 'csv_images/results_cross_val_model' + str(model_num) + '.csv'
        f = open(path_plot,'w+')
        f.close()
        path_mean = 'csv_images/roc_mean_model' + str(model_num) + '.csv'
        f = open(path_mean,'w+')
        f.close()

    for i in range(0,n_folds):
        index1 = i*num_patients_per_fold
        index2 = (i+1)*num_patients_per_fold
        temp_fold_X = shuffled_X[index1:index2]
        temp_fold_Y = shuffled_Y[index1:index2]

        if i != n_folds - 1:
            folds_X.append(temp_fold_X)
            folds_Y.append(temp_fold_Y)
            print("NOT the last fold \n")
        else:
            X_test = temp_fold_X
            Y_test = temp_fold_Y
            print("The last fold \n")

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    lst_acc = []
    lst_sens =  []


    n_samples = np.vstack(folds_X).shape[0]
    n_classes = len(set(np.hstack(folds_Y)))
    class_weights = n_samples / (n_classes * np.bincount(np.hstack(folds_Y)))
    class_weights = {0: class_weights[0], 1: class_weights[1]}
    print("class weights-------------", class_weights, "\n")

    for i in range(0, n_folds-1):
        copy_folds_X = folds_X.copy()
        copy_folds_Y = folds_Y.copy()

        X_valid = copy_folds_X[i]
        Y_valid = copy_folds_Y[i]
        del copy_folds_X[i]
        del copy_folds_Y[i]

        X_train = np.vstack(copy_folds_X)
        Y_train = np.hstack(copy_folds_Y)

        Y_train_non_zero = np.count_nonzero(Y_train)
        Y_test_non_zero = np.count_nonzero(Y_test)
        Y_valid_non_zero = np.count_nonzero(Y_valid)


        print("-----------------------------MODEL number------------------ ", model_num)
        print("----------------------------AUGMENT setting--------------------", aug_setting)
        print("-----------------------------FOLD number------------------ ", i+1)

        #print('% AFF train: ', Y_train_non_zero/len(Y_train))
        #print('% AFF test: ', Y_test_non_zero/len(Y_test))
        #print('% AFF valid: ', Y_valid_non_zero/len(Y_valid))

        #history = train_im_model(model, shuffled_Y, train_fraq)

        model, opt, callback  = build_im_model(learning_rate, patience, aug_setting, fold_num=(i+1))

        history = train_im_model(model, X_train, Y_train, X_valid, Y_valid, callback, class_weights, optimizer=opt, num_epochs=num_epochs)

        Y_pred = model.predict(X_test)

        mean_tpr, mean_fpr, lst_acc, lst_sens = Evaluation_CrossVal(history, X_test, Y_test, Y_pred, model, save_csv, i, model_num, mean_tpr, mean_fpr, lst_acc, lst_sens)

        plt_acc(path_csv=path_plot, fold=i, path_save='Plots_for_report_images')
        plt_loss(path_csv=path_plot, fold=i, path_save='Plots_for_report_images')

        keras.backend.clear_session()

    mean_tpr /= (n_folds - 1)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)

    #Calculate mean and std of accuracy and sensitivity, across all folds
    array_acc = np.asarray(lst_acc)
    array_sens = np.asarray(lst_sens)
    mean_acc = np.mean(array_acc)
    std_acc = np.std(array_acc)
    mean_sens = np.mean(array_sens)
    std_sens = np.std(array_sens)

    # Save mean roc
    if save_csv:
        path = 'csv_images/roc_mean_model' + str(model_num) + '.csv'
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([mean_tpr, mean_fpr, mean_auc, mean_acc, std_acc, mean_sens, std_sens])
            file.close()

def train_images_only(n_folds=5, num_epochs=100, save_csv=False, model_num=1, learning_rate=1e-5, patience=50, aug_setting=None, folder='csv_patients', img_res = 224, folder_for_numpys='numpys_for_crossval', path_to_save_model='image_model_weights', path_to_save_acc_loss='Plots_for_images', model_512_config = 0, repetition=1):
    n_folds = n_folds + 1
    if save_csv == True:
        #Clear csv file
        path_plot = folder + '/results_cross_val_model' + str(model_num) + '_repetition_' + str(repetition) + '.csv'
        f = open(path_plot,'w+')
        f.close()
        path_mean = folder + '/roc_mean_model' + str(model_num) + '_repetition_' + str(repetition) + '.csv'
        f = open(path_mean,'w+')
        f.close()

    im_path_fold1 = folder_for_numpys + '/im_fold_1.npy'
    im_path_fold2 = folder_for_numpys + '/im_fold_2.npy'
    im_path_fold3 = folder_for_numpys + '/im_fold_3.npy'
    im_path_fold4 = folder_for_numpys + '/im_fold_4.npy'
    im_path_fold5 = folder_for_numpys + '/im_fold_5.npy'
    im_path_fold6 = folder_for_numpys + '/im_fold_6.npy'

    im_lst_paths = []
    im_lst_paths.append(im_path_fold1)
    im_lst_paths.append(im_path_fold2)
    im_lst_paths.append(im_path_fold3)
    im_lst_paths.append(im_path_fold4)
    im_lst_paths.append(im_path_fold5)
    im_lst_paths.append(im_path_fold6)

    labels_path_fold1 = folder_for_numpys + '/labels_fold_1.npy'
    labels_path_fold2 = folder_for_numpys + '/labels_fold_2.npy'
    labels_path_fold3 = folder_for_numpys + '/labels_fold_3.npy'
    labels_path_fold4 = folder_for_numpys + '/labels_fold_4.npy'
    labels_path_fold5 = folder_for_numpys + '/labels_fold_5.npy'
    labels_path_fold6 = folder_for_numpys + '/labels_fold_6.npy'

    labels_lst_paths = []
    labels_lst_paths.append(labels_path_fold1)
    labels_lst_paths.append(labels_path_fold2)
    labels_lst_paths.append(labels_path_fold3)
    labels_lst_paths.append(labels_path_fold4)
    labels_lst_paths.append(labels_path_fold5)
    labels_lst_paths.append(labels_path_fold6)

    im_lst_np = []
    labels_lst_np = []

    for i in range(0, len(im_lst_paths)):
        if i == (len(im_lst_paths) - 1):
            im_test = np.load(im_lst_paths[i])
            labels_test = np.load(labels_lst_paths[i])
            patient_IDs_test = np.load(folder_for_numpys + '/patients_fold_6.npy')

        else:
            im_lst_np.append(np.load(im_lst_paths[i]))
            labels_lst_np.append(np.load(labels_lst_paths[i]))

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    lst_acc = []
    lst_sens = []
    lst_spec = []
    lst_mcc = []

    # Compensate for unbalanced dataset, via class weights
    n_samples = np.vstack(im_lst_np).shape[0]
    n_classes = len(set(np.hstack(labels_lst_np)))
    class_weights = n_samples / (n_classes * np.bincount(np.hstack(labels_lst_np)))
    class_weights = {0: class_weights[0], 1: class_weights[1]}
    print('class_weights: ', class_weights)

    for i in range(0, len(im_lst_np)):
        copy_folds_im = im_lst_np.copy()
        copy_folds_labels = labels_lst_np.copy()

        im_valid = copy_folds_im[i]
        labels_valid = copy_folds_labels[i]

        del copy_folds_im[i]
        del copy_folds_labels[i]

        im_train = np.vstack(copy_folds_im)
        labels_train = np.hstack(copy_folds_labels)

        labels_train_non_zero = np.count_nonzero(labels_train)
        labels_test_non_zero = np.count_nonzero(labels_test)
        labels_valid_non_zero = np.count_nonzero(labels_valid)

        print('% AFF train: ', labels_train_non_zero/len(labels_train))
        print('% AFF test: ', labels_test_non_zero/len(labels_test))
        print('% AFF valid: ', labels_valid_non_zero/len(labels_valid))

        print("-----------------------------MODEL number------------------ ", model_num)
        print("----------------------------AUGMENT setting--------------------", aug_setting)
        print("-----------------------------FOLD number------------------ ", i+1)

        model, opt, callback  = build_im_model(learning_rate, patience, aug_setting, fold_num=(i+1), img_res = img_res, model_setting = model_512_config)

        history = train_im_model(model, im_train, labels_train, im_valid, labels_valid, callback, class_weights, optimizer=opt, num_epochs=num_epochs)

        labels_pred = model.predict(im_test)

        new_labels_pred, new_patient_IDs_test, new_labels_test = pred_patient(labels_pred, patient_IDs_test, labels_test)

        #print("new_labels_pred-----------------", new_labels_pred, "\n")
        #print("new_patient_IDs_test----------------", new_patient_IDs_test, "\n")
        #print("new_labels_test--------------------", new_labels_test, "\n")

        mean_tpr, mean_fpr, lst_acc, lst_sens, lst_spec, lst_mcc = Evaluation_CrossVal_patients(history, new_labels_test, new_labels_pred, model, save_csv, i, model_num, mean_tpr, mean_fpr, lst_acc, lst_sens, lst_spec, lst_mcc, folder, repetition)

        model.save(path_to_save_model + '/fold_' + str(i) + '.h5')

        plt_acc(path_csv=path_plot, fold=i, path_save=path_to_save_acc_loss, repetition=repetition)
        plt_loss(path_csv=path_plot, fold=i, path_save=path_to_save_acc_loss, repetition=repetition)

        keras.backend.clear_session()

    mean_tpr /= (n_folds - 1)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)

    #Calculate mean and std of accuracy and sensitivity, across all folds
    array_acc = np.asarray(lst_acc)
    array_sens = np.asarray(lst_sens)
    array_mcc = np.asarray(lst_mcc)

    mean_acc = np.mean(array_acc)
    std_acc = np.std(array_acc)
    mean_sens = np.mean(array_sens)
    std_sens = np.std(array_sens)
    mean_mcc = np.mean(array_mcc)
    std_mcc = np.std(array_mcc)

    #Specificity
    array_spec = np.asarray(lst_spec)
    mean_spec = np.mean(array_spec)
    std_spec = np.std(array_spec)

    # Save mean roc
    if save_csv:
        path = folder + '/roc_mean_model' + str(model_num) + '_repetition_' + str(repetition) + '.csv'
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([mean_tpr, mean_fpr, mean_auc, mean_acc, std_acc, mean_sens, std_sens, mean_spec, std_spec, mean_mcc, std_mcc])
            file.close()




def crossval_patients(all_data, n_folds=5, num_epochs=100, save_csv=False, model_num=1, learning_rate=1e-5, patience=50, aug_setting=None, folder='csv_patients'):
    n_folds = n_folds + 1
    num_patients_per_fold = math.ceil(len(all_data) / n_folds)
    num_features = all_data.shape[1]
    folds_X = []
    folds_Y = []
    X_test = np.array([])
    Y_test = np.array([])
    print('len(all_data): ', len(all_data))
    print('num_patients_per_fold: ', num_patients_per_fold)

    if save_csv == True:
        #Clear csv file
        path_plot = folder + '/results_cross_val_model' + str(model_num) + '.csv'
        f = open(path_plot,'w+')
        f.close()
        path_mean = folder + '/roc_mean_model' + str(model_num) + '.csv'
        f = open(path_mean,'w+')
        f.close()

    for i in range(0,n_folds):
        index1 = i*num_patients_per_fold
        index2 = (i+1)*num_patients_per_fold
        temp_fold_X = []
        temp_fold_Y = []
        temp_fold_patient_ID = []
        temp_fold = all_data[index1:index2]

        #testar att använda en gammal funktion ist:

        temp_fold_X, temp_fold_Y, temp_fold_patient_ID = get_im_and_labels_as_array(temp_fold)

        temp_fold_X = normalize_im(temp_fold_X)

        for patient in temp_fold:
            patient_ID = patient[0]
            #print('---------------\npatient_ID: ', patient_ID)
            inner_dict = patient[1]
            #print('---------------\ninner_dict: ', inner_dict)
            images = inner_dict['images']
            #print('---------------\nimages: ', images)
            label = inner_dict['label']
            #print('---------------\nlabel: ', label)

            images = np.asarray(images)
            label = np.asarray(label)
            #print('shape images: ', images.shape)
            #print('shape label: ', label.shape)

            num_im = len(images)
            #print('num_im: ', num_im)
            #temp_fold_X.append(images[0])

            for j in range(0, num_im):
                temp_fold_Y.append(label)
                temp_fold_X.append(images[j,:,:])
                temp_fold_patient_ID.append(patient_ID)
                #images = np.asarray(images)
                #label = np.asarray(label)
                #print('shape images[j,:,:]: ', images[j,:,:].shape)
                #print('shape label: ', label.shape)
                #print(j)

        if i != n_folds - 1:
            folds_X.append(temp_fold_X)
            folds_Y.append(temp_fold_Y)

        else: #Test-folden
            X_test = temp_fold_X
            Y_test = temp_fold_Y
            patient_IDs = np.asarray(temp_fold_patient_ID)

            X_test = np.asarray(X_test)
            Y_test = np.asarray(Y_test)
            np.save("X_test_images.npy", X_test)
            np.save("Y_test_labels.npy", Y_test)
            np.save("patient_IDs_test.npy", patient_IDs)

    #Class weights
    n_samples = np.vstack(folds_X).shape[0]
    n_classes = len(set(np.hstack(folds_Y)))
    class_weights = n_samples / (n_classes * np.bincount(np.hstack(folds_Y)))
    class_weights = {0: class_weights[0], 1: class_weights[1]}
    print("class weights-------------", class_weights, "\n")

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    lst_acc = []
    lst_sens =  []
    lst_spec = []
    Y_valid = []
    X_valid = []

    for i in range(0, n_folds-1):
        copy_folds_X = folds_X.copy()
        copy_folds_Y = folds_Y.copy()

        X_valid = copy_folds_X[i]
        Y_valid = copy_folds_Y[i]
        X_valid = np.asarray(X_valid)
        Y_valid = np.asarray(Y_valid)

        del copy_folds_X[i]
        del copy_folds_Y[i]

        X_train = np.vstack(copy_folds_X)
        Y_train = np.hstack(copy_folds_Y)

        Y_train_non_zero = np.count_nonzero(Y_train)
        Y_test_non_zero = np.count_nonzero(Y_test)
        Y_valid_non_zero = np.count_nonzero(Y_valid)

        print('% AFF train: ', Y_train_non_zero/len(Y_train))
        print('% AFF test: ', Y_test_non_zero/len(Y_test))
        print('% AFF valid: ', Y_valid_non_zero/len(Y_valid),"\n")


        print("-----------------------------MODEL number------------------ ", model_num)
        print("----------------------------AUGMENT setting--------------------", aug_setting)
        print("-----------------------------FOLD number------------------ ", i+1)

        model, opt, callback  = build_im_model(learning_rate, patience, aug_setting, fold_num=(i+1))

        history = train_im_model(model, X_train, Y_train, X_valid, Y_valid, callback, class_weights, optimizer=opt, num_epochs=num_epochs)

        Y_pred = model.predict(X_test)

        new_Y_pred, new_patient_IDs, new_Y_test = pred_patient(Y_pred, patient_IDs, Y_test)

        print("new_Y_pred-----------------", new_Y_pred, "\n")
        print("new_patient_IDs----------------", new_patient_IDs, "\n")
        print("new_Y_test--------------------", new_Y_test, "\n")

        mean_tpr, mean_fpr, lst_acc, lst_sens, lst_spec = Evaluation_CrossVal_patients(history, new_Y_test, new_Y_pred, model, save_csv, i, model_num, mean_tpr, mean_fpr, lst_acc, lst_sens, lst_spec, folder)

        model.save('image_model_patients_extended_alt_lr/fold_' + str(i) + '.h5')

        plt_acc(path_csv=path_plot, fold=i, path_save='Plots_for_report_images_patients_alt_lr')
        plt_loss(path_csv=path_plot, fold=i, path_save='Plots_for_report_images_patients_alt_lr')

        keras.backend.clear_session()

    mean_tpr /= (n_folds - 1)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)

    #Calculate mean and std of accuracy and sensitivity, across all folds
    array_acc = np.asarray(lst_acc)
    array_sens = np.asarray(lst_sens)
    mean_acc = np.mean(array_acc)
    std_acc = np.std(array_acc)
    mean_sens = np.mean(array_sens)
    std_sens = np.std(array_sens)

    #Specificity
    array_spec = np.asarray(lst_spec)
    mean_spec = np.mean(array_spec)
    std_spec = np.std(array_spec)

    # Save mean roc
    if save_csv:
        path = folder + '/roc_mean_model' + str(model_num) + '.csv'
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([mean_tpr, mean_fpr, mean_auc, mean_acc, std_acc, mean_sens, std_sens, mean_spec, std_spec])
            file.close()



def plt_roc_patient_based(filename='results_cross_val_model', filename_mean_roc='roc_mean_model', save_or_show_fig='show', model_num=1):
    num_folds = 0
    sum_roc_auc = 0
    array_roc_auc = np.array([])
    filename = filename + str(model_num) + '.csv'
    filename_mean_roc  = filename_mean_roc + str(model_num) + '.csv'
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file, delimiter=',')
        c = 0
        tpr_temp_sum = np.zeros(200,)
        fpr_temp_sum = np.zeros(200,)

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
    plt.plot([0,1], [0,1], '--', color=(0.6,0.6,0.6), label='Random Classifier')
    model_num_title = str(model_num).split('_')
    model_num_title = model_num_title[0]
    plt.title('Model ' + str(model_num_title) + ': ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    if save_or_show_fig == 'show':
        plt.show()
    else:
        plt.savefig('Plots_for_report_images_patients_alt_lr/ROC_image_model' + str(model_num) + '.eps')
        plt.savefig('Plots_for_report_images_patients_alt_lr/ROC_image_model' + str(model_num) + '.png')


