import os
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
import numpy as np
from tensorflow import keras
import cv2
import matplotlib.image as imag
import tensorflow as tf
import math
import random

path = "/local/data1/andek67/AFFfusion/Data/images"

def read_im_and_label_to_dict(path = None):

    counter = 0
    data = {}
    counter_different_patients = 0

    assert len(os.listdir(path)) != 0, "directory is empty"

    for x in os.listdir(path):
        if x.endswith(".png"):

            print(counter)

            file_name = str(x)

            path_im = path + '/' + file_name

            file_name_words = file_name.split('_')
            patient_number = int(file_name_words[1])

            img = cv2.imread(path_im)

            shape = np.shape(img)
            print(path_im)
            print(shape)
            height = shape[0]
            h_c = height/2
            width = shape[1]
            w_c = width/2
            pad_size = max(height, width)
            p_c = pad_size/2
            pad_im = np.zeros((pad_size, pad_size, 3))

            pad_im[math.ceil(p_c-h_c):math.ceil(p_c+h_c),math.ceil(p_c-w_c):math.ceil(p_c+w_c)] = img

            pad_im_resize = cv2.resize(pad_im, (224, 224), interpolation = cv2.INTER_AREA)

            if "_AFF_" in path_im:
                print("AFF")
                patient_label = 1
            else:
                print("CONTROL")
                patient_label = 0

            data_keys = data.keys()

            if not patient_number in data_keys:
                data[patient_number] = {"images" : [pad_im_resize],
                "label" : patient_label}
                counter_different_patients += 1
                print("Number of different patients", counter_different_patients, "\n")
            else:
                data[patient_number]["images"].append(pad_im_resize)

        counter+=1

    return data


def read_im_and_label_to_list(path = None):

    counter = 0
    data_list = []

    assert len(os.listdir(path)) != 0, "directory is empty"

    for x in os.listdir(path):
        if x.endswith(".png"):

            print(counter)

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

            pad_im_resize = cv2.resize(pad_im, (224, 224), interpolation = cv2.INTER_AREA)

            if "_AFF_" in path_im:
                print("AFF")
                patient_label = 1
            else:
                print("CONTROL")
                patient_label = 0

            data_list.append([pad_im_resize, patient_label])

        counter+=1

    return data_list

def normalize_im(ds_images):
    ds_images = tf.keras.applications.resnet50.preprocess_input(ds_images) #intensity normalization
    return ds_images



def shuffle_and_convert_list_to_array(data_list):

    random.shuffle(data_list)

    images_list = []
    labels_list = []

    for row in data_list:
        images_list.append(row[0])
        labels_list.append(row[1])

    images_array = np.asarray(images_list)
    labels_array = np.asarray(labels_list)

    return images_array, labels_array


def split_images(images_array, labels_array, train_fraq =0.8):

    num_images = len(images_array)

    print("Total num of patients:", num_images, "\n")

    num_train = int(math.ceil(train_fraq * num_images))
    num_val = int((num_images - num_train) / 2)
    num_test = num_val

    if (num_val + num_test) < (num_images - num_train):
        num_val += 1
        print("Add one patient to the validation set \n")

    print("Number of training patients:", num_train, "\n")
    print("Number of validation patients:", num_val, "\n")
    print("Number of test patients:", num_test, "\n")

    num_tot = num_train + num_val + num_test

    print("Number of train + val + test patients:", num_tot, "\n")

    val_fraq = num_val / num_images
    test_fraq = val_fraq

    print("val and test fraq:", val_fraq)

    train_images_array = images_array[:num_train]
    valid_images_array = images_array[num_train:(num_train+num_val)]
    test_images_array = images_array[(num_train+num_val):]

    train_labels_array = labels_array[:num_train]
    valid_labels_array = labels_array[num_train:(num_train+num_val)]
    test_labels_array = labels_array[(num_train+num_val):]

    return train_images_array, valid_images_array, test_images_array, train_labels_array, valid_labels_array, test_labels_array



def split_and_shuffle_patients(data_dict, train_fraq = 0.75): # Only create list, not splitted

    data_items_list = list(data_dict.items())

    num_patients = len(data_items_list)
    random.shuffle(data_items_list)

    print("Total num of patients:", num_patients, "\n")

    num_train = int(math.ceil(train_fraq * num_patients))
    num_val = int((num_patients - num_train) / 2)
    num_test = num_val

    if (num_val + num_test) < (num_patients - num_train):
        num_val += 1
        print("Add one patient to the validation set \n")

    print("Number of training patients:", num_train, "\n")
    print("Number of validation patients:", num_val, "\n")
    print("Number of test patients:", num_test, "\n")

    num_tot = num_train + num_val + num_test

    print("Number of train + val + test patients:", num_tot, "\n")

    val_fraq = num_val / num_patients
    test_fraq = val_fraq

    print("val and test fraq:", val_fraq)


    train_patients_list = data_items_list[:num_train]
    valid_patients_list = data_items_list[num_train:(num_train + num_val)]
    test_patients_list = data_items_list[(num_train + num_val):]

    return train_patients_list, valid_patients_list, test_patients_list

def get_im_and_labels_as_array(dataset_list): 

    #Create a list of images and a corresponding list of labels, for each dataset
    ds_images = []
    ds_labels = []
    ds_patients = []

    for patient in dataset_list:
        #ds_patients.append(patient[0])
        inner_dict = patient[1]
        images = inner_dict["images"]
        label = inner_dict["label"]
        for image in images:
            ds_images.append(image)
            ds_labels.append(label)
            ds_patients.append(patient[0])

    ds_images = np.asarray(ds_images)
    ds_labels = np.asarray(ds_labels)

    return ds_images, ds_labels, ds_patients

def save_im_and_labels_np(path):
    data_dict = read_im_and_label_to_dict(path)
    train_patients_list, valid_patients_list, test_patients_list = split_and_shuffle_patients(data_dict)

    train_ds_images, train_ds_labels = get_im_and_labels_as_array(train_patients_list)
    valid_ds_images, valid_ds_labels = get_im_and_labels_as_array(valid_patients_list)
    test_ds_images, test_ds_labels = get_im_and_labels_as_array(test_patients_list)

    train_ds_images_norm = normalize_im(train_ds_images)
    valid_ds_images_norm = normalize_im(valid_ds_images)
    test_ds_images_norm = normalize_im(test_ds_images)

    np.save("image_data_np/train_images.npy", train_ds_images_norm)
    np.save("image_data_np/valid_images.npy", valid_ds_images_norm)
    np.save("image_data_np/test_images.npy", test_ds_images_norm)

    np.save("image_data_np/train_labels.npy", train_ds_labels)
    np.save("image_data_np/valid_labels.npy", valid_ds_labels)
    np.save("image_data_np/test_labels.npy", test_ds_labels)

def save_im_and_labels_not_sorted_on_patients(path):
    d_list = read_im_and_label_to_list(path)
    images_array, labels_array = shuffle_and_convert_list_to_array(d_list)
    images_array = normalize_im(images_array)

    train_images_array, valid_images_array, test_images_array, train_labels_array, valid_labels_array, test_labels_array = split_images(images_array, labels_array)

    np.save("image_data_np/train_images.npy", train_images_array)
    np.save("image_data_np/valid_images.npy", valid_images_array)
    np.save("image_data_np/test_images.npy", test_images_array)

    np.save("image_data_np/train_labels.npy", train_labels_array)
    np.save("image_data_np/valid_labels.npy", valid_labels_array)
    np.save("image_data_np/test_labels.npy", test_labels_array)

def save_im_and_labels_not_sorted_on_patients_not_split(path):
    d_list = read_im_and_label_to_list(path)
    images_array, labels_array = shuffle_and_convert_list_to_array(d_list)
    images_array = normalize_im(images_array)

    np.save("image_data_np/all_images.npy", images_array)
    np.save("image_data_np/all_labels.npy", labels_array)

def save_im_and_labels_sorted_on_patients_not_split_list(path):
    data_dict = read_im_and_label_to_dict(path)
    train_patients_list, valid_patients_list, test_patients_list = split_and_shuffle_patients(data_dict, train_fraq=1.0)

    np.save("image_data_np/all_images_list_no_duplicates.npy", train_patients_list)


def load_saved_data():
    train_images = np.load("image_data_np/train_images.npy")
    valid_images = np.load("image_data_np/valid_images.npy")
    test_images = np.load("image_data_np/test_images.npy")

    train_labels = np.load("image_data_np/train_labels.npy")
    valid_labels = np.load("image_data_np/valid_labels.npy")
    test_labels = np.load("image_data_np/test_labels.npy")

    return train_images, valid_images, test_images, train_labels, valid_labels, test_labels


def load_saved_data_not_split():
    all_images = np.load("image_data_np/all_images.npy")
    all_labels = np.load("image_data_np/all_labels.npy")

    return all_images, all_labels

def load_im_and_labels_sorted_on_patients_not_split_list(path):
    all_data = np.load(path, allow_pickle=True)

    return all_data

