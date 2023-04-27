from image_model import *
from load_images import *
from help_functions_register_data import *
from help_functions_prob_fusion import *
from help_functions_feature_fusion import *
from help_functions_learned_feature_fusion import *

from read_csv_for_plts import *

import time

epochs=10000

learningratePF=1e-4
learningrateFF=1e-4
learningrateLFF=1e-4

# Loop over repetitions, 5 times repeated 5 fold CV
for repetition in range(1, 6):
    print("Repetition ",repetition)

    # Run this function to create numpy arrays from the images (PNG files) and register data, in 6 different folds. 
    # This function includes preprocessing of the images, to downsample them.
    prepare_data_for_fusion(path = '/local/data1/andek67/AFFfusion/Data/images', img_size=224, folder_save = 'numpys_for_crossval', norm=False)
    #prepare_data_for_fusion(path = '/local/data1/andek67/AFFfusion/Data/images_two_per_patient', img_size=224, folder_save = 'numpys_for_crossval', norm=False)
    #prepare_data_for_fusion(path = '/local/data1/andek67/AFFfusion/Data/images_one_per_patient', img_size=224, folder_save = 'numpys_for_crossval', norm=False)


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

    # Create numpy arrays with only one register data instance per patient
    remove_duplicates_for_registerdata(path_register = registerdata_lst_paths, path_label = labels_lst_paths)












    lst_time = []

    #--------------------------------------------
    ## RUN ONLY IMAGES MODEL

    csv_folder = 'csv_images'

    aug = 3
    model = '3_newdataset'

    start = time.time()

    train_images_only(n_folds=5, num_epochs=epochs, save_csv=True, model_num=model, learning_rate=1e-5, patience=100, aug_setting=aug, folder=csv_folder, folder_for_numpys='numpys_for_crossval', repetition=repetition)

    end = time.time()
    elapsed_time = end - start
    lst_time.append(elapsed_time)

    plt_roc_patient_based_fusion(filename=csv_folder + '/results_cross_val_model', filename_mean_roc=csv_folder + '/roc_mean_model', save_or_show_fig='save', model_num=model, path_to_place_plots_in='Plots_for_images', model_type = 'Image Model', repetition=repetition)
    plt.clf()








    #--------------------------------------------
    ## RUN ONLY REGISTER DATA MODEL, age sex

    csv_folder = 'csv_registerdata_agesex'
    model_num = '2_agesex'	

    crossvalidation_registerdata(num_epochs=epochs, model_num=model_num, lr=1e-4, save_csv=True, folder=csv_folder, variable_selection_func = load_registerdata_age_sex_fold, repetition=repetition)

    plt_roc(filename=csv_folder + '/results_cross_val_model',filename_mean_roc=csv_folder + '/roc_mean_model', save_or_show_fig='save', model_num=model_num, folder="Plots_for_registerdata_agesex", repetition=repetition)
    plt.clf()

    #--------------------------------------------
    ## Probability Fusion with age sex

    model_num = '1_PF'
    csv_folder = 'csv_prob_fusion_agesex'

    crossvalidation_PF_model(n_folds=5, learning_rate=learningratePF, patience=100, epochs=epochs, save_csv=True, model_num=model_num, folder=csv_folder, variable_selection_func=load_registerdata_age_sex_fold, only_age_sex = True, all_register_parameters = False, train_metrics_path='Plots_for_prob_fusion_agesex', save_model_path='prob_fusion_agesex_saved_models', repetition=repetition)

    plt_roc_patient_based_fusion(filename=csv_folder + '/results_cross_val_model', filename_mean_roc=csv_folder + '/roc_mean_model', save_or_show_fig='save', model_num=model_num, path_to_place_plots_in='Plots_for_prob_fusion_agesex', model_type='Probability fusion', repetition=repetition)
    plt.clf()

    #--------------------------------------------
    ## Feature Fusion with age sex

    model_num = '1_FF'
    csv_folder = 'csv_feature_fusion_agesex'

    crossvalidation_FF_model(n_folds=5, learning_rate=learningrateFF, patience=100, epochs=epochs, save_csv=True, model_num=model_num, folder=csv_folder, variable_selection_func=load_registerdata_age_sex_fold, only_age_sex = True, all_register_parameters = False, train_metrics_path='Plots_for_feature_fusion_agesex', save_model_path='feature_fusion_saved_models_agesex', repetition=repetition)

    plt_roc_patient_based_fusion(filename=csv_folder + '/results_cross_val_model', filename_mean_roc=csv_folder + '/roc_mean_model', save_or_show_fig='save', model_num=model_num, path_to_place_plots_in='Plots_for_feature_fusion_agesex', model_type='Feature fusion', repetition=repetition)
    plt.clf()


    #--------------------------------------------
    ## Learned Feature Fusion with age sex

    model = '1_LFF'
    csv_folder = 'csv_learned_feature_fusion_agesex'

    start = time.time()

    crossvalidation_LFF_model(n_folds=5, learning_rate=learningrateLFF, patience=100, epochs=epochs, save_csv=True, model_num=model, folder=csv_folder, variable_selection_func=load_registerdata_age_sex_fold, only_age_sex = True, all_register_parameters = False, train_metrics_path='Plots_for_learned_feature_fusion_agesex', save_model_path='learned_feature_fusion_saved_models_agesex', repetition=repetition)

    plt_roc_patient_based_fusion(filename=csv_folder + '/results_cross_val_model', filename_mean_roc=csv_folder + '/roc_mean_model', save_or_show_fig='save', model_num=model, path_to_place_plots_in='Plots_for_learned_feature_fusion_agesex', model_type='Learned feature fusion', repetition=repetition)
    plt.clf()















    #--------------------------------------------
    ## RUN ONLY REGISTER DATA MODEL, binary 7

    csv_folder = 'csv_registerdata_7binary'
    model_num = '2_binary_chosen'	

    start = time.time()

    crossvalidation_registerdata(num_epochs=epochs, model_num=model_num, lr=1e-4, save_csv=True, folder=csv_folder, variable_selection_func = load_registerdata_7_binary_fold, path_to_place_plots_in='Plots_for_registerdata_7binary', repetition=repetition)

    plt_roc(filename=csv_folder + '/results_cross_val_model',filename_mean_roc=csv_folder + '/roc_mean_model', save_or_show_fig='save', model_num=model_num, folder="Plots_for_registerdata_7binary", repetition=repetition)
    plt.clf()

    end = time.time()
    elapsed_time = end - start
    lst_time.append(elapsed_time)
    

    #--------------------------------------------
    ## Probability Fusion with binary register data 7

    model_num = '1_PF'
    csv_folder = 'csv_prob_fusion_7binary'

    start = time.time()

    crossvalidation_PF_model(n_folds=5, learning_rate=learningratePF, patience=100, epochs=epochs, save_csv=True, model_num=model_num, folder=csv_folder, variable_selection_func=load_registerdata_7_binary_fold, only_age_sex = False, all_register_parameters = False,   train_metrics_path='Plots_for_prob_fusion_7binary', save_model_path='prob_fusion_saved_models_7binary', repetition=repetition)

    plt_roc_patient_based_fusion(filename=csv_folder + '/results_cross_val_model', filename_mean_roc=csv_folder+'/roc_mean_model', save_or_show_fig='save', model_num=model_num, path_to_place_plots_in='Plots_for_prob_fusion_7binary', model_type='Probability fusion', repetition=repetition)
    plt.clf()

    end = time.time()
    elapsed_time = end - start
    lst_time.append(elapsed_time)

    #--------------------------------------------
    ## Feature Fusion with binary register data 7

    model_num = '1_FF'
    csv_folder = 'csv_feature_fusion_7binary'

    start = time.time()

    crossvalidation_FF_model(n_folds=5, learning_rate=learningrateFF, patience=100, epochs=epochs, save_csv=True, model_num=model_num, folder=csv_folder, variable_selection_func=load_registerdata_7_binary_fold, only_age_sex = False, all_register_parameters = False, train_metrics_path='Plots_for_feature_fusion_7binary', save_model_path='feature_fusion_saved_models_7binary', repetition=repetition)

    plt_roc_patient_based_fusion(filename=csv_folder + '/results_cross_val_model', filename_mean_roc=csv_folder + '/roc_mean_model', save_or_show_fig='save', model_num=model_num, path_to_place_plots_in='Plots_for_feature_fusion_7binary', model_type='Feature fusion', repetition=repetition)
    plt.clf()

    end = time.time()
    elapsed_time = end - start
    lst_time.append(elapsed_time)

    #--------------------------------------------
    ## Learned Feature Fusion with binary register data 7

    model = '1_LFF'
    csv_folder = 'csv_learned_feature_fusion_7binary'

    start = time.time()

    crossvalidation_LFF_model(n_folds=5, learning_rate=learningrateLFF, patience=100, epochs=epochs, save_csv=True, model_num=model, folder=csv_folder, variable_selection_func=load_registerdata_7_binary_fold, only_age_sex = False, all_register_parameters = False, train_metrics_path='Plots_for_learned_feature_fusion_7binary', save_model_path='learned_feature_fusion_saved_models_7binary', repetition=repetition)

    plt_roc_patient_based_fusion(filename=csv_folder + '/results_cross_val_model', filename_mean_roc=csv_folder + '/roc_mean_model', save_or_show_fig='save', model_num=model, path_to_place_plots_in='Plots_for_learned_feature_fusion_7binary', model_type='Learned feature fusion', repetition=repetition)
    plt.clf()

    end = time.time()
    elapsed_time = end - start
    lst_time.append(elapsed_time)

    print("Times ",lst_time)















    #--------------------------------------------
    ## RUN ONLY REGISTER DATA MODEL, non-binary 7

    csv_folder = 'csv_registerdata_7nonbinary'
    model_num = '2_days_chosen'	

    start = time.time()

    crossvalidation_registerdata(num_epochs=epochs, model_num=model_num, lr=1e-4, save_csv=True, folder=csv_folder, variable_selection_func = load_registerdata_7_nonbinary_fold, path_to_place_plots_in='Plots_for_registerdata_7nonbinary', repetition=repetition)

    plt_roc(filename=csv_folder + '/results_cross_val_model',filename_mean_roc=csv_folder + '/roc_mean_model', save_or_show_fig='save', model_num=model_num, folder="Plots_for_registerdata_7nonbinary", repetition=repetition)
    plt.clf()

    end = time.time()
    elapsed_time = end - start
    lst_time.append(elapsed_time)
    

    #--------------------------------------------
    ## Probability Fusion with non-binary register data 7 

    model_num = '1_PF'
    csv_folder = 'csv_prob_fusion_7nonbinary'

    start = time.time()

    crossvalidation_PF_model(n_folds=5, learning_rate=learningratePF, patience=100, epochs=epochs, save_csv=True, model_num=model_num, folder=csv_folder, variable_selection_func=load_registerdata_7_nonbinary_fold, only_age_sex = False, all_register_parameters = False,   train_metrics_path='Plots_for_prob_fusion_7nonbinary', save_model_path='prob_fusion_saved_models_7nonbinary', repetition=repetition)

    plt_roc_patient_based_fusion(filename=csv_folder + '/results_cross_val_model', filename_mean_roc=csv_folder+'/roc_mean_model', save_or_show_fig='save', model_num=model_num, path_to_place_plots_in='Plots_for_prob_fusion_7nonbinary', model_type='Probability fusion', repetition=repetition)
    plt.clf()

    end = time.time()
    elapsed_time = end - start
    lst_time.append(elapsed_time)

    #--------------------------------------------
    ## Feature Fusion with non-binary register data 7

    model_num = '1_FF'
    csv_folder = 'csv_feature_fusion_7nonbinary'

    start = time.time()

    crossvalidation_FF_model(n_folds=5, learning_rate=learningrateFF, patience=100, epochs=epochs, save_csv=True, model_num=model_num, folder=csv_folder, variable_selection_func=load_registerdata_7_nonbinary_fold, only_age_sex = False, all_register_parameters = False, train_metrics_path='Plots_for_feature_fusion_7nonbinary', save_model_path='feature_fusion_saved_models_7nonbinary', repetition=repetition)

    plt_roc_patient_based_fusion(filename=csv_folder + '/results_cross_val_model', filename_mean_roc=csv_folder + '/roc_mean_model', save_or_show_fig='save', model_num=model_num, path_to_place_plots_in='Plots_for_feature_fusion_7nonbinary', model_type='Feature fusion', repetition=repetition)
    plt.clf()

    end = time.time()
    elapsed_time = end - start
    lst_time.append(elapsed_time)

    #--------------------------------------------
    ## Learned Feature Fusion with non-binary register data 7

    model = '1_LFF'
    csv_folder = 'csv_learned_feature_fusion_7nonbinary'

    start = time.time()

    crossvalidation_LFF_model(n_folds=5, learning_rate=learningrateLFF, patience=100, epochs=epochs, save_csv=True, model_num=model, folder=csv_folder, variable_selection_func=load_registerdata_7_nonbinary_fold, only_age_sex = False, all_register_parameters = False, train_metrics_path='Plots_for_learned_feature_fusion_7nonbinary', save_model_path='learned_feature_fusion_saved_models_7nonbinary', repetition=repetition)

    plt_roc_patient_based_fusion(filename=csv_folder + '/results_cross_val_model', filename_mean_roc=csv_folder + '/roc_mean_model', save_or_show_fig='save', model_num=model, path_to_place_plots_in='Plots_for_learned_feature_fusion_7nonbinary', model_type='Learned feature fusion', repetition=repetition)
    plt.clf()

    end = time.time()
    elapsed_time = end - start
    lst_time.append(elapsed_time)

    print("Times ",lst_time)












    #--------------------------------------------
    ## RUN ONLY REGISTER DATA MODEL, binary all

    csv_folder = 'csv_registerdata_allbinary'
    model_num = '2_binary_all'	

    start = time.time()

    crossvalidation_registerdata(num_epochs=epochs, model_num=model_num, lr=1e-4, save_csv=True, folder=csv_folder, variable_selection_func = load_registerdata_all_binary_fold, repetition=repetition)

    plt_roc(filename=csv_folder + '/results_cross_val_model',filename_mean_roc=csv_folder + '/roc_mean_model', save_or_show_fig='save', model_num=model_num, folder="Plots_for_registerdata_allbinary", repetition=repetition)
    plt.clf()

    #--------------------------------------------
    ## Probability Fusion with all binary register data 

    model_num = '1_PF'
    csv_folder = 'csv_prob_fusion_allbinary'

    start = time.time()

    crossvalidation_PF_model(n_folds=5, learning_rate=learningratePF, patience=100, epochs=epochs, save_csv=True, model_num=model_num, folder=csv_folder, variable_selection_func=load_registerdata_all_binary_fold, only_age_sex = False, all_register_parameters = True,   train_metrics_path='Plots_for_prob_fusion_allbinary', save_model_path='prob_fusion_saved_models_allbinary', repetition=repetition)

    plt_roc_patient_based_fusion(filename=csv_folder + '/results_cross_val_model', filename_mean_roc=csv_folder+'/roc_mean_model', save_or_show_fig='save', model_num=model_num, path_to_place_plots_in='Plots_for_prob_fusion_allbinary', model_type='Probability fusion', repetition=repetition)
    plt.clf()

    #--------------------------------------------
    ## Feature Fusion with all binary register data 

    model_num = '1_FF'
    csv_folder = 'csv_feature_fusion_allbinary'

    start = time.time()

    crossvalidation_FF_model(n_folds=5, learning_rate=learningrateFF, patience=100, epochs=epochs, save_csv=True, model_num=model_num, folder=csv_folder, variable_selection_func=load_registerdata_all_binary_fold, only_age_sex = False, all_register_parameters = True, train_metrics_path='Plots_for_feature_fusion_allbinary', save_model_path='feature_fusion_saved_models_allbinary', repetition=repetition)

    plt_roc_patient_based_fusion(filename=csv_folder + '/results_cross_val_model', filename_mean_roc=csv_folder + '/roc_mean_model', save_or_show_fig='save', model_num=model_num, path_to_place_plots_in='Plots_for_feature_fusion_allbinary', model_type='Feature fusion', repetition=repetition)
    plt.clf()

    #--------------------------------------------
    ## Learned Feature Fusion with all binary register data

    model = '1_LFF'
    csv_folder = 'csv_learned_feature_fusion_allbinary'

    start = time.time()

    crossvalidation_LFF_model(n_folds=5, learning_rate=learningrateLFF, patience=100, epochs=epochs, save_csv=True, model_num=model, folder=csv_folder, variable_selection_func=load_registerdata_all_binary_fold, only_age_sex = False, all_register_parameters = True, train_metrics_path='Plots_for_learned_feature_fusion_allbinary', save_model_path='learned_feature_fusion_saved_models_allbinary', repetition=repetition)

    plt_roc_patient_based_fusion(filename=csv_folder + '/results_cross_val_model', filename_mean_roc=csv_folder + '/roc_mean_model', save_or_show_fig='save', model_num=model, path_to_place_plots_in='Plots_for_learned_feature_fusion_allbinary', model_type='Learned feature fusion', repetition=repetition)
    plt.clf()










    #--------------------------------------------
    ## RUN ONLY REGISTER DATA MODEL, nonbinary all

    csv_folder = 'csv_registerdata_allnonbinary'
    model_num = '2_nonbinary_all'	

    start = time.time()

    crossvalidation_registerdata(num_epochs=epochs, model_num=model_num, lr=1e-4, save_csv=True, folder=csv_folder, variable_selection_func = load_registerdata_all_nonbinary_fold, repetition=repetition)

    plt_roc(filename=csv_folder + '/results_cross_val_model',filename_mean_roc=csv_folder + '/roc_mean_model', save_or_show_fig='save', model_num=model_num, folder="Plots_for_registerdata_allnonbinary", repetition=repetition)
    plt.clf()



    #--------------------------------------------
    ## Probability Fusion with all nonbinary register data 

    model_num = '1_PF'
    csv_folder = 'csv_prob_fusion_allnonbinary'

    start = time.time()

    crossvalidation_PF_model(n_folds=5, learning_rate=learningratePF, patience=100, epochs=epochs, save_csv=True, model_num=model_num, folder=csv_folder, variable_selection_func=load_registerdata_all_nonbinary_fold, only_age_sex = False, all_register_parameters = True,    train_metrics_path='Plots_for_prob_fusion_allnonbinary', save_model_path='prob_fusion_saved_models_allnonbinary', repetition=repetition)

    plt_roc_patient_based_fusion(filename=csv_folder + '/results_cross_val_model', filename_mean_roc=csv_folder+'/roc_mean_model', save_or_show_fig='save', model_num=model_num, path_to_place_plots_in='Plots_for_prob_fusion_allnonbinary', model_type='Probability fusion', repetition=repetition)
    plt.clf()

    #--------------------------------------------
    ## Feature Fusion with all nonbinary register data 

    model_num = '1_FF'
    csv_folder = 'csv_feature_fusion_allnonbinary'

    start = time.time()

    crossvalidation_FF_model(n_folds=5, learning_rate=learningrateFF, patience=100, epochs=epochs, save_csv=True, model_num=model_num, folder=csv_folder, variable_selection_func=load_registerdata_all_nonbinary_fold, only_age_sex = False, all_register_parameters = True, train_metrics_path='Plots_for_feature_fusion_allnonbinary', save_model_path='feature_fusion_saved_models_allnonbinary', repetition=repetition)

    plt_roc_patient_based_fusion(filename=csv_folder + '/results_cross_val_model', filename_mean_roc=csv_folder + '/roc_mean_model', save_or_show_fig='save', model_num=model_num, path_to_place_plots_in='Plots_for_feature_fusion_allnonbinary', model_type='Feature fusion', repetition=repetition)
    plt.clf()

    #--------------------------------------------
    ## Learned Feature Fusion with all nonbinary register data

    model = '1_LFF'
    csv_folder = 'csv_learned_feature_fusion_allnonbinary'

    start = time.time()

    crossvalidation_LFF_model(n_folds=5, learning_rate=learningrateLFF, patience=100, epochs=epochs, save_csv=True, model_num=model, folder=csv_folder, variable_selection_func=load_registerdata_all_nonbinary_fold, only_age_sex = False, all_register_parameters = True, train_metrics_path='Plots_for_learned_feature_fusion_allnonbinary', save_model_path='learned_feature_fusion_saved_models_allnonbinary', repetition=repetition)

    plt_roc_patient_based_fusion(filename=csv_folder + '/results_cross_val_model', filename_mean_roc=csv_folder + '/roc_mean_model', save_or_show_fig='save', model_num=model, path_to_place_plots_in='Plots_for_learned_feature_fusion_allnonbinary', model_type='Learned feature fusion', repetition=repetition)
    plt.clf()








