import csv
import matplotlib.pyplot as plt
import numpy as np

def make_plot_lst(str, mode=", "):

    str = str.replace('[','')
    str = str.replace(']','')
    plt_lst = []
    for e in str.split(mode):
        if e != '':
            plt_lst.append(float(e))
    return plt_lst

def plt_acc(path_csv, fold=0, path_save='Plots_for_report_2', repetition=1):
    fold = fold + 1

    with open(path_csv, 'r', newline='') as file:
        reader = csv.reader(file, delimiter=',')
        c = 0
        for row in reader:
            if c == 0:
                header = row
                c = c + 1
            else:
                c = c + 1
                model_nr = row[0]
                bin_acc = row[1]
                val_bin_acc = row[2]
                epochs = row[8]
                x_axis = range(1, int(epochs) + 1)

                bin_acc = make_plot_lst(bin_acc)
                val_bin_acc = make_plot_lst(val_bin_acc)

                plt.plot(x_axis, bin_acc)
                plt.plot(x_axis, val_bin_acc)

                plt.title('Fold ' + str(model_nr) + ' accuracy. Early stopping epoch: {:d}'.format(int(epochs)))
                plt.ylabel('Accuracy')
                plt.xlabel('Epochs')
                plt.legend(['Train', 'Valid'], loc='lower right')
                #plt.show()
                plt.savefig(path_save + '/Accuracy_fold' + str(fold) + '_repetition_' + str(repetition) + '.eps')
                plt.savefig(path_save + '/Accuracy_fold' + str(fold) + '_repetition_' + str(repetition) + '.png')
                plt.clf()

def plt_loss(path_csv, fold=0, path_save='Plots_for_report_2', repetition=1):
    fold = fold + 1

    with open(path_csv, 'r', newline='') as file:
        reader = csv.reader(file, delimiter=',')
        c = 0
        for row in reader:
            if c == 0:
                header = row
                c = c + 1
            else:
                c = c + 1
                model_nr = row[0]
                loss = row[3]
                val_loss = row[4]
                epochs = row[8]
                x_axis = range(1, int(epochs) + 1)

                loss = make_plot_lst(loss)
                val_loss = make_plot_lst(val_loss)

                plt.plot(x_axis, loss)
                plt.plot(x_axis, val_loss)

                plt.title('Fold ' + str(model_nr) + ' loss. Early stopping epoch: {:d}'.format(int(epochs)))
                plt.ylabel('Loss')
                plt.xlabel('Epochs')
                plt.legend(['Train', 'Valid'], loc='upper right')
                #plt.show()
                plt.savefig(path_save + '/Loss_fold' + str(fold) + '_repetition_' + str(repetition) + '.eps')
                plt.savefig(path_save + '/Loss_fold' + str(fold) + '_repetition_' + str(repetition) + '.png')
                plt.clf()


def plt_roc(filename='results_cross_val_model', filename_mean_roc='roc_mean_model', save_or_show_fig='show', model_num=1, folder="Plots", repetition = 1):
    num_folds = 0
    sum_roc_auc = 0
    array_roc_auc = np.array([])
    filename = filename + str(model_num) + '_repetition_' + str(repetition) + '.csv'
    filename_mean_roc  = filename_mean_roc + str(model_num) + '_repetition_' + str(repetition) + '.csv'
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
        c = 0
        for row in reader:
            if c == 0:
                header = row
                c = c + 1
            else:
                tpr_mean = row[0]
                fpr_mean = row[1]
                roc_auc_mean = row[2]

                tpr_mean = make_plot_lst(tpr_mean, mode=" ")
                fpr_mean = make_plot_lst(fpr_mean, mode=" ")

                plt.plot(fpr_mean, tpr_mean, 'k--', label=r'Mean ROC, AUC = {:.3f}'.format(float(roc_auc_mean)))
    plt.plot([0,1], [0,1], '--', color=(0.6,0.6,0.6), label='Random Classifier, AUC = 0.5')
    model_num_title = str(model_num).split('_')
    model_num_title = model_num_title[0]
    plt.title('Register data model ' + str(model_num_title) + ': ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    if save_or_show_fig == 'show':
        plt.show()
    else:
        plt.savefig(folder + '/ROC_register_data_model' + str(model_num) + '_repetition_' + str(repetition) + '.eps')
        plt.savefig(folder + '/ROC_register_data_model' + str(model_num) + '_repetition_' + str(repetition) + '.png')
