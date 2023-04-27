import numpy as np

def pred_patient(Y_pred, patient_nr, Y_GT_images):
    new_Y_pred = np.array([])
    new_Y_GT = np.array([])
    new_patient_nr = np.array([])
    temp_array = np.array([])

    for i in range(0,len(Y_pred)):
        isin = np.isin(patient_nr[i], new_patient_nr)

        if isin == False:
            new_patient_nr = np.append(new_patient_nr, patient_nr[i])
            new_Y_GT = np.append(new_Y_GT,Y_GT_images[i])

            if temp_array.size != 0:
                new_Y_pred = np.append(new_Y_pred, np.mean(temp_array))

            temp_array = np.array([])
            temp_array = np.append(temp_array,Y_pred[i])

            if i == (len(Y_pred) - 1):
                # Last element in Y_pred array
                new_Y_pred = np.append(new_Y_pred, np.mean(temp_array))

        else:
            temp_array = np.append(temp_array, Y_pred[i])
            if i == (len(Y_pred) - 1):
                new_Y_pred = np.append(new_Y_pred, np.mean(temp_array))
    return new_Y_pred, new_patient_nr, new_Y_GT


"""
# Test cases
Y_pred = np.array([[0.92],[0.73],[0.32],[0.85],[0.74],[0.12], [0.25], [0.25], [0.1]])
patient_nr = np.array([[1],[1],[1],[2],[2],[3],[4],[4],[4]])

Y_pred = np.array([[0.1],[0.1],[0.1],[0.4],[0.2],[0.4]])
patient_nr = np.array([[1],[1],[5],[4],[8],[8]])

Y_pred = np.array([[0.92],[0.73],[0.32],[0.85],[0.74],[0.12], [0.25], [0.25], [0.1]])
patient_nr = np.array([[1],[1],[1],[1],[2],[3],[4],[4],[5]])
Y_GT_images = np.array([[1],[1],[1],[1],[0],[0],[0],[0],[1]])

new_Y_pred, new_patient_nr, new_Y_GT = pred_patient(Y_pred, patient_nr, Y_GT_images)
print(new_Y_pred)
print(new_Y_GT)
print(new_Y_GT == new_Y_pred)
"""
