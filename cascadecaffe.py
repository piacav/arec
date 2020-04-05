from typing import List, Union
from sys import platform
import numpy as np
import cv2
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, multilabel_confusion_matrix


# Initialization variables
r_age, c_age, r_gen, c_gen = None, None, None, None
wrong_age, correct_age, wrong_gen, correct_gen, unrecognized = 0, 0, 0, 0, 0
y_test_age, y_pred_age, y_test_gen, y_pred_gen = [], [], [], []
FalsePositive_age, FalseNegative_age, TrueNegative_age = [], [], []
FalsePositive_gen, FalseNegative_gen, TrueNegative_gen = [], [], []
confusion_matrix_age = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
#confusion_matrix_age = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
confusion_matrix_gen = [[0, 0], [0, 0]]
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_class = [15, 30, 100]
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']
classes_age = {1: "Young",
               2: "Adult",
               3: "Old"}
classes_gen = {1: "Male",
               2: "Female"}
test_images = []

# Create dataset path
dataset_path = 'images/'

# Specify the Haar classifier
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_default.xml')

# Specify font used for plotting
font = cv2.FONT_HERSHEY_SIMPLEX

# Function to calculate the class of age
def classifier_age(age):
    if type(age) == str:
        max = ((age.strip("(").strip(")").replace(',', '')).split(' '))[1]
        max = int(max)
    else:
        max = age

    if max <= age_class[0]:
        return 1
    elif max <= age_class[1]:
        return 2
    return 3

# Return the appropriate string for gender class
def classifier_gender(gender):
    return 'Male' if gender.upper() == 'M' else 'Female'

# Load model for age called caffe
def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('genderage/deploy_age.prototxt', 'genderage/age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('genderage/deploy_gender.prototxt', 'genderage/gender_net.caffemodel')
    return (age_net, gender_net)

test_file = open("test_set.txt", "r")
for i in test_file:
    test_images.append(i.rstrip())

# Load a model
age_net, gender_net = load_caffe_models()
for test_image in test_images:

    print('Name Img       : ' + test_image)
    # Create image path
    imagePath = dataset_path + test_image + '.jpg'
    # Read image
    ti = cv2.imread(imagePath)

    # Create a copy of the image to prevent any changes to the original one
    image_copy = ti.copy()

    # Convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, 1.1, 5)
    print('Detected Faces : ' + str(len(faces_rect)))

    if len(faces_rect) == 0:
        unrecognized += 1
        continue

    for (x, y, w, h) in faces_rect:

        # Create rectangle on image
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face_img = image_copy[y:y + h, x:x + w]

        # Create the blob for age and gender detection
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gen_det = gender_list[gender_preds[0].argmax()]
        print("Gen Pred       : " + gen_det)

        # Get real gender
        real_gen = classifier_gender(test_image[-1])
        print("Gen Real       : " + real_gen)
        print("Gen Confidence : " + str(gender_preds[0]))

        # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age: Union[str, List[str]] = age_list[age_preds[0].argmax()]
        print("Age Range Pred : " + age)

        # Define real age and class
        real_age = test_image[4:6]
        print("Age Real       : " + real_age)

        # Define class
        age_det = classes_age.get(classifier_age(age))
        print("Age Class Pred : " + age_det)

        # Get real age class
        real_age_class = classes_age.get(classifier_age(int(real_age)))
        print("Age Real Class : " + real_age_class)

        print("Age Confidence : " + str(age_preds[0]))
        print('########################################################')

        y_pred_age.append(age_det)
        y_test_age.append(real_age_class)
        y_pred_gen.append(gen_det)
        y_test_gen.append(real_gen)

        # Age check
        if age_det == real_age_class:
            correct_age += 1
            for key, item in classes_age.items():
                if item == age_det:
                    confusion_matrix_age[key - 1][key - 1] += 1
                    break
        else:
            wrong_age += 1
            for wkey, witem in classes_age.items():
                if witem == real_age_class:
                    r_age = wkey
                if witem == age_det:
                    c_age = wkey
            if r_age is not None and c_age is not None:
                confusion_matrix_age[r_age - 1][c_age - 1] += 1

        # Gender check
        if real_gen == gen_det:
            correct_gen += 1
            for key, item in classes_gen.items():
                if item == gen_det:
                    confusion_matrix_gen[key - 1][key - 1] += 1
                    break
        else:
            wrong_gen += 1
            for wkey, witem in classes_gen.items():
                if witem == real_gen:
                    r_gen = wkey
                if witem == gen_det:
                    c_gen = wkey
            if r_gen is not None and c_gen is not None:
                confusion_matrix_gen[r_gen - 1][c_gen - 1] += 1

        # Visualize image
        #overlay_text = "%s %s" % (gender, age)
        #cv2.putText(image_copy, overlay_text, (0, 30), font, 1, (255, 0, 0), 2)
        #cv2.imshow('image', image_copy)
        #cv2.waitKey(0)

# Convert python lists in nparray
cmarray_age = np.array(confusion_matrix_age)
cmarray_gen = np.array(confusion_matrix_gen)

# Compute age and gender metrics for performance evaluation
TruePositive_age = np.diag(cmarray_age)
TruePositive_gen = np.diag(cmarray_gen)

for i_cm in range(len(classes_age)):
    FalsePositive_age.append(sum(cmarray_age[:, i_cm]) - cmarray_age[i_cm, i_cm])
    FalseNegative_age.append(sum(cmarray_age[i_cm, :]) - cmarray_age[i_cm, i_cm])
    temp = np.delete(cmarray_age, i_cm, 0)  # delete ith row
    temp = np.delete(temp, i_cm, 1)  # delete ith column
    TrueNegative_age.append(sum(sum(temp)))

    if i_cm == len(classes_gen): break
    FalsePositive_gen.append(sum(cmarray_gen[:, i_cm]) - cmarray_gen[i_cm, i_cm])
    FalseNegative_gen.append(sum(cmarray_gen[i_cm, :]) - cmarray_gen[i_cm, i_cm])
    temp2 = np.delete(cmarray_gen, i_cm, 0)  # delete ith row
    temp2 = np.delete(temp2, i_cm, 1)  # delete ith column
    TrueNegative_gen.append(sum(sum(temp2)))

# Plot non-normalized age confusion matrix
fig_age, ax_age = plot_confusion_matrix(conf_mat=cmarray_age,
                                colorbar=True,
                                class_names=classes_age.items())

# Plot non-normalized gender confusion matrix
fig_gen, ax_gen = plot_confusion_matrix(conf_mat=cmarray_gen,
                                colorbar=True,
                                class_names=classes_gen.items())

# Print results

output_tot = ('TOTAL     : ' + str(correct_age + wrong_age + unrecognized) +
            '\nNO FACE   : ' + str(unrecognized))

output_age = ('AGE DATA' + '\n' + classification_report(y_test_age, y_pred_age) +
            '\nCORRECT : ' + str(correct_age) +
            '\nWRONG   : ' + str(wrong_age) +
            '\nTRUE POSITIVES   : ' + str(TruePositive_age) +
            '\nFALSE POSITIVES  : ' + str(FalsePositive_age) +
            '\nFALSE NEGATIVES  : ' + str(FalseNegative_age) +
            '\nTRUE NEGATIVES   : ' + str(TrueNegative_age) +
            '\nCONFUSION MATRIX :\n' + str(cmarray_age))

output_gen = ('GENDER DATA' + '\n' + classification_report(y_test_gen, y_pred_gen) +
            '\nCORRECT : ' + str(correct_gen) +
            '\nWRONG   : ' + str(wrong_gen) +
            '\nTRUE POSITIVES   : ' + str(TruePositive_gen) +
            '\nFALSE POSITIVES  : ' + str(FalsePositive_gen) +
            '\nFALSE NEGATIVES  : ' + str(FalseNegative_gen) +
            '\nTRUE NEGATIVES   : ' + str(TrueNegative_gen) +
            '\nCONFUSION MATRIX :\n' + str(cmarray_gen))

line = ("________________________________________________________\n")

print('\n' + line + output_tot + '\n' + line + output_age + '\n' + line + output_gen + '\n' + line)
plt.show()
