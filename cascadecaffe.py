from typing import List, Union
from sys import platform
import numpy as np
import cv2
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# Initialization variables
r_age, c_age, r_gen, c_gen = None, None, None, None
wrong_age, correct_age, wrong_gen, correct_gen, unrecognized = 0, 0, 0, 0, 0
y_test_age, y_pred_age, y_test_gen, y_pred_gen = [], [], [], []
confusion_matrix_age = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
confusion_matrix_gen = [[0, 0], [0, 0]]
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_class = [10, 50, 100]
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']
classes_age = {1: "Young",
               2: "Teen",
               3: "Adult"}
classes_gen = {1: "Male",
               2: "Female"}
test_images = []

# Specify the Haar classifier
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_alt.xml')

# Specify font used for plotting
font = cv2.FONT_HERSHEY_SIMPLEX

# Creation dataset path for different O.S.
if platform == 'win32':
    #dataset_path = 'C:\\Users\\andry\\Desktop\\FGNET\\images\\'
    dataset_path = 'D:\\FGNET\\images\\'
elif platform == 'darwin':
    dataset_path = '/Users/piacavasinni/Desktop/FGNET/images/'
else:
    dataset_path = ''

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
    #elif max <= age_class[2]:
    #    return 3
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

for test_image in test_images:

    # Load a model
    age_net, gender_net = load_caffe_models()

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

    if len(faces_rect) == 0:
        unrecognized += 1
        continue

    for (x, y, w, h) in faces_rect:

        # Create rectangle on image
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face_img = image_copy[y:y + h, h:h + w]

        # Create the blob for age and gender detection
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gen_det = gender_list[gender_preds[0].argmax()]
        print("Gender Pred: " + gen_det)

        real_gen = classifier_gender(test_image[-1])
        print("Real Gen  : " + real_gen)

        # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age: Union[str, List[str]] = age_list[age_preds[0].argmax()]
        print("Age Range : " + age)

        # Define class
        age_det = classes_age.get(classifier_age(age))
        print("Age Class : " + age_det)

        # Define real age and class
        real_age = test_image[4:6]
        print("Real Age  : " + real_age)

        real_age_class = classes_age.get(classifier_age(int(real_age)))
        print("Real Class: " + real_age_class)

        print("Score Age : " + str(age_preds[0]))
        print("Score Gen : " + str(gender_preds[0]))
        print('#######################')

        y_pred_age.append(age_det)
        y_test_age.append(real_age_class)
        y_pred_gen.append(gen_det)
        y_test_gen.append(real_gen)

        # Verification age
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

# Compute metrics for performance evaluation
cmarray = np.array(confusion_matrix_age)
cmarray_gen = np.array(confusion_matrix_gen)

TruePositive = np.diag(cmarray)

FalsePositive, FalseNegative, TrueNegative, Accuracy = [], [], [], []

for ifp in range(3):
    FalsePositive.append(sum(cmarray[:, ifp]) - cmarray[ifp, ifp])

for ifn in range(3):
    FalseNegative.append(sum(cmarray[ifn, :]) - cmarray[ifn, ifn])

for itn in range(3):
    temp = np.delete(cmarray, itn, 0)  # delete ith row
    temp = np.delete(temp, itn, 1)  # delete ith column
    TrueNegative.append(sum(sum(temp)))

for c in range(3):
    Accuracy.append((TruePositive[c] + TrueNegative[c])/(TruePositive[c] +
                                                         TrueNegative[c] +
                                                         FalsePositive[c] +
                                                         FalseNegative[c]))

# Plot non-normalized confusion matrix
cmarray_age = np.array(confusion_matrix_age)
fig, ax = plot_confusion_matrix(conf_mat=cmarray_age,
                                colorbar=True,
                                class_names=classes_age.items())

print(classification_report(y_test_age, y_pred_age))
print(classification_report(y_test_gen, y_pred_gen))

print('INDOVINATI : ' + str(correct_age))
print('SBAGLIATI  : ' + str(wrong_age))
print('NON TROVATI: ' + str(unrecognized))
print('TOTALI     : ' + str(correct_age + wrong_age + unrecognized))
print('CONFUSION MATRIX')
print(cmarray)
print('TRUE POSITIVES')
print(TruePositive)
print('FALSE POSITIVES')
print(FalsePositive)
print('FALSE NEGATIVES')
print(FalseNegative)
print('TRUE NEGATIVES')
print(TrueNegative)
print('ACCURACY')
print(Accuracy)
plt.show()