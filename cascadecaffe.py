from typing import List, Union
from sys import platform
import numpy as np
import cv2
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

# Initialization variables
rigacm, colonnacm = None, None
sbagliati, indovinati, unrecognized = 0, 0, 0
confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_class = [10, 50, 100]
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']
classes = {1: "Young",
            2: "Teen",
            3: "Adult"}
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
        gender = gender_list[gender_preds[0].argmax()]
        print("Gender    : " + gender)

        # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age: Union[str, List[str]] = age_list[age_preds[0].argmax()]
        print("Age Range : " + age)

        # Define class
        classdet = classes.get(classifier_age(age))
        print("Class     : " + classdet)

        # Define real age and class
        real_age = test_image[4:6]
        print("Real Age  : " + real_age)

        real_class = classes.get(classifier_age(int(real_age)))
        print("Real Class: " + real_class)

        print("Score Age : " + str(age_preds[0]))
        print("Score Gen : " + str(gender_preds[0]))
        print('#######################')

        # Verification age
        if classdet == real_class:
            indovinati += 1
            for key, item in classes.items():
                if item == classdet:
                    confusion_matrix[key - 1][key - 1] += 1
                    break
        else:
            sbagliati += 1
            for wkey, witem in classes.items():
                if witem == real_class:
                    rigacm = wkey
                if witem == classdet:
                    colonnacm = wkey
            if rigacm is not None and colonnacm is not None:
                confusion_matrix[rigacm - 1][colonnacm - 1] += 1

        # Visualize image
        #overlay_text = "%s %s" % (gender, age)
        #cv2.putText(image_copy, overlay_text, (0, 30), font, 1, (255, 0, 0), 2)
        #cv2.imshow('image', image_copy)
        #cv2.waitKey(0)

# Compute metrics for performance evaluation
cmarray = np.array(confusion_matrix)

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
multiclass = np.array(confusion_matrix)
fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                colorbar=True,
                                class_names=classes.items())

print('INDOVINATI : ' + str(indovinati))
print('SBAGLIATI  : ' + str(sbagliati))
print('NON TROVATI: ' + str(unrecognized))
print('TOTALI     : ' + str(indovinati + sbagliati + unrecognized))
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