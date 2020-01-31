import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sys import platform

# Create training set and test set
if platform == 'win32':
    dataset_path = 'C:\\Users\\andry\\Desktop\\FGNET\\images\\'
elif platform == 'darwin':
    dataset_path = '/Users/piacavasinni/Desktop/FGNET/images2/'
else:
    dataset_path = ''

dataset_list = []

for file in os.listdir(dataset_path):
    if not file.startswith('.'):
        persona = int(file[:3])
        dataset_list.append(file[:-4])


cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_alt.xml')

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
age_list2 = ['(0, 15)', '(17, 30)', '(32, 50)', '(55, 65)', '(70, 100)']
gender_list = ['Male', 'Female']
font = cv2.FONT_HERSHEY_SIMPLEX

def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('genderage/deploy_age.prototxt', 'genderage/age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('genderage/deploy_gender.prototxt', 'genderage/gender_net.caffemodel')
    return (age_net, gender_net)

print(dataset_list)
for test_image in dataset_list:
    age_net, gender_net = load_caffe_models()

    imagePath = dataset_path + test_image + '.jpg'
    ti = cv2.imread(imagePath)

    # create a copy of the image to prevent any changes to the original one.
    image_copy = ti.copy()
    # convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, 1.1, 5)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face_img = image_copy[y:y + h, h:h + w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        print("Gender : " + gender)

        # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        print("Age Range: " + age)
        print(test_image )
        overlay_text = "%s %s" % (gender, age)
        cv2.putText(image_copy, overlay_text, (0, 30), font, 1, (255, 0, 0), 2)
        cv2.imshow('image', image_copy)
        cv2.waitKey(0)