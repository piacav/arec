from typing import List, Union
from sys import platform

import cv2

# Initialization variables
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_class = [14, 24, 59, 100]
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']
classes = {1: "Young",
            2: "Teen",
            3: "Adult",
            4: "Old", }
sbagliati, indovinati, unrecognized = 0, 0, 0
test_images = []

# Specify the Haar classifier
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_alt.xml')

# Specify font used for plotting
font = cv2.FONT_HERSHEY_SIMPLEX

# Creation dataset path for different O.S.
if platform == 'win32':
    dataset_path = 'C:\\Users\\andry\\Desktop\\FGNET\\images\\'
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
    elif max <= age_class[2]:
        return 3
    return 4

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
    if faces_rect == ():
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

        # Verification
        l = len(test_images)
        if classdet == real_class:
            indovinati += 1
        else:
            sbagliati += 1

        # Visualize image
        #overlay_text = "%s %s" % (gender, age)
        #cv2.putText(image_copy, overlay_text, (0, 30), font, 1, (255, 0, 0), 2)
        #cv2.imshow('image', image_copy)
        #cv2.waitKey(0)

print('INDOVINATI : ' + str(indovinati))
print('SBAGLIATI  : ' + str(sbagliati))
print('NON TROVATI: ' + str(unrecognized))
print('TOTALI     : ' + str(indovinati + sbagliati + unrecognized))