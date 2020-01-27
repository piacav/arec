import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
#img.shape per misure img
img_raw = cv2.imread('/Users/piacavasinni/Desktop/FGNET/images/001A29.jpg')
img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

class_img = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


faces_rects = class_img.detectMultiScale(img_gray, scaleFactor = 1.2, minNeighbors = 5);
print('Faces found: ', len(faces_rects))

for (x,y,w,h) in faces_rects:
     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

plt.imshow(img)

#v2.rectangle(img,(50, 0), (100,100), (255, 0, 0), 5)

#plt.imshow(img)
plt.show()

'''
#test_image = cv2.imread('/Users/piacavasinni/Desktop/FGNET/images/001A29.jpg')
test_image = cv2.imread('/Users/piacavasinni/Desktop/FotoDB/FOTO4.png')

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_eye.xml')


def detect_faces(cascade, test_image, scaleFactor=1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    # convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 3)

    return image_copy


img_raw = detect_faces(cascade, test_image, 1.1)
img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
