import requests
import wget
import cv2
import os

face_url = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect'
image_url = "https://scontent-fco1-1.cdninstagram.com/v/t51.2885-15/sh0.08/e35/p640x640/79352806_584020185765626_2889454158845427787_n.jpg?_nc_ht=scontent-fco1-1.cdninstagram.com&_nc_cat=1&_nc_ohc=a0W9Al5Kq6wAX88enwo&oh=9cf04dd716a26a7bc2203f951bc343c1&oe=5ED368A1"
headers = {"Ocp-Apim-Subscription-Key": "00039e60551b48f7b5cdd977f95e9d5c"}
params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'age,gender',
    'recognitionModel': 'recognition_01',
    'returnRecognitionModel': 'false',
    'detectionModel': 'detection_01'
}

response = requests.post(face_url, params=params, headers=headers, json={"url": image_url})
resjson = response.json()[0]
localimg = wget.download(image_url)

face_rect = resjson['faceRectangle']
x = face_rect['left']
y = face_rect['top']
w = face_rect['width']
h = face_rect['height']
age = int(resjson['faceAttributes']['age'])
gender = resjson['faceAttributes']['gender']

img_resp = requests.get(image_url, stream=True)
image = cv2.imread(localimg)
image_copy = image.copy()
cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
face_img = image_copy[y:y + h, h:h + w]
font = cv2.FONT_HERSHEY_SIMPLEX

overlay_text = "%s %s" % (gender, age)
cv2.putText(image_copy, overlay_text, (0, 30), font, 1, (255, 0, 0), 2)
cv2.imshow('image', image_copy)
cv2.waitKey(0)
os.remove(localimg)
