# Face recognition

import cv2

# create object (Loading cascades)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# face detection func
def detect(gray, original): # params are gray image and original image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

