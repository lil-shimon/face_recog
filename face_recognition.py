# Face recognition

import cv2

# create object (Loading cascades)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# face detection func
def detect(gray, original): # params are gray image and original image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # x, y, w, h
    for (x, y, w, h) in (faces):
        # call recognition func from opencv 
        # first arg : image
        # second arg: cordinate upper left angle
        # third arg : low right left angle
        # fourth arg : colour
        # last arg : edges of rectangle
        cv2.rectangle(original, (x, y), (x+w, y+h), (255, 0, 0), 2)

