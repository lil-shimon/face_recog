# Face recognition

import cv2

# create object (Loading cascades)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# face detection func
def detect(gray, original): # params are gray image and original image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # x, y, w, h
    for (x, y, w, h) in faces:
        # call rectangle func from opencv 
        # first arg  : image
        # second arg : cordinate upper left angle
        # third arg  : low right left angle
        # fourth arg : colour
        # last arg   : edges of rectangle
        cv2.rectangle(original, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # region of interests
        roi_g = gray[y:y+h, x:x+w]
        roi_colour = original[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_g, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_colour, (ew,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return original # original image w/ the rectangles detecting the face and eyes

