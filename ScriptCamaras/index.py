# import tensorflow as tf
# from tensorflow.keras.models import load_model
import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import os

# DIR_KNOWNS = 'knowns'
# DIR_UNKNOWNS = 'unknowns'
# DIR_RESULTS = 'results'

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('img.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceClassif.detectMultiScale(gray,
                                     scaleFactor = 1.2,
                                     minNeighbors = 4,
                                    #  minSize = (30,30),
                                    #  maxSize=(200,200)
                                    )

for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0),2)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()