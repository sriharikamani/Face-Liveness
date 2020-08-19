# ### Find and recognize unknown faces from known people faces using face_recognition API
#Seps involved are

#1. Find face in a image
#2. Preprocess image
#   Converting it to grayscale
#   Resizing the image
#   CLAHE (Contrast Limited Adaptive Histogram Equalization) to smooth out lighting differences
#   Align
#3. Analyze facial features and capture 128 basic measurements from each face using CNN
#4. Compare each unknown person against known faces measurements captured and get a euclidean distance having the closest   
#   measurement
#5. The distance tells how similar the faces are and that’s our match!

import numpy as np
import pandas as pd
import face_recognition
from PIL import Image,ImageEnhance #, ImageDraw
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import os
import cv2
import dlib

CASE_PATH = "C:/Users/sriha/Documents/My_Data/Work_Documents/Posidex/face_detection_emotion/Data/haarcascade_frontalface_default.xml"
path      = "C:/Users/sriha/Documents/My_Data/Work_Documents/Posidex/face_rec/face_rec-blink/"
os.chdir(path)

################## Face detectors

hog_face_detector           = dlib.get_frontal_face_detector()
face_classifier             = cv2.CascadeClassifier(CASE_PATH)
predictor                   = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa                          = FaceAligner(predictor, desiredFaceWidth=256)

known_face_encodings        = []
known_face_names            = []
known_face_encode_error     = []

os.chdir(path + "dataset/knownImages/")

# #### Load all known images from drive to get basic 128 measurements with upsample = 1

def initial():
    
    global known_face_encode_error
    global known_face_encodings
    global known_face_names
    
    known_face_encodings = []
    known_face_names     = []
    Upsample             = 1

    for dirpath, dnames, knownimages in os.walk(path +"dataset/knownImages/"):
        for knownimage in knownimages:
            img   = face_recognition.load_image_file(path + "dataset/knownImages/" + knownimage)
            known_image_encode(img,Upsample,knownimage)

    # Try capturing basic facial measurements for failed images by incrementing the Upsample. Repeat for 4 iterations
    # Repeat for 3 iterations. Upsampling the image helps to detect smaller faces

    repeat = 1
    while ( (len(known_face_encode_error) > 0) and (repeat <=3)):

        known_face_encode_error_bk  = known_face_encode_error
        known_face_encode_error     = []
        Upsample                    = Upsample + 1
        
        for i in range(len(known_face_encode_error_bk)):
            img   = face_recognition.load_image_file(path + "dataset/knownImages/" + known_face_encode_error_bk[i])
            known_image_encode(img,Upsample,known_face_encode_error_bk[i])

        repeat += 1
        
    return(known_face_encodings,known_face_names)

# ### Routine to capture basic 128 measurements (known as embedding) for each known face images 
def known_image_encode(img,Upsample,knownimage):
    
    global known_face_encode_error
    global known_face_encodings
    global known_face_names
    
    img   = imutils.resize(img, width=400)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,8)) # Divide image into small blocks called “tiles” (Size is 8x8 by default in OpenCV)
    gray  = clahe.apply(gray)
    #res   = np.hstack((gray,clahe)) #stacking images side-by-side
    gray  = cv2.bilateralFilter(gray,5,75,75) # bilateral filter d = 15 and sigmaColor = sigmaSpace = 75. The greater its value, the more further pixels will mix together, given that their colors lie within the sigmaColor range
    rects = hog_face_detector(gray, Upsample) # Upsampling each image to detect faces
    if (len(rects) > 0):

        for rect in rects:
            faceAligned   = fa.align(img, gray, rect)
            try:
                face_encoding = face_recognition.face_encodings(faceAligned)[0]  # Generate basic measurements
                known_face_encodings.append(face_encoding)
                known_face_names.append(knownimage)
            except Exception as e:
                known_face_encode_error.append(knownimage)
            break
    else:
        known_face_encode_error.append(knownimage)

    return()

if __name__ == '__main__':
    known_face_encodings,known_face_names = initial()

#### End of the Code  ###