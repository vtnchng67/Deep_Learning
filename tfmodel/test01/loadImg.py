import tensorflow as ts
from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
import glob
model = load_model('E:/Deep Learning/tfmodel/gender_detection.model')

classes = ['man', 'woman']
path = glob.glob("E:/Deep Learning/tfmodel/test01/01 (1).JPG")

for file in path:
    test_img = cv2.imread(file)
    for idx, f in enumerate(test_img):

        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(test_img, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(test_img[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.imshow("gender detection", test_img)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()