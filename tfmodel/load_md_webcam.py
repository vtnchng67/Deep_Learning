import tensorflow as ts
from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# load model
model = load_model('gender_detection.model')

# mở webcam
webcam = cv2.VideoCapture('E:/Deep learning/tfmodel/test01/vd4.mp4')
    #E:/Deep learning/tfmodel/test01/vd4.mp4
classes = ['man','woman']

# loop through frames
while webcam.isOpened():

    # đọc frame từ webcam
    status, frame = webcam.read()

    # áp dụng face detection
    face, confidence = cv.detect_face(frame)


    # lặp qua các khuôn mặt được phát hiện
    for idx, f in enumerate(face):

        # lấy điểm góc của hình chữ nhật khuôn mặt
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # vẽ hình chữ nhật trên khuôn mặt
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # cắt vùng khuôn mặt được phát hiện
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # tiền xử lý cho mô hình phát hiện giới tính
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # áp dụng gender detection trên gương mặt
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # lấy nhãn với độ chính xác tối đa
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # viết nhãn và độ tự tin trên khuôn mặt hình chữ nhật
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # xuất ra màn hình
    cv2.imshow("gender detection", frame)

    # nhấn "Q" để dừng
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release tài nguyên
webcam.release()
cv2.destroyAllWindows()