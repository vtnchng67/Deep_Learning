import numpy as np
import cv2
import os
import glob
import faceRecognition as fr
print (fr)
path = glob.glob("E:/Deep Learning/tfmodel/test01/01 (1).JPG")
for file in path:
    test_img=cv2.imread(file) 
    faces_detected,gray_img=fr.faceDetection(test_img)
    print("face Detected: ",faces_detected)


    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(r'E:\Deep learning\tfmodel\gender_detection.model') #Give path of where trainingData.yml is saved

    name={0:"man",1:"woman"} #Change names accordingly. If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1.

    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+h,x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)
        print ("Confidence :",confidence)
        print("label :",label)
        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        fr.put_text(test_img,predicted_name,x,y)    
    resized_img=cv2.resize(test_img,(1000,700))
    cv2.imshow("face detection ", resized_img)
    if cv2.waitKey(2000) & 0xFF == ord('q'):
      break
  
    
cv2.destroyAllWindows()
    




