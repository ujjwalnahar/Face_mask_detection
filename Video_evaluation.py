#importing modules
print(1)
from tensorflow.keras.models import model_from_json
print(1)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
print(1)
import cv2
print(1)
import numpy as np
print(1)
import matplotlib.pyplot as plt
print(1)
json_file = open('model.json', 'r')
print(2)
loaded_model_json = json_file.read() #reading the json file
json_file.close()
loaded_model = model_from_json(loaded_model_json)
print(3)
loaded_model.load_weights("model.h5")  # load weights into new model
print(4)
face_clsfr=cv2.CascadeClassifier('face_xml.xml') #Loading the face Detector classifier
source=cv2.VideoCapture(0) #starting the video feed of our web cam 
labels_dict={0:'MASK',1:'NO MASK'}     #Labelling the categories  
color_dict={0:(0,255,0),1:(0,0,255)}   #Defining colors for categories

mask=0
print(5)
while(True):
    ret,img=source.read()
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray_img,1.05,5)  
    label=0

    for (x,y,w,h) in faces:
    
        face_img=img[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(224,224))
        new_img=preprocess_input(resized)
        reshaped=np.reshape(new_img,(1,224,224,3))
        result=loaded_model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==ord('q')):
        break
cv2.destroyAllWindows()
source.release()
