import cv2,os
import numpy as np
import pickle

from PIL import Image


faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);

rec=cv2.face.LBPHFaceRecognizer_create();



rec.read("recognizer//trainingData.yml")
id=0
font = cv2.FONT_HERSHEY_SIMPLEX\

     
fontScale = 1
fontColor = (255 )

ret,img=cam.read();
 
locy = int(img.shape[0]/2) # the text location will be in the middle
locx = int(img.shape[1]/2) #           of the frame for this example

name ="ünknown"

#font=cv2.cv.Initfont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)

while(True):
        _,img=cam.read();
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(gray,1.3,5);
        for(x,y,w,h) in faces:
                cv2.imwrite("person.jpg",img)
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                id,conf=rec.predict(gray[y:y+h,x:x+w])
                if (conf<60):
                        if id == 1:
                                name = "sanjiv"
                                print ("sanjiv")
                                

                else:
                                name ="ünknown"
                                print ("unknown")
        
                                
               # cv2.putText(img,str(name),(x,y+h),font,2,(0,255,0), 2)
               #cv2.cv.putText(cv2.cv.fromarray(img),str(name),(x,y+h),font,255)
                cv2.putText(img, str(name), (locx, locy), font, fontScale, fontColor) 
        cv2.imshow("Face",img);
        if(cv2.waitKey(1)==ord('q')):
                break;

cam.release()
cv2.destroyAllWindows()
