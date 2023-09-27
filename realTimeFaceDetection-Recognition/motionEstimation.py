import numpy as np
import os
import cv2
import math
import time

class FaceDetection():
    def __init__(self):
        self.face_cascasde_frontal = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.face_cascasde_profile = cv2.CascadeClassifier('profile.xml')
        self.eye_cascade=cv2.CascadeClassifier('eye.xml')
        self.frame_count = -1
        self.cache={}
        self.prev_frame=None

    def face_detection(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frame_count += 1
            if self.frame_count % 10 == 0 :
                faces_frontal = self.face_cascasde_frontal.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces_frontal:
                    grayRec=cv2.rectangle(gray,(x,y),(x+w,y+h), (255,0,0), 2)
                    self.cache={'x':x, 'y':y, 'w':w, 'h':h}
                    self.prev_frame = gray

            else:
                t21=time.time()
                [n1, n2]=self.motion_estimation(gray,self.prev_frame, 16)
                print('-----------------------')
                x=self.cache['x']
                y =self.cache['y']
                w=self.cache['w']
                h=self.cache['h']
                gray=cv2.rectangle(gray,(x+n1,y+n2),(x+n1+w,y+n2+h),(255,255,100),5)
                cv2.imshow('Predicted',gray)
                cv2.waitKey(1)
                self.prev_frame=gray
                x=x+n1
                y=y+n2
                self.cache = {'x': x, 'y': y, 'w': w, 'h': h}
                t22=time.time()
                print('Total Time for Else : '+str(t22-t21))



    def motion_estimation(self,frame_current,frame_prev,W):
        minMSE = math.inf
        x = self.cache['x']
        y = self.cache['y']
        w = self.cache['w']
        h = self.cache['h']
        motion_vector=[0, 0]
        t11 = time.time()
        p1 = frame_current.astype(np.int16)
        p2 = frame_prev.astype(np.int16)
        for n1 in range(-W, W+1):
            for n2 in range(-W, W+1):

                MSE = (np.sum(np.sum(np.absolute(p1[y+n2:y+h+n2, x+n1:x+w+n1] - p2[y:y+h, x:x+w]))))/(w*h)
                if MSE < minMSE:
                    minMSE = MSE
                    motion_vector = [n1, n2]
        t12 = time.time()
        print('Time for SME : ' + str(t12 - t11))

        return motion_vector



faceDetector = FaceDetection()
faceDetector.face_detection()
