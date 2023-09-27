import numpy as np
import cv2
import math
# import time
import pickle
# from sklearn.neighbors import KNeighborsClassifier
from faceDet import FaceRecognition


class face_package:
    def __init__(self, block_size, stride, frame_skip):
        # self.face_cascade_frontal = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # self.face_cascade_profile = cv2.CascadeClassifier('profile.xml')
        # self.eye_cascade = cv2.CascadeClassifier('eye.xml')
        self.detect = FaceRecognition(block_size, stride)
        self.frame_skip = frame_skip
        self.frame_count = -1
        self.cache = {}
        self.prev_frame = None

    def face_system_entire(self):
        pickle_in = open("savedmodel.pickle", "rb")
        model = pickle.load(pickle_in)
        cap = cv2.VideoCapture(0)
        predict_estimate = None
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frame_count += 1
            if self.frame_count % self.frame_skip == 0:
                faces = self.detect.find_face(frame)
                if faces == 0:
                    image = cv2.putText(frame, "No Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Frame', image)
                    cv2.waitKey(1)
                    continue
                for boundary in faces:
                    (x, y, w, h) = boundary
                    disp = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    grayRec = gray[y:y + h, x:x + w, :]
                    grayRec = cv2.resize(grayRec, (360, 480))
                    predict = self.detect.predictImg(grayRec, model)
                    predict_estimate = str(predict)  # Save label for later
                    disp = cv2.putText(disp, str(predict), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                       1, (255, 0, 0), 2, cv2.LINE_AA)
                    self.cache = {'x': x, 'y': y, 'w': w, 'h': h}
                    self.prev_frame = gray
                    cv2.imshow('Prediction', disp)
                    cv2.waitKey(1)
            else:
                # t21 = time.time()
                [n1, n2] = self.motion_estimation(gray, self.prev_frame, 16)
                # print('-----------------------')
                x = self.cache['x']
                y = self.cache['y']
                w = self.cache['w']
                h = self.cache['h']
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mid_frame = cv2.rectangle(frame, (x+n1, y+n2), (x+n1+w, y+n2+h), (255, 255, 100), 5)
                mid_frame = cv2.putText(mid_frame, predict_estimate, (x+n1, y+n2), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('Estimation', mid_frame)
                cv2.waitKey(1)
                self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                x = x+n1
                y = y+n2
                self.cache = {'x': x, 'y': y, 'w': w, 'h': h}
                # t22 = time.time()
                # print('Total Time for Else : '+str(t22-t21))

    def motion_estimation(self, frame_current, frame_prev, W):
        minMSE = math.inf
        x = self.cache['x']
        y = self.cache['y']
        w = self.cache['w']
        h = self.cache['h']
        motion_vector = [0, 0]
        # t11 = time.time()
        p1 = frame_current.astype(np.int16)
        p2 = frame_prev.astype(np.int16)
        for n1 in range(-W, W+1):
            for n2 in range(-W, W+1):
                MSE = (np.sum(np.sum(np.absolute(p1[y+n2:y+h+n2, x+n1:x+w+n1] - p2[y:y+h, x:x+w]))))/(w*h)
                if MSE < minMSE:
                    minMSE = MSE
                    motion_vector = [n1, n2]
        # t12 = time.time()
        # print('Time for SME : ' + str(t12 - t11))
        return motion_vector


system = face_package(16, 2, 5)
system.face_system_entire()
