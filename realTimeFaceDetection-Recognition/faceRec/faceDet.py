import cv2
import numpy as np
from testldgp import Calculator
from mtcnn import MTCNN


class FaceRecognition:
    def __init__(self, block_size, stride):
        self.desc = Calculator(block_size, stride)
        self.detector = MTCNN()

    def predictImg(self, img, model):  # Input : Gray Image; Output : Face Prediction
        hist = self.desc.calc_hist(img)
        hist = np.array(hist)
        prediction = model.predict(hist.reshape(1, -1))
        return prediction

    def find_face(self, img):         # Input : BGR Image; Output : box boundaries
        face_bound = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(imgRGB)
        if len(faces) == 0:
            return 0
        for face in faces:
            xf, yf, wf, hf = face['box']
            face_bound.append((xf, yf, wf, hf))
            # cv2.rectangle(imgRGB, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 3)
            # img = imgRGB[yf:yf + hf, xf:xf + wf, :]
            # imgfinal = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # break
        # imgfinal=cv2.resize(imgfinal,(360,480))
        return face_bound
