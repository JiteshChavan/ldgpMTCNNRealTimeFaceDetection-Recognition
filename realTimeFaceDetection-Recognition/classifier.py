import os
import cv2
import numpy as np
from LDGP import Calculator

trainPath = "G:\\train"
testPath = "G:\\test"
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
desc = Calculator(8)
data = []
#labels = []

for imageFolder in os.listdir(trainPath):
    imagePath = os.path.join(trainPath, imageFolder)
    for trainImg in os.listdir(imagePath):
        gray = cv2.imread(os.path.join(imagePath, trainImg), 0)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            images = cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
            imageFac = cv2.resize(images, (360, 480))
        hist = desc.calc_hist(imageFac)
        labels.append(int(imageFolder[-2:]))
        data.append(hist)
        
    print("Processed folder " + imageFolder)
    if imageFolder == "Subject03":
        break

   # For a label L, avg of the histograms
#avgHist={#Classes}
        
        
for imageFolder in os.listdir(testPath):
    imagePath = os.path.join(testPath, imageFolder)
    for testImg in os.listdir(imagePath):
        gray = cv2.imread(os.path.join(imagePath, testImg), 0)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            images = cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
            imageFac = cv2.resize(images, (360, 480))
        hist = desc.calc_hist(imageFac)
        hist = np.array(hist, dtype=np.float32)
        for i in range(len(data)):
            hist2 = data[i]
            hist2 = np.array(hist2, dtype=np.float32)
            #for i in range ():
            v = cv2.compareHist(hist, hist2, cv2.HISTCMP_CORREL)
            
            print(v)
    print("Processed folder " + imageFolder)
    if imageFolder == "Subject03":
        break
