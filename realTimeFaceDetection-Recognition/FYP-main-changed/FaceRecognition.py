import cv2
import numpy as np
import os
import random
from sklearn.neighbors import KNeighborsClassifier
from LDGP import Calculator
import time
from mtcnn import MTCNN


class FaceRecognition:
    def __init__(self,block_size):
        self.desc=Calculator(block_size)
        self.detector=MTCNN()

    def predictPath(self, path, model):
        image=cv2.imread(path,0)
        hist = self.desc.calc_hist(image[40:-70, 120:-180])
        hist = np.array(hist)
        prediction = model.predict(hist.reshape(1, -1))
        return prediction

    def predictImg(self,img,model):  # Input : Gray Image; Output : Face Prediction
        #image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        hist = self.desc.calc_hist(img)
        hist = np.array(hist)
        prediction = model.predict(hist.reshape(1, -1))
        return prediction

    def find_face(self,img):          # Input : BGR Image; Output : GRAY Image with bounded faces and resized to 360x480
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(imgRGB)
        if len(faces)==0:
            return np.asarray([0,0,0])
        
        i = 0
        memo = {}
        for face in faces:
            key = "Face" + str (i)
            x1, y1, w1, h1 = face['box']
            # cv2.rectangle(imgRGB, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 3)
            memo[key] = [x1, y1, w1, h1]
            #memo.add (key, [x1, y1, w1, h1])
            #imgRGB = imgRGB[y1:y1 + h1, x1:x1 + w1, :]
            #imgfinal = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)
            i += 1    
            ##break
        #imgfinal=cv2.resize(imgfinal,(360,480))
        return memo
        #return imgfinal

    def captureData(self,path):
        counter=0
        i = 0
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        while True:

            ret, img = cap.read()
            if counter%7==0:
                imgFinal = self.find_face(img)
                if not imgFinal is None:
                    filename=str(i)+'.jpeg'
                    cv2.imwrite(os.path.join(path,filename),imgFinal)
                    i += 1
                    print('No. of images : '+ str(i))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            counter+=1

    def trainRecognizer(self, trainPath, printAcc, testPath = None):
        data = []
        labels = []
        print('Extracting Features.....')
        for imageFolder in os.listdir(trainPath):
            imagePath = os.path.join(trainPath, imageFolder)
            for trainImg in os.listdir(imagePath):
                image = cv2.imread(os.path.join(imagePath, trainImg))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #print(gray[50:-70,130:-170].shape)
                hist = self.desc.calc_hist(gray)
                labels.append(int(imageFolder[-1]))
                data.append(hist)
            print("Processed folder " + imageFolder)
            if imageFolder == "Subject09":
                break
        print('Completed Feature Extraction!')
        print('Training Classifier.......')
        temp = list(zip(data, labels))
        random.shuffle(temp)
        data, labels = zip(*temp)
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(data, labels)
        print('Done Training')
        if printAcc:
            correct = 0
            total = 0
            print('Calculating accuracy....')
            for imageFolder in os.listdir(testPath):
                imagePath = os.path.join(testPath, imageFolder)
                for testImg in os.listdir(imagePath):
                    image = cv2.imread(os.path.join(imagePath, testImg))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    hist = self.desc.calc_hist(gray)
                    hist = np.array(hist)
                    prediction = model.predict(hist.reshape(1, -1))
                    total += 1
                    if prediction == int(imageFolder[-1]):
                        correct += 1
                print("Done "+imageFolder)
                tempacc = (correct/total)*100
                print("Total : "+str(total)+" Correct : "+str(correct)+" Accuracy : "+str(tempacc))
                if imageFolder == "Subject09":
                    break
            acc = (correct/total)*100
            print('Accuracy on test set is : '+str(acc)+'%')
        return model








