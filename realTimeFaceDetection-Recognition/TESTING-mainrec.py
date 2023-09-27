import cv2
import numpy as np
import os
import random
from sklearn.neighbors import KNeighborsClassifier
from LDGP import Calculator
import time
#from mtcnn import MTCNN
#from keras.models import Sequential
#from keras.layers import Dense
#from sklearn.preprocessing import OneHotEncoder

# modelNN = Sequential()
# modelNN.add(Dense(12, input_dim=37288,  activation='relu'))
# modelNN.add(Dense(8,activation='relu'))
# modelNN.add(Dense(3,activation='softmax'))
# modelNN.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# ohe=OneHotEncoder()


class FaceRecognition:
    def __init__(self,block_size, stride):
        self.desc = Calculator(block_size)
        #self.detector = MTCNN()
        self.stride = stride
        self.block_size=block_size

    def trainRecognizertesting(self, trainPath, testPath , no_of_classes, printAcc=True ):
        data = []
        labels = []
        acc=0
        print('Extracting Features.....')
        for imageFolder in os.listdir(trainPath):
            count = 0

            imagePath = os.path.join(trainPath, imageFolder)
            for trainImg in os.listdir(imagePath):
                #count+=1
                image = cv2.imread(os.path.join(imagePath, trainImg))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                t1=time.time()
                hist = self.desc.calc_hist(gray, self.stride)
                t2=time.time()
                #print(t2-t1)
                label=imageFolder[-2]+imageFolder[-1]
                labels.append(int(label))
                data.append(hist)
                #if count>5:
                   # break
            print("Processed folder " + imageFolder)
            if no_of_classes <= 9:
                stopFolder = "Subject0"+str(no_of_classes)
            else:
                stopFolder = "Subject"+str(no_of_classes)
            if imageFolder == stopFolder:
                break
        print('Completed Feature Extraction!')
        print('Training Classifier.......')
        temp = list(zip(data, labels))
        random.shuffle(temp)
        data, labels = zip(*temp)
        model = KNeighborsClassifier(n_neighbors = 1)
        model.fit(data, labels)
        print('Done Training KNN')
        if printAcc:
            correct = 0
            total = 0
            print('Calculating accuracy....')
            for imageFolder in os.listdir(testPath):
                imagePath = os.path.join(testPath, imageFolder)
                for testImg in os.listdir(imagePath):
                    image = cv2.imread(os.path.join(imagePath, testImg))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    hist = self.desc.calc_hist(gray, self.stride)
                    hist = np.array(hist)
                    prediction = model.predict(hist.reshape(1, -1))
                    total += 1
                    label = imageFolder[-2] + imageFolder[-1]
                    if prediction == int(label):
                        correct += 1
                print("Done "+imageFolder)
                tempacc = (correct/total)*100
                #print("Total : "+str(total)+" Correct : "+str(correct)+" Accuracy : "+str(tempacc))

                if no_of_classes <= 9:
                    stopFolder = "Subject0" + str(no_of_classes)
                else:
                    stopFolder = "Subject" + str(no_of_classes)
                if imageFolder == stopFolder:
                    break
            acc = (correct/total)*100
            #print('Accuracy on test set is : '+str(acc)+'%')
        return model, acc






