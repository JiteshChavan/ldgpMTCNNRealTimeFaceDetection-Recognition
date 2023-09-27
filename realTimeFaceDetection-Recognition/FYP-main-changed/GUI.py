import cv2
from FaceRecognition import FaceRecognition
from tkinter import *

trainPath=""
testPath=""
printAcc=True
import pickle

def click():
    trainPath=train.get()
    testPath=test.get()
    print(trainPath, testPath)
    faceRec = FaceRecognition(8)
    model = faceRec.trainRecognizer(trainPath=trainPath, printAcc=printAcc, testPath=testPath)
    pickel_out=open("savedmodel.pickle","wb")
    pickle.dump(model,pickel_out)
    pickel_out.close()
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # while True:
    #     ret, img = cap.read()
    #     imgFaces = faceRec.find_face(img)
    #     if imgFaces.any()==None:
    #         break
    #     prediction=faceRec.predictImg(imgFaces, model)
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     org = (50, 50)
    #     fontScale = 1
    #     color = (255, 0, 0)
    #     thickness = 2
    #     image = cv2.putText(img, str(prediction), org, font,
    #                         fontScale, color, thickness, cv2.LINE_AA)
    #     cv2.imshow('frame', image)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    close_window()


def click2():
    faceRec=FaceRecognition(8)
   # trainmodelpath=modelpath.get()
    pickle_in=open("savedmodel.pickle","rb")
    model = pickle.load(pickle_in)
    cap = cv2.VideoCapture(0)
    
    while True :
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        ret, img = cap.read()
        if ret:

            #dict length = num faces
            
            faceMemo = faceRec.find_face(img)
            faceCount = len (faceMemo.keys ()) 
            #if imgFaces.all()==0:
            if faceCount==0:
                print('No Face Found!')
                image = cv2.putText(img,"None", org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
                cv2.imshow('frame', image)
                cv2.waitKey(1)
                continue
            
            #iterate over length dict
            predictions = []
            for faceIndex in range(faceCount):
                imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                key = "Face" + str (faceIndex)
                x1,y1,w1,h1 = faceMemo[key]
                imageFace = imgGray[y1:y1 + h1, x1:x1 + w1]
                imageFace=cv2.resize(imageFace,(360,480))
                predictions.append(faceRec.predictImg(imageFace, model)) 

                orgTemp = (x1, y1)
                image = cv2.putText(img, str(predictions[-1]), orgTemp, font,
                                fontScale, color, thickness, cv2.LINE_AA)
            
            cv2.imshow('frame', image)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    close_window()


def close_window():
    window.destroy()
def store_selection():
    printAcc=var1.get()


window = Tk()
window.title("Face Recognition")
window.configure(background="black")
Label(window, text="Welcome to Real Time Face Detection ", bg="black", fg="white", font="none 12 bold"). grid (row=1, column=1, sticky=W)
Label(window, text="Enter train path : ", bg="black", fg="white", font="none 12 bold"). grid (row=2, column=0, sticky=W)
train = Entry(window, width=20, bg="white")
train.grid(row=2, column=1, sticky=W)
Label(window, text="Enter test path : ", bg="black", fg="white", font="none 12 bold"). grid (row=4, column=0, sticky=W)
test = Entry(window, width=20, bg="white")
test.grid(row=4, column=1, sticky=W)
var1 = IntVar()
c1 = Checkbutton(window, text="Print Accuracy?",variable=var1, onvalue=True, offvalue=False, command=store_selection)
c1.grid(row=5)
Button(window, text="Train Model", width=6, command=click) .grid(row=6)

Label (window, text="Use a pretrained model. Enter the path : ", bg="black", fg="white", font="none 12 bold"). grid (row=7, column=0, sticky=W)
modelpath= Entry(window, width=20, bg="white")
Button(window, text="Use trained model", width=6, command=click2) .grid(row=8)

window.mainloop()
