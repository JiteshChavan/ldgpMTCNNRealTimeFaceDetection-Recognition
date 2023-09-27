import numpy as np
import cv2
from mainRec import FaceRecognition
import pickle
from LDGP import Calculator
import time


#---------------------------------------------------------------DataSet 1-------------------------------------------------------------------------#
trainPath="D:\\8th Semester\\FYP\Dataset1\\training"
testPath="D:\\8th Semester\\FYP\\Dataset1\\testing"
# # ----------------------------Stride = 2 ----------------------------------------------#
# # --- 15 Classes --------#
# FaceRec1=FaceRecognition(8,2)
# m1, acc1DS1= FaceRec1.trainRecognizertesting(trainPath, testPath, 15)
# print("DATASET-1\nNo. of classes =15\tStride=2\nAccuracy = "+str(acc1DS1))
# print("##############################################################")
# # --- 20 Classes --------#
# FaceRec2=FaceRecognition(8,2)
# m2, acc2DS1= FaceRec2.trainRecognizertesting(trainPath, testPath, 20)
# print("DATASET-1\nNo. of classes =20\tStride=2\nAccuracy = "+str(acc2DS1))
# print("##############################################################")
#
#
# # ----------------------------Stride = 3 ----------------------------------------------#
# # --- 15 Classes --------#
# FaceRec3=FaceRecognition(8,3)
# m4, acc3DS1= FaceRec3.trainRecognizertesting(trainPath, testPath, 15)
# print("DATASET-1\nNo. of classes =15\tStride=3\nAccuracy = "+str(acc3DS1))
# print("##############################################################")
#
# # --- 20 Classes --------#
# FaceRec4=FaceRecognition(8,3)
# m5, acc4DS1= FaceRec4.trainRecognizertesting(trainPath, testPath, 20)
# print("DATASET-1\nNo. of classes =20\tStride=3\nAccuracy = "+str(acc4DS1))
# print("##############################################################")
#
#
# # ----------------------------Stride = 4 ----------------------------------------------#
# # --- 15 Classes --------#
# FaceRec5=FaceRecognition(8,4)
# m6, acc5DS1= FaceRec5.trainRecognizertesting(trainPath, testPath, 15)
# print("DATASET-1\nNo. of classes =15\tStride=4\nAccuracy = "+str(acc5DS1))
# print("##############################################################")
#
# # --- 20 Classes --------#
# FaceRec6=FaceRecognition(8,4)
# m7, acc6DS1= FaceRec6.trainRecognizertesting(trainPath, testPath, 20)
# print("DATASET-1\nNo. of classes =20\tStride=4\nAccuracy = "+str(acc6DS1))
# print("##############################################################")
#
# #------------------------------------------------------DataSet 2--------------------------------------------------------------------#
# trainPath="D:\\8th Semester\\FYP\Dataset2\\training"
# testPath="D:\\8th Semester\\FYP\\Dataset2\\testing"
# # ----------------------------Stride = 2 ----------------------------------------------#
# # --- 15 Classes --------#
# FaceRec7=FaceRecognition(8,2)
# Bm1, Bacc1DS1= FaceRec7.trainRecognizertesting(trainPath, testPath, 15)
# print("DATASET-2\nNo. of classes =15\tStride=2\nAccuracy = "+str(Bacc1DS1))
# print("##############################################################")
# # --- 20 Classes --------#
# FaceRec8=FaceRecognition(8,2)
# Bm2, Bacc2DS1= FaceRec8.trainRecognizertesting(trainPath, testPath, 20)
# print("DATASET-2\nNo. of classes =20\tStride=2\nAccuracy = "+str(Bacc2DS1))
# print("##############################################################")
#
#
# # ----------------------------Stride = 3 ----------------------------------------------#
# # --- 15 Classes --------#
# FaceRec9=FaceRecognition(8,3)
# Bm4, Bacc3DS1= FaceRec9.trainRecognizertesting(trainPath, testPath, 15)
# print("DATASET-2\nNo. of classes =15\tStride=3\nAccuracy = "+str(Bacc3DS1))
# print("##############################################################")
#
# # --- 20 Classes --------#
# FaceRec10=FaceRecognition(8,3)
# Bm5, Bacc4DS1= FaceRec10.trainRecognizertesting(trainPath, testPath, 20)
# print("DATASET-2\nNo. of classes =20\tStride=3\nAccuracy = "+str(Bacc4DS1))
# print("##############################################################")
#
#
# # ----------------------------Stride = 4 ----------------------------------------------#
# # --- 15 Classes --------#
# FaceRec11=FaceRecognition(8,4)
# Bm6, Bacc5DS1= FaceRec11.trainRecognizertesting(trainPath, testPath, 15)
# print("DATASET-2\nNo. of classes =15\tStride=4\nAccuracy = "+str(Bacc5DS1))
# print("##############################################################")
#
# # --- 20 Classes --------#
# FaceRec12=FaceRecognition(8,4)
# Bm7, Bacc6DS1= FaceRec12.trainRecognizertesting(trainPath, testPath, 20)
# print("DATASET-2\nNo. of classes =20\tStride=4\nAccuracy = "+str(Bacc6DS1))
# print("##############################################################")
#
#--------------------------------------------------DataSet 3-----------------------------------------------------------#

trainPath="D:\\8th Semester\\FYP\Dataset3\\training"
testPath="D:\\8th Semester\\FYP\\Dataset3\\testing"

# ----------------------------Stride = 2 ----------------------------------------------#
# --- 15 Classes --------#
# FaceRec=FaceRecognition(8,2)
# Cm1, Cacc1DS1= FaceRec.trainRecognizertesting(trainPath, testPath, 15)
# print("DATASET-3\nNo. of classes =15\tStride=2\nAccuracy = "+str(Cacc1DS1))
# print("##############################################################")
# # --- 20 Classes --------#
# FaceRec=FaceRecognition(8,2)
# Cm2, Cacc2DS1= FaceRec.trainRecognizertesting(trainPath, testPath, 20)
# print("DATASET-3\nNo. of classes =20\tStride=2\nAccuracy = "+str(Cacc2DS1))
# print("##############################################################")
#
#
# # ----------------------------Stride = 3 ----------------------------------------------#
# # --- 15 Classes --------#
# FaceRec=FaceRecognition(8,3)
# Cm4, Cacc3DS1= FaceRec.trainRecognizertesting(trainPath, testPath, 15)
# print("DATASET-3\nNo. of classes =15\tStride=3\nAccuracy = "+str(Cacc3DS1))
# print("##############################################################")
#
# # --- 20 Classes --------#
# FaceRec=FaceRecognition(8,3)
# Cm5, Cacc4DS1= FaceRec.trainRecognizertesting(trainPath, testPath, 20)
# print("DATASET-3\nNo. of classes =20\tStride=3\nAccuracy = "+str(Cacc4DS1))
# print("##############################################################")


# ----------------------------Stride = 6 ----------------------------------------------#
# --- 15 Classes --------#
FaceRec=FaceRecognition(16,4)
Cm6, Cacc5DS1= FaceRec.trainRecognizertesting(trainPath, testPath, 90)
print("DATASET-3\nNo. of classes =15\tBlock=16\nAccuracy = "+str(Cacc5DS1))
print("##############################################################")

# # --- 20 Classes --------#
# FaceRec=FaceRecognition(16,4)
# Cm7, Cacc6DS1= FaceRec.trainRecognizertesting(trainPath, testPath, 20)
# print("DATASET-3\nNo. of classes =20\tBlock=16\nAccuracy = "+str(Cacc6DS1))
# print("##############################################################")
