import numpy as np
import math
import cv2
import time
from matplotlib import pyplot as plt
import statistics

class customMe:                             #Main LBP Class. Initialized with block size 
	def __init__(self, block_size):
		self.block_size=block_size  ##block_size : Block size into which the image is to be divided.

	def findLBP(self,img):                   #Main function to calculate LBP. Input: Image, Output : LBP Codes
		(N, M) = img.shape               ## N: Number of Rows; M: Number of coloumns 
		codes=[]			 ## Initializing a an empty list. It will contain the LBPs
		r = 0				##Row Iterator
		while True:
			c = 0			#Column Iterator 
			while True:                                             #Loop Selects a 3x3 window over which LBP is to be computed.
				sliceMe = img[r:r + 3, c:c + 3]                 ##SliceMe: A 3x3 portion of image 
				thresh = sliceMe[1, 1]                          ##Thresh : Thresholding value. Selected to be the center pixel in 3x3 window. 
				sliceCopy = sliceMe                             ##SliceCopy: Creating a copy so as to not alter the original window.
				sliceCopy = np.where(sliceCopy >= thresh, 1, 0)     #Thresholding with center element 
				##The follow computes the LBP in an anticlockwise manner (first pixel->MSB)
				code = sliceCopy[0, 0] * 128 + sliceCopy[0, 1] * 64 + sliceCopy[0, 2] * 32 + sliceCopy[1, 2] * 16 + \
				       sliceCopy[2, 2] * 8 + sliceCopy[2, 1] * 4 + sliceCopy[2, 0] * 2 + sliceCopy[1, 0]
				
				codes.append(code)                                    #Appending the code calculated into a list 
				##The following 3 lines move the window horizontally by 3 pixels 
				c = c + 3
				if c + 2 > M - 1:
					break
			##The following 3 lines move the window vertically by 3 pixels
			r = r + 3
			if r + 2 > N - 1:
				break
				
		codes = np.asarray(codes)                                       #Converting the list into a numpy array 
		[hist, edges] = np.histogram(codes.ravel(), bins=range(0, 256)) #Computing the historgram. hist is the final return variable of this method.
		hist = hist.astype("float")                                     #Converting from uint8 to float. 
		return hist                                                	#Returning hist.    	
