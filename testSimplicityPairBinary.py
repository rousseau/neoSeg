# coding=utf-8

import os,sys
import numpy as np
import math
from skimage import measure
from scipy import ndimage
import matplotlib.pyplot as plt


def isBinarySimplePoint(im):
	simplePoint=0
	X = im.copy()
	### 6-connected test ###
	## X without corners ##
	corners=np.array([[[1,0,1],[0,0,0],[1,0,1]],[[0,0,0],[0,0,0],[0,0,0]],[[1,0,1],[0,0,0],[1,0,1]]])
	X[corners==1]=0
	## Labels 6-connected keeping the center ##
	label,numLabelCenter=measure.label(X,connectivity=1,return_num=True)
	## Labels 6-connected removing the center ##
	X[1,1,1] = 1-X[1,1,1]
	label,numLabelNoCenter=measure.label(X,connectivity=1,return_num=True)
	#print '\t1st test, number of labels with and without center: '+str(numLabelCenter)+' and '+str(numLabelNoCenter)
	## Numbers of labels have to be equals ##
	if numLabelCenter-numLabelNoCenter==0:
		### 26-connected background test ###
		## Complementary of X without center ##
		X = 1-im.copy()
		X[1,1,1] = 0
		## Labels 26-connected ##
		label,numLabel=measure.label(X,connectivity=3,return_num=True)
		## Result ###
		simplePoint=1 if numLabel==1 else 0

	return simplePoint


def isBinarySimplePair(im,center):
	simplePoint=0
	X = im.copy()
	### 6-connected test ####
	## X without corners ##
	#corners=np.array([[[1,0,1],[0,0,0],[1,0,1]],[[0,0,0],[0,0,0],[0,0,0]],[[1,0,1],[0,0,0],[1,0,1]]])
	X[0,0,0]=0
	X[0,0,-1]=0
	X[0,-1,0]=0
	X[0,-1,-1]=0
	X[-1,0,0]=0
	X[-1,0,-1]=0
	X[-1,-1,0]=0
	X[-1,-1,-1]=0
	## Labels 6-connected keeping the center ##
	label,numLabelCenter=measure.label(X,connectivity=1,return_num=True)
	## Labels 6-connected removing the center ##
	X[center] = 1-X[1,1,1]
	label,numLabelNoCenter=measure.label(X,connectivity=1,return_num=True)
	#print '\t1st test, number of labels with and without center: '+str(numLabelCenter)+' and '+str(numLabelNoCenter)
	## Numbers of labels have to be equals ##
	if numLabelCenter-numLabelNoCenter==0:
		### 26-connected background test ###
		## Complementary of X without center ##
		X = 1-im.copy()
		X[center] = 0
		## Labels 26-connected ##
		label,numLabel=measure.label(X,connectivity=3,return_num=True)
		## Result ###
		simplePoint=1 if numLabel==1 else 0
	# 	print '\t2nd test, number of labels for the complementary: '+str(numLabel)
	# else:
	# 	print '\t2nd test is not needed'

	return simplePoint


def showSubplot(im,columns,title):
	numIm = im.shape[2]
	rows = int((numIm/columns))

	plt.figure(title)
	i = 0
	for i in range(numIm):
		plt.subplot(rows,columns,i+1)
		plt.imshow(im[i,:,:],cmap='gray')
		plt.title('z = '+str(i))
		plt.axis('off')




if __name__ == '__main__':
	'''
	Test input:
		numExample			=>	Selected a predefined example
		switchConnectivity	=>	Switch the order of connectivity, e.g. switch
								6-26 to 26-6
	Function to test:
		isBinarySimplePair(im,connectivity)
				Input:
					im		=> 	Binary image corresponding to the neighbourhood
								of the pair (posible sizes are 4 x 3 x 3,
								3 x 4 x 3 or 3 x 3 x 4 vx)
					center	=>	The coordinates of pair to be evalued
				Return:
				 	1 		if the center of im is simple in 6-26 connectivity
					0		otherwise
	'''



	## Test Binary Simple Points ##

	numExample=0
	switchConnectivity=0


	for numExample in range(1,3+1):
		for switchConnectivity in range(2):

			print '\nExample '+str(numExample)+' - '+str(switchConnectivity)

			if numExample==1:
				# Example 1 #
				im=np.array([[[0,1,1,0],[1,1,1,1],[0,1,1,0]],\
							[[1,1,1,1],[1,1,1,1],[1,1,1,1]],\
							[[0,0,0,0],[0,0,1,0],[0,0,0,0]]])
				center=([1,1],[1,1],[1,2])
			elif numExample==2:
				# Example 2 #
				im=np.array([[[0,0,0],[0,1,1],[0,1,0],[0,1,0]],\
							[[0,1,0],[1,1,1],[0,1,0],[0,1,0]],\
							[[0,0,0],[0,1,1],[0,1,0],[0,1,0]]])
				center=([1,1],[1,2],[1,1])
			elif numExample==3:
				# Example 3 #
				im=np.array([[[0,0,0],[0,0,0],[0,0,0]],\
							[[0,0,0],[0,1,0],[0,0,0]],\
							[[0,1,0],[1,1,1],[0,1,0]],\
							[[0,1,0],[1,1,1],[0,1,0]]])
				center=([1,2],[1,1],[1,1])

			print im.shape

			if switchConnectivity==1:
				im=1-im


			print 'Result = '+str(isBinarySimplePair(im,center))

	list = [[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],\
			[[0,0,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],\
			[[0,0,0,0,1,0],[0,0,0,1,0,1],[0,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,0,0],[0,0,0,0,0,0]],\
			[[0,0,0,0,1,0],[0,0,0,1,0,1],[0,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,0,0],[0,0,0,0,0,0]],\
			[[0,0,0,0,1,0],[0,0,0,1,0,1],[0,0,1,0,0,1],[0,1,0,0,1,0],[0,0,1,1,0,0],[0,0,0,0,0,0]],\
			[[0,0,0,0,0,0],[0,0,0,0,1,0],[0,0,0,1,1,0],[0,0,1,1,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]]

	image = np.array(list)
	showSubplot(image,3,'Input image')

	padding = 1
	image = np.pad(image,[(padding,padding),(padding,padding),(padding,padding)],mode='constant', constant_values=0)

	homogeneous = 1*(ndimage.convolve(image,np.array([[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,-27,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]))==0)
	showSubplot(homogeneous[1:-1,1:-1,1:-1],3,'Homogeneous areas')

	simpleMap = np.zeros_like(image)
	for i in range(padding,image.shape[0]-padding):
		for j in range(padding,image.shape[1]-padding):
			for k in range(padding,image.shape[2]-padding):
				simpleMap[i,j,k] = isBinarySimplePoint(image[i-1:i+2,j-1:j+2,k-1:k+2])
	showSubplot(simpleMap[1:-1,1:-1,1:-1],3,'Simple points')

	paddingImage = np.ones_like(image)
	paddingImage[padding:-padding,padding:-padding,padding:-padding]=0
	mask = 1*((simpleMap + homogeneous + paddingImage)!=0)


	mask = (1-mask)
	showSubplot(mask[1:-1,1:-1,1:-1],3,'Area of simple pair candidates')

	simplePairMap = np.zeros_like(image)

	maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0,0,0]],[[1,-1,0]],[[0,0,0]]]))==0)
	maskFilteredPairs = 1*(maskFiltered + mask*(ndimage.convolve(mask,np.array([[[0,0,0]],[[0,1,-1]],[[0,0,0]]]))==0)!=0)
	showSubplot(maskFilteredPairs[1:-1,1:-1,1:-1],3,'Simple pair candidates in x-axis')
	ct = np.where(maskFiltered==1)
	for ict in range(len(ct[0])):
		simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict]] = isBinarySimplePair( image[ct[0][ict]-1:ct[0][ict]+2,ct[1][ict]-1:ct[1][ict]+2,ct[2][ict]-1:ct[2][ict]+3],([1,1],[1,1],[1,2]) )

	maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0],[0],[0]],[[1],[-1],[0]],[[0],[0],[0]]]))==0)
	maskFilteredPairs = 1*(maskFiltered + mask*(ndimage.convolve(mask,np.array([[[0],[0],[0]],[[0],[-1],[1]],[[0],[0],[0]]]))==0)!=0)
	showSubplot(maskFilteredPairs[1:-1,1:-1,1:-1],3,'Simple pair candidates in y-axis')
	ct = np.where(maskFiltered==1)
	for ict in range(len(ct[0])):
		simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict]] = isBinarySimplePair( image[ct[0][ict]-1:ct[0][ict]+2,ct[1][ict]-1:ct[1][ict]+3,ct[2][ict]-1:ct[2][ict]+2],([1,1],[1,2],[1,1]) )

	maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0],[1],[0]],[[0],[-1],[0]],[[0],[0],[0]]]))==0)
	maskFilteredPairs = 1*(maskFiltered + mask*(ndimage.convolve(mask,np.array([[[0],[0],[0]],[[0],[1],[0]],[[0],[-1],[0]]]))==0)!=0)
	showSubplot(maskFilteredPairs[1:-1,1:-1,1:-1],3,'Simple pair candidates in z-axis')
	ct = np.where(maskFiltered==1)
	for ict in range(len(ct[0])):
		simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict]] = isBinarySimplePair( image[ct[0][ict]-1:ct[0][ict]+3,ct[1][ict]-1:ct[1][ict]+2,ct[2][ict]-1:ct[2][ict]+2],([1,2],[1,1],[1,1]) )

	showSubplot(simplePairMap[1:-1,1:-1,1:-1],3,'Simple pairs')

	plt.show()
