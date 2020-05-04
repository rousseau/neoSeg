# coding=utf-8
import numpy as np
from skimage import measure
import nibabel
from numba import jit


def checkConnectedComponents(image,totalLabel,connectivity):
	'''
	Check the connected components in a image given the number of labels and
	the connectivity configuration of each one
	-------------------------------
	Input:	image 			--> An image of labels (in 2D or 3D)
			totalLabel		--> Number of labels of the image
			connectivity	--> An array of the connectivity configuration
	Output:	connComp	--> An array of connected components corresponding for
							each label
	'''
	connComp=list()
	for label in range(totalLabel):
		conn = 1 if connectivity[label]==6 else 3
		im=1*(image==label)
		x,numLabel=measure.label(im,connectivity=conn,return_num=True)
		connComp.append(numLabel)

	return connComp


def computeEulerCharacteristic3D(im):
	'''
	Compute the Euler Characteristic number of a 3D image
	-------------------------------
	Input:	im 	--> A 3D image of labels
	Output:	eulerCharacteristic	--> Euler characteristic number
	'''
	sizex,sizey,sizez = im.shape
	khalimskiGrid = np.zeros((2*sizex+1,2*sizey+1,2*sizez+1))
	##Fill the Khalimski grid##
	for x in range(sizex):
		for y in range(sizey):
			for z in range(sizez):
				if im[x,y,z]==1:
					khalimskiGrid[2*x:2*x+3, 2*y:2*y+3, 2*z:2*z+3] = 1 	#center: (2*x+1,2*y+1,2*z+1)
	k0 = np.count_nonzero(khalimskiGrid[::2,::2,::2])
	k1 = np.count_nonzero(khalimskiGrid[::2,::2,1::2]) + np.count_nonzero(khalimskiGrid[::2,1::2,::2]) + np.count_nonzero(khalimskiGrid[1::2,::2,::2])
	k2 = np.count_nonzero(khalimskiGrid[1::2,1::2,::2]) + np.count_nonzero(khalimskiGrid[1::2,::2,1::2]) + np.count_nonzero(khalimskiGrid[::2,1::2,1::2])
	k3 = np.count_nonzero(khalimskiGrid[1::2,1::2,1::2])
	eulerCharacteristic = k0 - k1 + k2 - k3
	return eulerCharacteristic



def computeBettiNumbers(image,totalLabel,connectivity):
	'''
	Compute the Betti numbers of a 3D image
	-------------------------------
	Input:	image 			--> A 3D image of labels
			totalLabel 		--> Number of labels of the image
			connectivity 	--> An array of the connectivity configuration
	Output:	bettiNumbers	--> An array of 3-length arrays corresponding to the
								3 Betti numbes of each label
	'''
	bettiNumbers = np.zeros((totalLabel,3))
	for l in range(totalLabel):
		im = 1*(image==l)
		if connectivity[l]==6:
			eulerCharacteristic = computeEulerCharacteristic3D(1-im)
			conn0 = 1
			conn2 = 3
		else:
			eulerCharacteristic = computeEulerCharacteristic3D(im)
			conn0 = 3
			conn2 = 1
		_,b0 = measure.label(im,connectivity=conn0,return_num=True)
		_,b2 = measure.label(1-im,connectivity=conn2,return_num=True)
		b1 = b0 + b2 - eulerCharacteristic
		if connectivity[l]==6:
			b2 -= 1
		bettiNumbers[l,:] = np.array([b0,b1,b2])
	return bettiNumbers
