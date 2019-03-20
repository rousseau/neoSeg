# coding=utf-8

import os,sys
import numpy as np
import math
from skimage import measure


def isBinarySimplePairPoint(im,center):
	simplePoint=0
	X = im.copy()
	### 6-connected test ###
	## Labels 6-connected keeping the center ##
	label,numLabelCenter=measure.label(X,connectivity=1,return_num=True)
	## Labels 6-connected removing the center ##
	X[center] = 1-X[1,1,1]
	label,numLabelNoCenter=measure.label(X,connectivity=1,return_num=True)
	print '\t1st test, number of labels with and without center: '+str(numLabelCenter)+' and '+str(numLabelNoCenter)
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
		print '\t2nd test, number of labels for the complementary: '+str(numLabel)
	else:
		print '\t2nd test is not needed'

	return simplePoint




if __name__ == '__main__':
	'''
	Test input:
		numExample			=>	Selected a predefined example, visual version
								in 'examples/examplesSimplicityBinary.pdf'
		switchConnectivity	=>	Switch the order of connectivity, e.g. switch
								6-26 to 26-6
	Function to test:
		isBinarySimplePairPoint(im,connectivity)
				Input:
					im		=> 	binary image of size 3 x 3 x 3 vx
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


			print 'Result = '+str(isBinarySimplePairPoint(im,center))
