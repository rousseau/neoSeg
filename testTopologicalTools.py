# coding=utf-8

import os,sys
import numpy as np
import nibabel
from skimage import measure


def checkConnectedComponents(Ilabel,totalLabel):
	connComp=list()
	for label in range(totalLabel):
		im=1*(Ilabel==label)
		x,numLabel=measure.label(im,connectivity=1,return_num=True)
		connComp.append(numLabel)
	return connComp




if __name__ == '__main__':
	'''
	Global:
		Tools to check the topology of an image of labels
	Functions to test:
		checkConnectedComponents(Ilabel,totalLabel)
				Input:
					Ilabel			=> 	3D image of labels
					totalLabel		=>	number of labels in Ilabel
				Return:
					connComp		=>	array of length = totalLabel, with
										the number of connected components of
										each label
	'''


	epath='/home/carlos/Test/Topology/examples'
	IlabelFull = nibabel.load(epath+'/Ilabel.nii.gz')

	totalLabel=4

	print checkConnectedComponents(IlabelFull.get_data(),totalLabel)
