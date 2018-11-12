# coding=utf-8

import os,sys
import numpy as np
import math
import nibabel
from skimage import measure
from scipy import ndimage
from numba import jit
from testDownsamplingSref import downsampling
from createInputsTopology import buildInputs


def checkConnectedComponents(image,totalLabel):
	connComp=list()
	for label in range(totalLabel):
		im=1*(image==label)
		x,numLabel=measure.label(im,connectivity=1,return_num=True)
		connComp.append(numLabel)
	return connComp

def checkHoles(image,totalLabel):
	holes=list()
	for label in range(totalLabel):
		im=1*(image==label)
		x,numLabel=measure.label(im,connectivity=1,return_num=True) ##################CAMBIAR A FUNCION HOLES#############################
		holes.append(numLabel)
	return holes

def checkLabelThickness(image,label,totalLabel):
	thickness=list()
	for label in range(totalLabel):
		im=1*(image==label)
		x,numLabel=measure.label(im,connectivity=1,return_num=True) ##################CAMBIAR A FUNCION THICKNESS#############################
		thickness.append(numLabel)
	return thickness




if __name__ == '__main__':
	'''
	Global:
		Tests with one omega and positive
	Functions to test:
		initPos(Ilabel,Sref,mask,topologyList,totalLabel)
				Input:
					Ilabel			=> 	3D image of labels with
										size = size(Sref) / (2**omega)
					Sref			=> 	3D image of segmentation
										Size(Sref) >= size(Ilabel)
					mask			=>	binary mask avoiding points which are
					 					not simple
					topologyList 	=>	groups of labels to check its topology
					totalLabel		=>	number of labels in Ilabel
				Return:
					simpleMap		=>	initial map of simple points with same
										size as	Sref
					benefitMap		=>	initial map of benefits with same size
					 					as Sref

		updateMapsPos(Ilabel,Sref,simpleMap,benefitMap,x,y,z,newLabel,
						topologyList,totalLabel)
				Input:
					Ilabel			=> 	3D image of labels with
										size = size(Sref) / (2**omega)
					Sref			=> 	3D image of segmentation
										Size(Sref) >= size(Ilabel)
					simpleMap		=>	map of simple points with same size as
					 					Sref
					benefitMap		=>	map of benefits with same size as Sref
					x,y,z 			=>	point to be evaluated
					newLabel		=>	value of the label to be evaluated
					topologyList 	=>	groups of labels to check its topology
					totalLabel		=>	number of labels in Ilabel
				Return:
					simpleMap		=>	update map of simple points	Sref
					benefitMap		=>	update map ap of benefits
	'''


	epath='/home/carlos/Test/Topology/examples'
	opath='/home/carlos/Test/Topology'
	tpath=opath+'/test'
	os.system('mkdir -p '+opath+' '+epath+' '+tpath)
	IlabelFull = nibabel.load(epath+'/Ilabel.nii.gz')
	######
	IlabelFull = nibabel.load(opath+'/Ilabel_omega1_allLabels.nii.gz')
	#Ilabel=upsampling3D(IlabelFull.get_data(),2).astype(int)
	######
	SrefFull = nibabel.load(epath+'/Sref.nii.gz')
	Sref = SrefFull.get_data()

	totalLabel=4

	print checkConnectedComponents(IlabelFull.get_data(),totalLabel)
