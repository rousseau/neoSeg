# coding=utf-8

import os,sys
import numpy as np
import math
from skimage import measure
from scipy import ndimage
import nibabel
from testSimplicityBinary import isBinarySimplePoint



def isSimple(Ilabel,topologyList,totalLabel,x,y,z):
	simplicity=np.zeros([totalLabel],dtype=int)
	im=Ilabel[x-1:x+2,y-1:y+2,z-1:z+2]
	centralLabel=Ilabel[x,y,z]
	im[im==centralLabel]=-1
	candidatesLabel=np.unique(im)[1:]
	print '\tCandidate labels to evaluate its simplicity: '+str(candidatesLabel)[1:-1]
	numLabel=len(candidatesLabel)
	numTopology=len(topologyList)
	## Loop of labels in the neighbourhood of (x,y,z) ##
	while(numLabel>0):
		evalLabel=candidatesLabel[numLabel-1]
		simplicityLabel=1
		## Evaluate label simplicity in each topology ##
		while((simplicityLabel==1) & (numTopology>0)):
			topo=topologyList[numTopology-1]
			## Discard when current label and evalued label are in a topology group or there are not at all ##
			if ((evalLabel in topo)&(centralLabel in topo)&(len(topo)==2)) & ((evalLabel not in topo)&(centralLabel not in topo)):
				im=Ilabel[x-1:x+2,y-1:y+2,z-1:z+2]
				im[im not in topo]=0
				im[im in topo]=1
				im[1,1,1]=1
				simplicityLabel=isBinarySimplePoint(im)
			numTopology-=1
		simplicity[evalLabel]=simplicityLabel
		numLabel-=1
	if len(np.unique(simplicity))==2:
		print '\tSimplicity found in labels: '+str([isim for isim,sim in enumerate(simplicity) if sim==1])[1:-1]
	else:
		print '\tNo simplicity found.'
	return simplicity




if __name__ == '__main__':
	'''
	Function to test:
		isSimple(Ilabel,topologyList,totalLabel,x,y,z)
				Input:
					Ilabel			=> 	3D image of labels
					topologyList 	=>	groups of labels to check its topology
					totalLabel		=>	number of labels in Ilabel
					x,y,z 			=>	point to be evaluated
				Return:
					Binary array with a size equal to the number of labels
					(totalLabel). Position in this array determines the evalued
					label and its value is:
					 	1 		if (x,y,z) is simple in 6-26 connectivity
						0		otherwise
	'''



	## Test Binary Simple Points ##
	ipath='/home/carlos/Code/neoSeg'
	Ilabel = nibabel.load(ipath+'/examples/Ilabel.nii.gz')
	Ilabel = Ilabel.get_data().astype(int)
	x,y,z = 14,34,25 # Background point #
	x,y,z = 21,36,17 # Cortex point #
	x,y,z = 24,46,18 # WM point #

	print '\nInput neighbourhood to be evaluated:\n\n'+str(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2])+'\n'

	topologyList=[[1,2],[2,3],[1,2,3],[1]]
	totalLabel=4

	print '\nResult = '+str(isSimple(Ilabel,topologyList,totalLabel,x,y,z))+'\n'
