# coding=utf-8

import os,sys
import numpy as np
import math
from skimage import measure
from scipy import ndimage
import nibabel
from testSimplicityBinary import isBinarySimplePoint



def containsList(smallList, bigList):
    for i in xrange(len(bigList)-len(smallList)+1):
        for j in xrange(len(smallList)):
            if bigList[i+j] != smallList[j]:
                break
        else:
            return i, i+len(smallList)
    return False

def isSimple(Ilabel,topologyList,totalLabel,x,y,z):
	simplicity=np.zeros([totalLabel],dtype=int)
	corners=np.array([[[1,0,1],[0,0,0],[1,0,1]],[[0,0,0],[0,0,0],[0,0,0]],[[1,0,1],[0,0,0],[1,0,1]]],dtype=int)
	im=Ilabel[x-1:x+2,y-1:y+2,z-1:z+2].copy()
	im[corners==1]=-2
	allLabels=np.unique(im)[1:]
	centralLabel=Ilabel[x,y,z]
	im[im==centralLabel]=-1
	candidatesLabel=np.unique(im)[2:]
	print '\tCandidate labels to evaluate its simplicity: '+str(candidatesLabel)[1:-1]
	numLabel=len(candidatesLabel)
	## Loop of labels in the neighbourhood of (x,y,z) ##
	while(numLabel>0):
		evalLabel=candidatesLabel[numLabel-1]
		print evalLabel
		simplicityLabel=1
		#topoList=[topo for topo in topologyList for l in np.unique(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2]) if l in topo]
		topoList=[topo for topo in topologyList if (evalLabel in topo) & (containsList(allLabels,topo)==False) ]
		numTopology=len(topoList)
		## Evaluate label simplicity in each topology ##
		while((simplicityLabel==1) & (numTopology>0)):
			topo=topoList[numTopology-1]
			print topo
			## Discard when current label and evalued label are in a topology group or there are not at all ##
			im=Ilabel[x-1:x+2,y-1:y+2,z-1:z+2].copy()
			## Evaluation of all labels ##
			for l in range(totalLabel):
				im[Ilabel[x-1:x+2,y-1:y+2,z-1:z+2]==l]=1 if l in topo else 0
			#im[1,1,1]=1
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



	## Test Multilabel Simple Points ##
	ipath='/home/carlos/Code/neoSeg'
	Ilabel = nibabel.load(ipath+'/examples/Ilabel.nii.gz')
	Ilabel = Ilabel.get_data().astype(int)
	x,y,z = 21,36,17 # Cortex point #
	x,y,z = 14,34,25 # Background point #
	x,y,z = 14,36,25 # Cortex point #
	x,y,z = 26,51,35 # Cortex point #
	x,y,z = 25,36,12 # Cortex point #

	print '\nInput neighbourhood to be evaluated:\n\n'+str(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2])+'\n'

	topologyList=[[1,2],[2,3],[1,2,3],[1],[0]]
	totalLabel=4

	print '\nResult = '+str(isSimple(Ilabel,topologyList,totalLabel,x,y,z))+'\n'
