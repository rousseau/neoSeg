# coding=utf-8

import os,sys
import numpy as np
import math
import nibabel
from skimage import measure
from scipy import ndimage
from numba import jit
from testDownsamplingSref import downsampling


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

def isSimple(Ilabel,topologyList,totalLabel,x,y,z):
	simplicity=np.zeros([totalLabel],dtype=int)
	im=Ilabel[x-1:x+2,y-1:y+2,z-1:z+2].copy()
	allLabels=np.unique(im)
	centralLabel=Ilabel[x,y,z]
	im[im==centralLabel]=-1
	candidatesLabel=np.unique(im)[1:]
	numLabel=len(candidatesLabel)
	## Loop of labels in the neighbourhood of (x,y,z) ##
	while(numLabel>0):
		evalLabel=candidatesLabel[numLabel-1]
		numTopology=len(topologyList)
		## Evaluate label simplicity in each topology ##
		simplicityLabel=1 if numTopology>0 else 0
		while((simplicityLabel==1) & (numTopology>0)):
			topo=topologyList[numTopology-1]
			im=Ilabel[x-1:x+2,y-1:y+2,z-1:z+2].copy()
			## Evaluation of all labels ##
			for l in range(totalLabel):
				im[Ilabel[x-1:x+2,y-1:y+2,z-1:z+2]==l]=1 if l in topo else 0
			if len(np.unique(im))>1:
				simplicityLabel*=isBinarySimplePoint(im)
				simplicityLabel*=isBinarySimplePoint(1-im)
			numTopology-=1
		simplicity[evalLabel]=simplicityLabel
		numLabel-=1
	return simplicity



def computeError(Ilabel,newSref,totalLabel):
	IlabelRef=np.zeros(newSref.shape)
	for l in xrange(totalLabel):
		im=np.zeros(Ilabel.shape)
		im[Ilabel==l]=1
		IlabelRef[:,:,:,l]=im
	error=np.linalg.norm(IlabelRef-newSref)
	return error


def getNeighbourCoordonates(x,y,z,sizeIm):
	minx=x-1 if x!=0 else 0
	miny=y-1 if y!=0 else 0
	minz=z-1 if z!=0 else 0
	maxx=x+2 if x!=sizeIm[0] else sizeIm[0]+1
	maxy=y+2 if y!=sizeIm[1] else sizeIm[1]+1
	maxz=z+2 if z!=sizeIm[2] else sizeIm[2]+1

	coordonates=list()
	for ii in range(minx,maxx):
		for jj in range(miny,maxy):
			for kk in range(minz,maxz):
				coordonates.append((ii,jj,kk))
	return coordonates



def computeBenefitPos(Ilabel,Sref,x,y,z,newLabel,totalLabel):
	currentVal=np.zeros([totalLabel])
	currentVal[Ilabel[x,y,z]]=1
	nextVal=np.zeros([totalLabel])
	nextVal[newLabel]=1
	return np.linalg.norm(currentVal-Sref[x,y,z,:])-np.linalg.norm(nextVal-Sref[x,y,z,:])
	#return the benefit of a point in Ilabel



@jit
def initPos(Ilabel,Sref,mask,topologyList,totalLabel):
	simpleMap,benefitMap=np.zeros(Sref.shape,dtype=int),np.zeros(Sref.shape)

	## Simple Points ##
	pts=np.where(mask==1)
	for ipt in xrange(len(pts[0])):
		x,y,z=pts[0][ipt],pts[1][ipt],pts[2][ipt]
		simpleMap[x,y,z,:]=isSimple(Ilabel,topologyList,totalLabel,x,y,z)

	## Benefit Map ##
	pts=np.where(simpleMap==1)
	for ipt in xrange(len(pts[0])):
		x,y,z,newLabel=pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]
		benefitMap[x,y,z,newLabel]=computeBenefitPos(Ilabel,Sref,x,y,z,newLabel,totalLabel)

	return simpleMap,benefitMap




def updateMapsPos(Ilabel,Sref,simpleMap,benefitMap,x,y,z,newLabel,topologyList,totalLabel):
	neighbours=getNeighbourCoordonates(x,y,z,Ilabel.shape)
	for nb in neighbours:
		## update simpleMap ##
		simpleMap[nb[0],nb[1],nb[2],:]=isSimple(Ilabel,topologyList,totalLabel,nb[0],nb[1],nb[2])
		## update benefitMap ##
		for l in range(totalLabel):
			if simpleMap[nb[0],nb[1],nb[2],l]==1:
				benefitMap[nb[0],nb[1],nb[2],l]=computeBenefitPos(Ilabel,Sref,nb[0],nb[1],nb[2],l,totalLabel)
			else:
				benefitMap[nb[0],nb[1],nb[2],l]=0

	return simpleMap,benefitMap




if __name__ == '__main__':
	'''
	Global:
		Tests with one omega and positive
		typeInput parameter switch Sref input as synthesis or real data
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



	## Test topological evolution of Ilabel given a positive omega ##
	typeInput='synthesis'	# or 'real'
	ipath='/home/carlos/Code/neoSeg'
	opath=ipath+'/test'
	os.system('mkdir -p '+opath)
	IlabelFull = nibabel.load(ipath+'/examples/Ilabel.nii.gz')
	Ilabel = IlabelFull.get_data().astype(int)
	omega = 2
	mask=np.zeros(Ilabel.shape)
	mask[1:-1,1:-1,1:-1]=1
	topologyList=[[3],[2],[1]]
	topologyList=[[1,2],[2,3],[1,2,3],[1],[2],[3]]
	topologyList=[[1,2],[2,3],[1,2,3],[1]]
	totalLabel=4

	## Downsampling Sref ##
	if typeInput=='synthesis':
		newSref = nibabel.load(ipath+'/examples/Sref_omega2_synthesized.nii.gz').get_data()
	else:
		SrefFull = nibabel.load(ipath+'/examples/Sref.nii.gz')
		Sref = SrefFull.get_data()
		newSref=downsampling(Sref,omega)
	nibabel.save(nibabel.Nifti1Image(newSref.astype(float),IlabelFull.affine),ipath+'/newSref_omega'+str(omega)+'.nii.gz')



	print 'Initial total error: '+str(computeError(Ilabel,newSref,totalLabel))

	## Inisialisation ##
	simpleMap,benefitMap=initPos(Ilabel,newSref,mask,topologyList,totalLabel)

	## Save initial simpleMap and benefitMap ##
	nibabel.save(nibabel.Nifti1Image(simpleMap.astype(float),IlabelFull.affine),ipath+'/simpleMap_omega'+str(omega)+'_initial.nii.gz')
	nibabel.save(nibabel.Nifti1Image(benefitMap.astype(float),IlabelFull.affine),ipath+'/benefitMap_omega'+str(omega)+'_initial.nii.gz')

	loops=0
	while(len(np.where(benefitMap>0)[0])>0):
		if loops%1000==0:
			print str(loops)+' loops, \t'+str(len(np.where(benefitMap>0)[0]))+' simple points'
			nibabel.save(nibabel.Nifti1Image(Ilabel.astype(float),IlabelFull.affine),opath+'/Ilabel_iter'+str(int(loops/1000))+'.nii.gz')
		## Select point and label which give the maximum benefit ##
		pt=np.where(benefitMap==np.max(benefitMap))
		x,y,z,newLabel=pt[0][0],pt[1][0],pt[2][0],pt[3][0]
		## Update its values ##
		Ilabel[x,y,z]=newLabel
		simpleMap,benefitMap=updateMapsPos(Ilabel,newSref,simpleMap,benefitMap,x,y,z,newLabel,topologyList,totalLabel)
		loops+=1

	## Save final simpleMap and benefitMap ##
	nibabel.save(nibabel.Nifti1Image(simpleMap.astype(float),IlabelFull.affine),ipath+'/simpleMap_omega'+str(omega)+'_final.nii.gz')
	nibabel.save(nibabel.Nifti1Image(benefitMap.astype(float),IlabelFull.affine),ipath+'/benefitMap_omega'+str(omega)+'_final.nii.gz')

	## Save Ilabel result ##
	nibabel.save(nibabel.Nifti1Image(Ilabel.astype(float),IlabelFull.affine),ipath+'/Ilabel_omega'+str(omega)+'.nii.gz')

	print 'Final total error: '+str(computeError(Ilabel,newSref,totalLabel))
