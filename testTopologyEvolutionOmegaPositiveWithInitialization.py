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
from testTopologicalTools import checkConnectedComponents


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
		simplicityLabel=1
		numTopology=len(topologyList)
		## Evaluate label simplicity in each topology ##
		while((simplicityLabel==1) & (numTopology>0)):
			topo=topologyList[numTopology-1]
			## Discard when current label and evalued label are in a topology group or there are not at all ##
			im=Ilabel[x-1:x+2,y-1:y+2,z-1:z+2].copy()
			## Evaluation of all labels ##
			for l in range(totalLabel):
				im[Ilabel[x-1:x+2,y-1:y+2,z-1:z+2]==l]=1 if l in topo else 0
			if len(np.unique(im))>1:
				simplicityLabel=isBinarySimplePoint(im)
			numTopology-=1
		simplicity[evalLabel]=simplicityLabel
		numLabel-=1
	return simplicity


def containsList(smallList, bigList):
    for i in xrange(len(bigList)-len(smallList)+1):
        for j in xrange(len(smallList)):
            if bigList[i+j] != smallList[j]:
                break
        else:
            return i, i+len(smallList)
    return False



def getNeighbourCoordinates(x,y,z,sizeIm,index=False,distance=1):
	minx=x-distance
	miny=y-distance
	minz=z-distance
	maxx=x+distance+1
	maxy=y+distance+1
	maxz=z+distance+1

	coordinates=list()
	indexCoord=list()
	for iii,ii in enumerate(range(minx,maxx)):
		for ijj,jj in enumerate(range(miny,maxy)):
			for ikk,kk in enumerate(range(minz,maxz)):
				coordinates.append((ii,jj,kk))
				indexCoord.append((iii,ijj,ikk))
	if index==True:
		return coordinates,indexCoord
	else:
		return coordinates


def get6NeighbourCoordinates(x,y,z,sizeIm,index=False,distance=1):
	minx=x-distance
	miny=y-distance
	minz=z-distance
	maxx=x+distance+1
	maxy=y+distance+1
	maxz=z+distance+1

	coordinates=list()
	indexCoord=list()
	for iii,ii in enumerate(range(minx,maxx)):
		for ijj,jj in enumerate(range(miny,maxy)):
			for ikk,kk in enumerate(range(minz,maxz)):
				if 1*(ii==x) + 1*(jj==y) + 1*(kk==z) >=2:
					coordinates.append((ii,jj,kk))
					indexCoord.append((iii,ijj,ikk))
	if index==True:
		return coordinates,indexCoord
	else:
		return coordinates



def computeBenefitPos(Ilabel,Sref,x,y,z,newLabel,totalLabel):
	currentVal=np.zeros([totalLabel])
	currentVal[Ilabel[x,y,z]]=1
	nextVal=np.zeros([totalLabel])
	nextVal[newLabel]=1
	#return the benefit of a point in Ilabel
	return np.linalg.norm(currentVal-Sref[x,y,z,:])-np.linalg.norm(nextVal-Sref[x,y,z,:])


def fuseLabelsByTopology(Ilabel,topologyList,totalLabel):
	newIlabel=Ilabel.copy()
	if len(topologyList)<totalLabel:
		count=0
		for topo in topologyList:
			if 0 in topo:
				tlabel=0
			else:
				count+=1
				tlabel=count
			for t in topo:
				newIlabel[newIlabel==t]=tlabel
	return newIlabel


def fuseSrefByTopology(Sref,topologyList,totalLabel):
	newSref=Sref.copy()
	if len(topologyList)<totalLabel:
		count=0
		for topo in topologyList:
			if 0 in topo:
				newSref
			else:
				count+=1
				tlabel=count
				for t in topo:
					newSref[:,:,:,tlabel]+=newSref[:,:,:,t]
					if t!=tlabel:
						newSref[:,:,:,t]=0
	##Normalise##
	sum=np.sum(newSref,axis=3)
	sum[sum==0]=1
	for l in range(totalLabel):
		newSref[:,:,:,l]/=(sum+0.0)
	return newSref


def upsampling3D(im,factor):
	sizeIm=im.shape
	newIm=np.zeros([im.shape[0]*2,im.shape[1]*2,im.shape[2]*2])
	for ii in xrange(sizeIm[0]):
		for jj in xrange(sizeIm[1]):
			for kk in xrange(sizeIm[2]):
				newIm[ii*factor:(ii+1)*factor,jj*factor:(jj+1)*factor,kk*factor:(kk+1)*factor]=im[ii,jj,kk]
	return newIm



def modifyPoint(Ilabel,newIlabel,newLabel,x,y,z,distance,totalLabel): ###NEW###
	Ilabel[x,y,z]=newLabel*3
	newIlabel[x,y,z]=newLabel

	## Topological changes ##
	for label in list(reversed(range(1,totalLabel))):
		tmp=Ilabel[x-distance:x+distance+1,y-distance:y+distance+1,z-distance:z+distance+1].copy()
		##Check the neighbours (and the point x,y,z)##
		Neighbours=getNeighbourCoordinates(x,y,z,Ilabel.shape,index=True,distance=distance)
		neighbours=[[(Neighbours[0][inb][0],Neighbours[0][inb][1],Neighbours[0][inb][2]),(Neighbours[1][inb][0],Neighbours[1][inb][1],Neighbours[1][inb][2])] for inb in range(len(Neighbours[0])) if (Ilabel[Neighbours[0][inb][0],Neighbours[0][inb][1],Neighbours[0][inb][2]]!=0)]
		for nb in neighbours:
			neighbours2=getNeighbourCoordinates(nb[0][0],nb[0][1],nb[0][2],Ilabel.shape)
			if newLabel==0:
				if(label==2):
					if(len([nb2 for nb2 in neighbours2 if Ilabel[nb2[0],nb2[1],nb2[2]]==0]) > 0):
						tmp[nb[1][0],nb[1][1],nb[1][2]]=3

					if(len([nb2 for nb2 in neighbours2 if Ilabel[nb2[0],nb2[1],nb2[2]]==3]) == 0):
						tmp[nb[1][0],nb[1][1],nb[1][2]]=1
				if(label==1):
					if(len([nb2 for nb2 in neighbours2 if Ilabel[nb2[0],nb2[1],nb2[2]]==3]) > 0) & (len([nb2 for nb2 in neighbours2 if Ilabel[nb2[0],nb2[1],nb2[2]]==0]) == 0):
						im=Ilabel[nb[0][0]-1:nb[0][0]+2,nb[0][1]-1:nb[0][1]+2,nb[0][2]-1:nb[0][2]+2].copy()
						tmp[nb[1][0],nb[1][1],nb[1][2]]=2
						im=1*(im==2)
						if(isBinarySimplePoint(im)):
							tmp[nb[1][0],nb[1][1],nb[1][2]]=2
			elif newLabel==1:
				if(label==3):
				 	if(len([nb2 for nb2 in neighbours2 if Ilabel[nb2[0],nb2[1],nb2[2]]==0]) == 0):
						im=Ilabel[nb[0][0]-1:nb[0][0]+2,nb[0][1]-1:nb[0][1]+2,nb[0][2]-1:nb[0][2]+2].copy()
						im=1*(im==2)
						if(isBinarySimplePoint(im)):
							tmp[nb[1][0],nb[1][1],nb[1][2]]=2
				if(label==2):
					if(len([nb2 for nb2 in neighbours2 if Ilabel[nb2[0],nb2[1],nb2[2]]==0]) > 0):
						tmp[nb[1][0],nb[1][1],nb[1][2]]=3
					elif(len([nb2 for nb2 in neighbours2 if Ilabel[nb2[0],nb2[1],nb2[2]]==3]) == 0):
						tmp[nb[1][0],nb[1][1],nb[1][2]]=1
		Ilabel[x-distance:x+distance+1,y-distance:y+distance+1,z-distance:z+distance+1]=tmp

	##Topology correction##
	tmp=1*(Ilabel==3)
	label,numLabel=measure.label(tmp,connectivity=1,return_num=True)
	if numLabel>1:
		labLength=[len(np.where(label==lab)[0]) for lab in range(1,numLabel+1)]
		labMax=labLength.index(np.max(labLength))+1
		for lab in range(1,numLabel+1):
			if lab!=labMax:
				Ilabel[label==lab]=1

	tmp=1*(Ilabel==2)
	label,numLabel=measure.label(tmp,connectivity=1,return_num=True)
	if numLabel>1:
		labLength=[len(np.where(label==lab)[0]) for lab in range(1,numLabel+1)]
		labMax=labLength.index(np.max(labLength))+1
		for lab in range(1,numLabel+1):
			if lab!=labMax:
				Ilabel[label==lab]=3

	tmp=1*(Ilabel==1)
	label,numLabel=measure.label(tmp,connectivity=1,return_num=True)
	if numLabel>1:
		labLength=[len(np.where(label==lab)[0]) for lab in range(1,numLabel+1)]
		labMax=labLength.index(np.max(labLength))+1
		for lab in range(1,numLabel+1):
			if lab!=labMax:
				Ilabel[label==lab]=2


	return Ilabel,newIlabel


@jit
def initPos(Ilabel,Sref,mask,topologyList,totalLabel):
	simpleMap,benefitMap=np.zeros(Sref.shape,dtype=int),np.zeros(Sref.shape)



	## Simple Points ##
	pts=np.where(mask==1)
	for ipt in xrange(len(pts[0])):
		x,y,z=pts[0][ipt],pts[1][ipt],pts[2][ipt]
		simpleMap[x,y,z,:]=isSimple(Ilabel,topologyList,totalLabel,x,y,z) ###NEW###

	## Benefit Map ##
	pts=np.where(simpleMap==1)
	for ipt in xrange(len(pts[0])):
		x,y,z,newLabel=pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]
		benefitMap[x,y,z,newLabel]=computeBenefitPos(Ilabel,Sref,x,y,z,newLabel,totalLabel) ###NEW###

	return simpleMap,benefitMap




def updateMapsPos(Ilabel,Sref,simpleMap,benefitMap,x,y,z,newLabel,topologyList,totalLabel):
	neighbours=getNeighbourCoordinates(x,y,z,Ilabel.shape)
	for nb in neighbours:

		## update simpleMap ##
		simpleMap[nb[0],nb[1],nb[2],:]=isSimple(Ilabel,topologyList,totalLabel,nb[0],nb[1],nb[2]) ###NEW###
		## update benefitMap ##
		for l in range(totalLabel):
			if simpleMap[nb[0],nb[1],nb[2],l]==1:
				benefitMap[nb[0],nb[1],nb[2],l]=computeBenefitPos(Ilabel,Sref,nb[0],nb[1],nb[2],l,totalLabel) ###NEW###
			else:
				benefitMap[nb[0],nb[1],nb[2],l]=0

	return simpleMap,benefitMap




if __name__ == '__main__':
	'''
	Global:
		Tests with positives values of omega with an initialisation until
		a threshold for omega (var: thresholdAllLabels)
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

	## Create inputs ##
	ipath='/home/carlos/Test/IMAPA/31/results'
	epath='/home/carlos/Test/Topology/examples'
	os.system('mkdir -p '+epath)

	labelsPath=list()
	labelsPath.append(ipath+'/WM/Simapa_wm_atlasdHCP30_K15_hps1_hss3_alpha0.25.nii.gz')
	labelsPath.append(ipath+'/Cortex/Simapa_cortex_atlasdHCP30_K15_hps1_hss3_alpha0.25.nii.gz')
	labelsPath.append(ipath+'/CSF/Simapa_csf_atlasdHCP30_K15_hps1_hss3_alpha0.25.nii.gz')

	omega=4

	threshold=1

	distance=5

	padding3D=[(distance,distance),(distance,distance),(distance,distance)]
	padding4D=[(distance,distance),(distance,distance),(distance,distance),(0,0)]

	buildInputs(labelsPath,epath,omega,threshold)

	## Test topological evolution of Ilabel given a positive omega ##
	epath='/home/carlos/Test/Topology/examples'
	opath='/home/carlos/Test/Topology'
	tpath=opath+'/test'
	os.system('mkdir -p '+opath+' '+epath+' '+tpath)
	IlabelFull = nibabel.load(epath+'/Ilabel.nii.gz')
	Ilabel = IlabelFull.get_data().astype(int)
	SrefFull = nibabel.load(epath+'/Sref.nii.gz')
	Sref = SrefFull.get_data()

	omegas = [4,3,2,1,0]

	totalLabel=4

	thresholdAllLabels=2

	thresholdLoopPrint=10000

	union=''

	for omega in omegas:

		if omega==thresholdAllLabels:
			iterations=[1,2]
		else:
			iterations=[1]

		for iter in iterations:

			Ilabel=np.pad(Ilabel,padding3D,mode='constant', constant_values=0)
			mask=np.zeros(Ilabel.shape)
			mask[distance:-distance-1,distance:-distance-1,distance:-distance-1]=1

			## Downsampling Sref ##
			newSref=downsampling(Sref,omega)

			newSref=np.pad(newSref,padding4D,mode='constant', constant_values=0)



			nibabel.save(nibabel.Nifti1Image(newSref.astype(float),IlabelFull.affine),opath+'/newSref_omega'+str(omega)+'.nii.gz')

			if union!='_allLabels':
				## New Sref 2 ##
				topologyList=[[1,2,3],[0]]
				newSref2=fuseSrefByTopology(newSref,topologyList,totalLabel)
				nibabel.save(nibabel.Nifti1Image(newSref2.astype(float),IlabelFull.affine),opath+'/newSref2_omega'+str(omega)+'.nii.gz')
				## New Ilabel ##
				newIlabel=fuseLabelsByTopology(Ilabel,topologyList,totalLabel)
				nibabel.save(nibabel.Nifti1Image(newIlabel.astype(float),IlabelFull.affine),opath+'/newIlabel_omega'+str(omega)+'.nii.gz')
				## Inisialisation
				simpleMap,benefitMap=initPos(newIlabel,newSref2,mask,topologyList,totalLabel)
			else:
				topologyList=[[3],[2],[1],[0]]
				simpleMap,benefitMap=initPos(Ilabel,newSref,mask,topologyList,totalLabel)

			nibabel.save(nibabel.Nifti1Image(simpleMap.astype(float),IlabelFull.affine),opath+'/simpleMap_omega'+str(omega)+'.nii.gz')

			print '\nSize of Ilabel = \t'+str(simpleMap.shape)+'\n\t\t\t'+str(benefitMap.shape)+'\n'


			loops=0
			while(len(np.where(benefitMap>0)[0])>0):
				if loops%thresholdLoopPrint==0:
					print str(loops)+' loops, \t'+str(len(np.where(benefitMap>0)[0]))+' simple points'
					nibabel.save(nibabel.Nifti1Image(Ilabel.astype(float),IlabelFull.affine),tpath+'/Ilabel_iter'+str(int(loops/thresholdLoopPrint))+'.nii.gz')
				pt=np.where(benefitMap==np.max(benefitMap))

				x,y,z,newLabel=pt[0][0],pt[1][0],pt[2][0],pt[3][0]
				## Update values ##
				if union!='_allLabels':
					Ilabel,newIlabel=modifyPoint(Ilabel,newIlabel,newLabel,x,y,z,distance,totalLabel) ### NEW ###
					simpleMap,benefitMap=updateMapsPos(newIlabel,newSref2,simpleMap,benefitMap,x,y,z,newLabel,topologyList,totalLabel) ### NEW ###
				else:
					Ilabel[x,y,z]=newLabel
					simpleMap,benefitMap=updateMapsPos(Ilabel,newSref,simpleMap,benefitMap,x,y,z,newLabel,topologyList,totalLabel)

				loops+=1


			nibabel.save(nibabel.Nifti1Image(Ilabel.astype(float),IlabelFull.affine),opath+'/Ilabel_pad_omega'+str(omega)+union+'.nii.gz')
			Ilabel=Ilabel[padding3D[0][0]:-padding3D[0][1],padding3D[1][0]:-padding3D[1][1],padding3D[2][0]:-padding3D[2][1]]
			nibabel.save(nibabel.Nifti1Image(Ilabel.astype(float),IlabelFull.affine),opath+'/Ilabel_omega'+str(omega)+union+'.nii.gz')

			print '\nConnected components of Ilabel = '+str(checkConnectedComponents(Ilabel,totalLabel))+'\n'

			if (len(iterations)==2):
				union='_allLabels'

		Ilabel=upsampling3D(Ilabel,2).astype(int)
		IlabelFull = nibabel.load(opath+'/Ilabel_omega'+str(omega)+union+'.nii.gz')
