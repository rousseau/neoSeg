# coding=utf-8

import os,sys
import numpy as np
import math
import nibabel
from skimage import measure
from scipy import ndimage
from numba import jit
from time import time
from skimage import morphology,feature
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

@jit
def get6NeighbourCoordinates(x,y,z,index=False,distance=1):
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
				if 1*(ii==x) + 1*(jj==y) + 1*(kk==z) == 2: ####ANTES ==2 #####
					coordinates.append((ii,jj,kk))
					indexCoord.append((iii,ijj,ikk))
	if index==True:
		return coordinates,indexCoord
	else:
		return coordinates

@jit
def get18NeighbourCoordinates(x,y,z,index=False,distance=1):
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
				if (1*(ii==x) + 1*(jj==y) + 1*(kk==z) <= 2) & (1*(ii==x) + 1*(jj==y) + 1*(kk==z) > 0):
					coordinates.append((ii,jj,kk))
					indexCoord.append((iii,ijj,ikk))
	if index==True:
		return coordinates,indexCoord
	else:
		return coordinates



def computeMajorityVoting(Sref):
	return np.argmax(Sref,axis=3)



def computeThicknessDistance(Ilabel,label,totalLabel):
	thicknessLabelMap=np.zeros(Ilabel.shape)
	thicknessMap1=ndimage.morphology.distance_transform_edt(1-(1*(Ilabel==(label+1)%totalLabel)))
	thicknessMap2=ndimage.morphology.distance_transform_edt(1-(1*(Ilabel==(label-1)%totalLabel)))
	thicknessLabelMap=(Ilabel==label)*((thicknessMap1+thicknessMap2)-1)
	return thicknessLabelMap

def computeDistanceMap(Sref,totalLabel):
	distanceMap=np.zeros(Sref.shape)
	binSref=computeMajorityVoting(Sref)
	#nibabel.save(nibabel.Nifti1Image(binSref.astype(float),IlabelFull.affine),opath+'/MV_3l_omega0.nii.gz')
	for l in xrange(totalLabel):
		#distanceMap[:,:,:,l]=ndimage.morphology.distance_transform_cdt(1-(1*(binSref==l)), metric='taxicab')
		distanceMap[:,:,:,l]=ndimage.morphology.distance_transform_edt(1-(1*(binSref==l)))
	return distanceMap


def computeThicknessMap(Ilabel,milimeterSize,labels,totalLabel):
	nx, ny, nz = Ilabel.shape
	thicknessMap=np.zeros([nx,ny,nz,totalLabel])

	for l in labels:
		r = 1*(Ilabel==l) #structure
		S = 1*(Ilabel==(l+1)%totalLabel) #1st boundary
		S_prime = 1*(Ilabel==(l-1)%totalLabel) #2nd boundary
		R = np.where(r != 0)
		r[...,0] = 0
		r[...,nz-1] = 0
		#Initial conditions
		dx,dy,dz=milimeterSize[0],milimeterSize[1],milimeterSize[2]

		voxel_volume = dx * dy * dz

		u = np.zeros((nx,ny,nz))  #u(i)
		u_n = np.zeros((nx,ny,nz)) #u(i+1)
		L0,L1 = np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz))
		L0_n,L1_n = np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz))
		Nx,Ny,Nz = np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz))
		#Dirichlet boundary conditions
		u[np.where(S_prime != 0)] =  100
		# Boundary conditions for computing L0 and L1
		L0[:,:,:]= -(dx+dy+dz)/6
		L1[:,:,:]= -(dx+dy+dz)/6
		# Explicit iterative scheme to solve the Laplace equation by the simplest method  (the  Jacobi method).
		# We obtain the harmonic function u(x,y,z) by solving the Laplace equation over R
		n_iter = 200
		for it in range (n_iter):
		    u_n = u
		    u[R[0],R[1],R[2]]= (u_n[R[0]+1,R[1],R[2]] + u_n[R[0]-1,R[1],R[2]] +  u_n[R[0],R[1]+1,R[2]] + \
		    	u_n[R[0],R[1]-1,R[2]] + u_n[R[0],R[1],R[2]+1] + u_n[R[0],R[1],R[2]-1]) / 6
		del u_n
		##Compute the normalized tangent vector field of the correspondence trajectories
		N_xx, N_yy, N_zz = np.gradient(u)
		grad_norm = np.sqrt(N_xx**2 + N_yy**2 + N_zz**2)
		grad_norm[np.where(grad_norm==0)] = 1 ## to avoid dividing by zero
		# Normalization
		np.divide(N_xx, grad_norm, Nx)
		np.divide(N_yy, grad_norm, Ny)
		np.divide(N_zz, grad_norm, Nz)
		del grad_norm, N_xx, N_yy, N_zz
		den = np.absolute(Nx)+ np.absolute(Ny) + np.absolute(Nz)
		den[np.where(den==0)] = 1 ## to avoid dividing by zero
		# iteratively compute correspondence trajectory lengths L0 and L1
		for it in range (100):
		    L0_n = L0
		    L1_n = L1
		    L0[R[0],R[1],R[2]] =  (1 + np.absolute(Nx[R[0],R[1],R[2]])* L0_n[(R[0]-np.sign(Nx[R[0],R[1],R[2]])).astype(int),R[1],R[2]]+ \
		    np.absolute(Ny[R[0],R[1],R[2]]) * L0_n[R[0],(R[1]-np.sign(Ny[R[0],R[1],R[2]])).astype(int),R[2]]  \
		    	+ np.absolute(Nz[R[0],R[1],R[2]]) * L0_n[R[0],R[1],(R[2]-np.sign(Nz[R[0],R[1],R[2]])).astype(int)])  /   den[R[0],R[1],R[2]]
		    L1[R[0],R[1],R[2]] =  (1 + np.absolute(Nx[R[0],R[1],R[2]])* L1_n[(R[0]+np.sign(Nx[R[0],R[1],R[2]])).astype(int),R[1],R[2]]+ \
		    	np.absolute(Ny[R[0],R[1],R[2]]) * L1_n[R[0],(R[1]+np.sign(Ny[R[0],R[1],R[2]])).astype(int),R[2]]+  \
		    	np.absolute(Nz[R[0],R[1],R[2]]) * L1_n[R[0],R[1],(R[2]+np.sign(Nz[R[0],R[1],R[2]])).astype(int)])  /   den[R[0],R[1],R[2]]
		del L0_n, L1_n
		# compute  the thickness of the tissue region inside R
		thicknessMap[R[0],R[1],R[2],l] = L0[R[0],R[1],R[2]] + L1[R[0],R[1],R[2]]

	return thicknessMap



def computeDistanceMapBenefit(Ilabel,distanceMap,newLabel,x,y,z):
	neighbours=get6NeighbourCoordinates(x,y,z)
	currentLabel=Ilabel[x,y,z]
	nbCurrent=[nb for nb in neighbours if Ilabel[nb[0],nb[1],nb[2]]==currentLabel]
	nbCurrentMin=np.min([distanceMap[nb[0],nb[1],nb[2],currentLabel] for nb in nbCurrent])
	##
	nbNew=[nb for nb in neighbours if Ilabel[nb[0],nb[1],nb[2]]==newLabel]
	if len(nbNew)==0:
		nbNewMin=0
	else:
		nbNewMin=np.min([distanceMap[nb[0],nb[1],nb[2],newLabel] for nb in nbNew])
	#nbNewMin=np.min([distanceMap[nb[0],nb[1],nb[2],newLabel] for nb in nbNew])
	###########nbNewMin=np.min([distanceMap[nb[0],nb[1],nb[2],newLabel] for nb in nbNew])
	#if len(bnNew)>=4
	#####nb18=len(get18NeighbourCoordinates(x,y,z))
	#return (nbNewMin-distanceMap[x,y,z,newLabel])#-((5-len(nbNew)/5))#-(nbCurrentMin-distanceMap[x,y,z,currentLabel])####((nb18+12)/18)*
	return (nbNewMin-distanceMap[x,y,z,newLabel])-(nbCurrentMin-distanceMap[x,y,z,currentLabel])####((nb18+12)/18)*





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
	#Ilabel[x,y,z]=newLabel*3
	Ilabel[x,y,z]=newLabel*2
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
						tmp[nb[1][0],nb[1][1],nb[1][2]]=2
					elif(len([nb2 for nb2 in neighbours2 if Ilabel[nb2[0],nb2[1],nb2[2]]==0]) == 0):
						tmp[nb[1][0],nb[1][1],nb[1][2]]=1
				if(label==1):
					if(len([nb2 for nb2 in neighbours2 if Ilabel[nb2[0],nb2[1],nb2[2]]==0]) == 0):
						tmp[nb[1][0],nb[1][1],nb[1][2]]=1
			elif newLabel==1:
				if(label==2):
				 	if(len([nb2 for nb2 in neighbours2 if Ilabel[nb2[0],nb2[1],nb2[2]]==0]) == 0):
						im=Ilabel[nb[0][0]-1:nb[0][0]+2,nb[0][1]-1:nb[0][1]+2,nb[0][2]-1:nb[0][2]+2].copy()
						im=1*(im==1)
						if(isBinarySimplePoint(im)):
							tmp[nb[1][0],nb[1][1],nb[1][2]]=1
				if(label==1):
					if(len([nb2 for nb2 in neighbours2 if Ilabel[nb2[0],nb2[1],nb2[2]]==0]) > 0):
						tmp[nb[1][0],nb[1][1],nb[1][2]]=2
		Ilabel[x-distance:x+distance+1,y-distance:y+distance+1,z-distance:z+distance+1]=tmp


	tmp=1*(Ilabel==2)
	label,numLabel=measure.label(tmp,connectivity=1,return_num=True)
	if numLabel>1:
		labLength=[len(np.where(label==lab)[0]) for lab in range(1,numLabel+1)]
		labMax=labLength.index(np.max(labLength))+1
		for lab in range(1,numLabel+1):
			if lab!=labMax:
				Ilabel[label==lab]=1

	tmp=1*(Ilabel==1)
	label,numLabel=measure.label(tmp,connectivity=1,return_num=True)
	if numLabel>1:
		labLength=[len(np.where(label==lab)[0]) for lab in range(1,numLabel+1)]
		labMax=labLength.index(np.max(labLength))+1
		for lab in range(1,numLabel+1):
			if lab!=labMax:
				Ilabel[label==lab]=2


	return Ilabel,newIlabel


def computeCostMap(Ilabel,newSref,distanceMap,thicknessPriors,totalLabel,terms):
	costMap=np.zeros(np.concatenate((newSref.shape,[len(terms)])))
	index=0
	if 'classification' in terms:
		for l in range(totalLabel):
			tmp=np.zeros(newSref.shape)
			tmp[:,:,:,l]=1*(Ilabel==l)
			costMap[:,:,:,l,index]=np.linalg.norm(tmp-newSref,axis=3)
		index+=1
	if 'distanceMap' in terms:
		for l in range(totalLabel):
			costMap[:,:,:,l,index]=distanceMap[:,:,:,l]*(Ilabel==l)
		index+=1
	if 'geometry' in terms:
		costMap[:,:,:,:,index]=0
		index+=1
	return  costMap


def computeBenefitPos(Ilabel,Sref,distanceMap,thicknessMap,thicknessPriors,x,y,z,newLabel,totalLabel,terms):
	currentVal=np.zeros([totalLabel])
	currentVal[Ilabel[x,y,z]]=1
	nextVal=np.zeros([totalLabel])
	nextVal[newLabel]=1
	benefitMap=np.zeros(len(terms))
	#return the benefit of a point in Ilabel
	#regularisation = distanceMap[x,y,z,Ilabel[x,y,z]] - distanceMap[x,y,z,newLabel]
	benefitMap[:]=0.0
	index=0
	if 'classification' in terms:
		#benefitMap+=(np.linalg.norm(currentVal-Sref[x,y,z,:])-np.linalg.norm(nextVal-Sref[x,y,z,:]))
		benefitMap[index]=(np.linalg.norm(currentVal-Sref[x,y,z,:])-np.linalg.norm(nextVal-Sref[x,y,z,:]))
		index+=1
	if 'distanceMap' in terms:
		#benefitMap+=computeDistanceMapBenefit(Ilabel,distanceMap,newLabel,x,y,z)
		benefitMap[index]+=computeDistanceMapBenefit(Ilabel,distanceMap,newLabel,x,y,z)
		index+=1
	if 'geometry' in terms:
		thicknessMin=thicknessPriors[0]
		thicknessMax=thicknessPriors[1]
		#for val in [Ilabel[x,y,z],newLabel]:
		#	if val in thicknessPriors[2]:
		#		sign=1 if val==newLabel else -1
		#		if thicknessMap[x,y,z,val]>thicknessMax:
		#			#benefitMap+=sign*(thicknessMax-thicknessMap[x,y,z,val])
		#			benefitMap[index]=sign*(thicknessMax-thicknessMap[x,y,z,val])
		#		elif thicknessMap[x,y,z,val]<thicknessMin:
		#			#benefitMap+=sign*(thicknessMin-thicknessMap[x,y,z,val])
		#			benefitMap[index]=sign*(thicknessMin-thicknessMap[x,y,z,val])
		if newLabel in thicknessPriors[2]:
			neighbours=get6NeighbourCoordinates(x,y,z)
			nbNewThickness=[thicknessMap[nb[0],nb[1],nb[2],newLabel] for nb in neighbours if Ilabel[nb[0],nb[1],nb[2]]==newLabel]
			if len(nbNewThickness)!=0:
				newMaxThickness=np.max(nbNewThickness)
				#newMinThickness=np.min(nbNewThickness)
				#newMeanThickness=np.mean(nbNewThickness)
				if newMaxThickness>thicknessMax:
					#benefitMap+=sign*(thicknessMax-thicknessMap[x,y,z,val])
					benefitMap[index]=thicknessMax-newMaxThickness
				elif newMaxThickness<thicknessMin:
					#benefitMap+=sign*(thicknessMin-thicknessMap[x,y,z,val])
					benefitMap[index]=thicknessMin-newMaxThickness
				if (x==17) & (y==22) & (z==30):
					print  nbNewThickness
					print np.max(nbNewThickness)
					print benefitMap[index]
		index+=1
	return  benefitMap

@jit
def initPos(Ilabel,Sref,mask,distanceMap,thicknessMap,thicknessPriors,topologyList,totalLabel,terms):
	simpleMap=np.zeros(Sref.shape,dtype=int)
	benefitMap=np.zeros(np.concatenate((Sref.shape,[len(terms)])))

	## Simple Points ##
	pts=np.where(mask==1)
	for ipt in xrange(len(pts[0])):
		x,y,z=pts[0][ipt],pts[1][ipt],pts[2][ipt]
		simpleMap[x,y,z,:]=isSimple(Ilabel,topologyList,totalLabel,x,y,z) ###NEW###

	## Benefit Map ##
	#pts=np.where(simpleMap==1)
	pts=np.where(mask==1)
	for ipt in xrange(len(pts[0])):
		###############x,y,z,newLabel=pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]
		###############benefitMap[x,y,z,newLabel,:]=computeBenefitPos(Ilabel,Sref,distanceMap,thicknessMap,thicknessPriors,x,y,z,newLabel,totalLabel,terms) ###NEW###
		x,y,z=pts[0][ipt],pts[1][ipt],pts[2][ipt]
		for lab in xrange(totalLabel):
			benefitMap[x,y,z,lab,:]=computeBenefitPos(Ilabel,Sref,distanceMap,thicknessMap,thicknessPriors,x,y,z,lab,totalLabel,terms) ###NEW###


	return simpleMap,benefitMap




def updateMapsPos(Ilabel,Sref,simpleMap,benefitMap,distanceMap,thicknessMap,thicknessPriors,x,y,z,newLabel,topologyList,totalLabel,terms):
	neighbours=getNeighbourCoordinates(x,y,z,Ilabel.shape)
	for nb in neighbours:

		## update simpleMap ##
		simpleMap[nb[0],nb[1],nb[2],:]=isSimple(Ilabel,topologyList,totalLabel,nb[0],nb[1],nb[2]) ###NEW###
		## update benefitMap ##
		for l in range(totalLabel):
			#if simpleMap[nb[0],nb[1],nb[2],l]==1:#########################################################
			#	benefitMap[nb[0],nb[1],nb[2],l,:]=computeBenefitPos(Ilabel,Sref,distanceMap,thicknessMap,thicknessPriors,nb[0],nb[1],nb[2],l,totalLabel,terms) ###NEW###
			#else:
			#	benefitMap[nb[0],nb[1],nb[2],l,:]=0#########################################################
			benefitMap[nb[0],nb[1],nb[2],l,:]=computeBenefitPos(Ilabel,Sref,distanceMap,thicknessMap,thicknessPriors,nb[0],nb[1],nb[2],l,totalLabel,terms) ###NEW##

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

	mode='FarSpheres' # 'CrossEllipsoids' | 'FarSpheres' | 'IMAPA'

	labelsPath=list()
	labelsPath.append(ipath+'/WM/Simapa_wm_atlasdHCP30_K15_hps1_hss3_alpha0.25.nii.gz')
	labelsPath.append(ipath+'/Cortex/Simapa_cortex_atlasdHCP30_K15_hps1_hss3_alpha0.25.nii.gz')
	#labelsPath.append(ipath+'/CSF/Simapa_csf_atlasdHCP30_K15_hps1_hss3_alpha0.25.nii.gz')

	threshold=1
	distance=5
	padding3D=[(distance,distance),(distance,distance),(distance,distance)]
	padding4D=[(distance,distance),(distance,distance),(distance,distance),(0,0)]

	if mode=='IMAPA':
		omega=4
		buildInputs(labelsPath,epath,omega,threshold)

	## Test topological evolution of Ilabel given a positive omega ##
	epath='/home/carlos/Test/Topology/examples'
	opath='/home/carlos/Test/Topology'
	tpath=opath+'/test'
	os.system('mkdir -p '+opath+' '+epath+' '+tpath)
	if mode=='IMAPA':
		IlabelFull = nibabel.load(epath+'/Ilabel_3l.nii.gz')
		SrefFull = nibabel.load(epath+'/Sref_3l.nii.gz')
		omegas = [4,3,2,1]
	elif mode=='FarSpheres':
		IlabelFull = nibabel.load(epath+'/testFarSpheres_Ilabel.nii.gz')
		SrefFull = nibabel.load(epath+'/testFarSpheres_Sref.nii.gz')
		omegas = [0]
	elif mode=='CrossEllipsoids':
		IlabelFull = nibabel.load(epath+'/testCrossEllipsoids_Ilabel.nii.gz')
		SrefFull = nibabel.load(epath+'/testCrossEllipsoids_Sref.nii.gz')
		omegas = [0]
	Ilabel = IlabelFull.get_data().astype(int)
	Sref = SrefFull.get_data()


	totalLabel=3


	thresholdLoopPrint=10000

	#union='_dist'

	for omega in omegas:


		thicknessSize=np.array(SrefFull.header.get_zooms()[:-1])/(2**(omega))
		#thicknessSize=[1,1,1]#############################################################################
		thicknessMin=2
		thicknessMax=4
		#thicknessLabels=[2,3]
		thicknessLabels=[2]
		thicknessPriors=[thicknessMin,thicknessMax,thicknessLabels]

		iterations=range(1)

		for iter in iterations:

			#terms=['distanceMap','classification']
			terms=['classification']
			union='_'.join(terms)
			#if (iter==0):
			#	union='_dist'
			#	terms=['distanceMap','classification']
			#	#if omega>2:
			#	#	terms=['distanceMap','classification']
			#	#else:
			#	#	terms=['distanceMap','classification','geometry']
			#elif (iter==1):
			#	union='_nodist'
			#	terms=['classification']
			#	#if omega>2:
			#	#	terms=['classification']
			#	#else:
			#	#	terms=['classification','geometry']

			Ilabel=np.pad(Ilabel,padding3D,mode='constant', constant_values=0)
			mask=np.zeros(Ilabel.shape)
			mask[distance:-distance-1,distance:-distance-1,distance:-distance-1]=1

			## Downsampling Sref ##
			newSref=downsampling(Sref,omega)

			newSref=np.pad(newSref,padding4D,mode='constant', constant_values=0)

			## Distance Maps ##
			distanceMap=computeDistanceMap(newSref,totalLabel)
			nibabel.save(nibabel.Nifti1Image(distanceMap.astype(float),IlabelFull.affine),opath+'/distanceMap_'+mode+'_omega'+str(omega)+'.nii.gz')



			#nibabel.save(nibabel.Nifti1Image(newSref.astype(float),IlabelFull.affine),opath+'/newSref_'+mode+'_omega'+str(omega)+'.nii.gz')


			#thicknessMap=computeThicknessMap(1*(Ilabel==2),1*(Ilabel==1),1*(Ilabel==0),thicknessSize)
			thicknessMap=computeThicknessMap(Ilabel,thicknessSize,thicknessLabels,totalLabel)
			nibabel.save(nibabel.Nifti1Image(thicknessMap.astype(float),IlabelFull.affine),opath+'/thicknessMap_'+mode+'_omega'+str(omega)+'.nii.gz')


			topologyList=[[3],[2],[1],[0]]
			#topologyList=[[2],[1],[0]]
			simpleMap,benefitMap=initPos(Ilabel,newSref,mask,distanceMap,thicknessMap,thicknessPriors,topologyList,totalLabel,terms)
			costMap=computeCostMap(Ilabel,newSref,distanceMap,thicknessPriors,totalLabel,terms)
			sumCost=np.sum(benefitMap,axis=4)*(simpleMap==1)


			print str(loops)+' loops, \t'+str(len(np.where(sumCost>0)[0]))+' simple points'
			nibabel.save(nibabel.Nifti1Image(Ilabel.astype(float),IlabelFull.affine),tpath+'/Ilabel_'+mode+'_'+union+'_omega'+str(omega)+'_initial.nii.gz')
			nibabel.save(nibabel.Nifti1Image(simpleMap.astype(float),IlabelFull.affine),tpath+'/simpleMap_'+mode+'_'+union+'_omega'+str(omega)+'_initial.nii.gz')
			for t in range(len(terms)):
				nibabel.save(nibabel.Nifti1Image(benefitMap[:,:,:,:,t].astype(float),IlabelFull.affine),tpath+'/benefit'+str(t)+'Map_'+mode+'_'+union+'_omega'+str(omega)+'_initial.nii.gz')
				nibabel.save(nibabel.Nifti1Image(costMap[:,:,:,:,t].astype(float),IlabelFull.affine),tpath+'/cost'+str(t)+'Map_'+mode+'_'+union+'_omega'+str(omega)+'_initial.nii.gz')
			#nibabel.save(nibabel.Nifti1Image(benefitMap.astype(float),IlabelFull.affine),tpath+'/benefitMap_'+mode+'_'+union+'_omega'+str(omega)+'_initial.nii.gz')
			nibabel.save(nibabel.Nifti1Image(thicknessMap.astype(float),IlabelFull.affine),tpath+'/thicknessMap_'+mode+'_'+union+'_omega'+str(omega)+'_initial.nii.gz')

			#nibabel.save(nibabel.Nifti1Image(simpleMap.astype(float),IlabelFull.affine),opath+'/simpleMap_'+mode+'_omega'+str(omega)+'.nii.gz')

			#print '\nSize of Ilabel = \t'+str(simpleMap.shape)+'\n\t\t\t'+str(benefitMap.shape)+'\n'


			loops=0
			while(len(np.where( (sumCost>0) )[0])>0):


				if (loops%thresholdLoopPrint==0):# | (loops==4):
					print str(loops)+' loops, \t'+str(len(np.where(sumCost>0)[0]))+' simple points'
					nibabel.save(nibabel.Nifti1Image(Ilabel.astype(float),IlabelFull.affine),tpath+'/Ilabel_'+mode+'_iter'+str(int(loops/thresholdLoopPrint))+'.nii.gz')
					nibabel.save(nibabel.Nifti1Image(simpleMap.astype(float),IlabelFull.affine),tpath+'/simpleMap_'+mode+'_iter'+str(int(loops/thresholdLoopPrint))+'.nii.gz')
					for t in range(len(terms)):
						nibabel.save(nibabel.Nifti1Image(benefitMap[:,:,:,:,t].astype(float),IlabelFull.affine),tpath+'/benefit'+str(t)+'Map_'+mode+'_iter'+str(int(loops/thresholdLoopPrint))+'.nii.gz')
						nibabel.save(nibabel.Nifti1Image(costMap[:,:,:,:,t].astype(float),IlabelFull.affine),tpath+'/cost'+str(t)+'Map_'+mode+'_iter'+str(int(loops/thresholdLoopPrint))+'.nii.gz')
					#nibabel.save(nibabel.Nifti1Image(benefitMap.astype(float),IlabelFull.affine),tpath+'/benefitMap_'+mode+'_iter'+str(int(loops/thresholdLoopPrint))+'.nii.gz')
					nibabel.save(nibabel.Nifti1Image(thicknessMap.astype(float),IlabelFull.affine),tpath+'/thicknessMap_'+mode+'_iter'+str(int(loops/thresholdLoopPrint))+'.nii.gz')
					maxBenefit=np.max(sumCost)
					print maxBenefit
					#sys.exit()
				maxBenefit=np.max(sumCost)
				pt=np.where(sumCost==maxBenefit)

				x,y,z,newLabel=pt[0][0],pt[1][0],pt[2][0],pt[3][0]
				## Update values ##
				Ilabel[x,y,z]=newLabel
				#if loops%1==0:
				#thicknessMap=computeThicknessMap(1*(Ilabel==2),1*(Ilabel==1),1*(Ilabel==0),thicknessSize)
				####thicknessMap=computeThicknessMap(Ilabel,thicknessSize,thicknessLabels,totalLabel)
				simpleMap,benefitMap=updateMapsPos(Ilabel,newSref,simpleMap,benefitMap,distanceMap,thicknessMap,thicknessPriors,x,y,z,newLabel,topologyList,totalLabel,terms)
				costMap=computeCostMap(Ilabel,newSref,distanceMap,thicknessPriors,totalLabel,terms)
				sumCost=np.sum(benefitMap,axis=4)*(simpleMap==1)

				loops+=1

			#nibabel.save(nibabel.Nifti1Image(Ilabel.astype(float),IlabelFull.affine),opath+'/Ilabel_'+mode+'_pad_omega'+str(omega)+union+'.nii.gz')
			Ilabel=Ilabel[padding3D[0][0]:-padding3D[0][1],padding3D[1][0]:-padding3D[1][1],padding3D[2][0]:-padding3D[2][1]]
			nibabel.save(nibabel.Nifti1Image(Ilabel.astype(float),IlabelFull.affine),opath+'/Ilabel_'+mode+'_omega'+str(omega)+union+'.nii.gz')

			print '\nConnected components of Ilabel = '+str(checkConnectedComponents(Ilabel,totalLabel))+'\n'

		print str(loops)+' loops, \t'+str(len(np.where(sumCost>0)[0]))+' simple points'
		nibabel.save(nibabel.Nifti1Image(Ilabel.astype(float),IlabelFull.affine),tpath+'/Ilabel_'+mode+'_'+union+'_omega'+str(omega)+'_final.nii.gz')
		nibabel.save(nibabel.Nifti1Image(simpleMap.astype(float),IlabelFull.affine),tpath+'/simpleMap_'+mode+'_'+union+'_omega'+str(omega)+'_final.nii.gz')
		for t in range(len(terms)):
			nibabel.save(nibabel.Nifti1Image(benefitMap[:,:,:,:,t].astype(float),IlabelFull.affine),tpath+'/benefit'+str(t)+'Map_'+mode+'_'+union+'_omega'+str(omega)+'_final.nii.gz')
			nibabel.save(nibabel.Nifti1Image(costMap[:,:,:,:,t].astype(float),IlabelFull.affine),tpath+'/cost'+str(t)+'Map_'+mode+'_'+union+'_omega'+str(omega)+'_final.nii.gz')
		#nibabel.save(nibabel.Nifti1Image(benefitMap.astype(float),IlabelFull.affine),tpath+'/benefitMap_'+mode+'_'+union+'_omega'+str(omega)+'_final.nii.gz')
		nibabel.save(nibabel.Nifti1Image(thicknessMap.astype(float),IlabelFull.affine),tpath+'/thicknessMap_'+mode+'_'+union+'_omega'+str(omega)+'_final.nii.gz')

		Ilabel=upsampling3D(Ilabel,2).astype(int)
		IlabelFull = nibabel.load(opath+'/Ilabel_'+mode+'_omega'+str(omega)+union+'.nii.gz')
