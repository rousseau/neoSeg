# coding=utf-8

import os,sys
import numpy as np
import math
import nibabel
import random
from skimage import measure
from scipy import ndimage
from numba import jit
from simplicity import isSimple6_26, isSimplePair6_26, isSimplePair26_6, isSimplePair6_26Label
from distances import computeBenefitPosDistance, computeBenefitPairPosDistance, computeBenefitPosSimilarity, computeBenefitPairPosSimilarity, \
						computeBenefitNegDistance, computeBenefitNegSimilarity




def initPos(Ilabel,Sref,distanceMap,topologyList,totalLabel,metric,padding):
	simpleMap=np.zeros(Sref.shape,dtype=int)
	benefitMap=np.zeros(Sref.shape)

	## Simple Points ##
	#pts=np.where(mask==1)
	#for ipt in xrange(len(pts[0])):
	#x,y,z=pts[0][ipt],pts[1][ipt],pts[2][ipt]
	for x in xrange(padding,Ilabel.shape[0]-padding):
		for y in xrange(padding,Ilabel.shape[1]-padding):
			for z in xrange(padding,Ilabel.shape[2]-padding):
				simpleMap[x,y,z,:]=isSimple6_26(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2],topologyList,totalLabel) ###NEW###

	## Benefit Map ##

	pts=np.where(simpleMap==1)
	if metric=='similarity':
		for ipt in xrange(len(pts[0])):
			#x,y,z,l=pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]]=computeBenefitPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],Sref[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric=='distanceMap':
		for ipt in xrange(len(pts[0])):
			#x,y,z,l=pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]]=computeBenefitPosDistance(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt])
			#benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]]=computeBenefitPosGradientDistance(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt])
	#benefitMap*=simpleMap

	return simpleMap,benefitMap

def initAtlasPos(Ilabel,Sref,atlasPrior,distanceMap,topologyList,topologyAtlasList,totalLabel,metric,padding):
	simpleMap=np.zeros(Sref.shape,dtype=int)
	benefitMap=np.zeros(Sref.shape)

	## Simple Points ##
	#pts=np.where(mask==1)
	#for ipt in xrange(len(pts[0])):
	#x,y,z=pts[0][ipt],pts[1][ipt],pts[2][ipt]
	for x in xrange(padding,Ilabel.shape[0]-padding):
		for y in xrange(padding,Ilabel.shape[1]-padding):
			for z in xrange(padding,Ilabel.shape[2]-padding):
				if atlasPrior[x,y,z]==0:
					simpleMap[x,y,z,:]=isSimple6_26(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2],topologyList,totalLabel) ###NEW###
				else:
					simpleMap[x,y,z,:]=isSimple6_26(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2],topologyAtlasList,totalLabel)

	## Benefit Map ##

	pts=np.where(simpleMap==1)
	if metric=='similarity':
		for ipt in xrange(len(pts[0])):
			#x,y,z,l=pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]]=computeBenefitPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],Sref[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric=='distanceMap':
		for ipt in xrange(len(pts[0])):
			#x,y,z,l=pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]]=computeBenefitPosDistance(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt])
			#benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]]=computeBenefitPosGradientDistance(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt])
	#benefitMap*=simpleMap

	return simpleMap,benefitMap



def initSimplePairPos(Ilabel,newSref,simpleMap,distanceMap,topologyList,totalLabel,metric,padding):
	simplePairMap = np.zeros((Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],3,totalLabel))
	benefitPairMap = np.zeros((Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],3,totalLabel))

	## Simple Pairs ##
	##Binary mask avoiding homogeneous ares, simple points and padding##
	homogeneous = computeHomogeneousArea(Ilabel,totalLabel)
	paddingImage = np.ones_like(Ilabel)
	paddingImage[padding:Ilabel.shape[0]-padding,padding:Ilabel.shape[1]-padding,padding:Ilabel.shape[2]-padding]=0
	for l in xrange(totalLabel):
		mask = 1*((simpleMap[:,:,:,l] + homogeneous + paddingImage)!=0)
		mask = (1-mask)
		##Simple pair in x-axis##
		maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0,0,0]],[[1,-1,0]],[[0,0,0]]]))==0)
		ct = np.where(maskFiltered==1)
		for ict in range(len(ct[0])):
			simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],2,l] = isSimplePair6_26Label( Ilabel[ct[0][ict]-1:ct[0][ict]+2,ct[1][ict]-1:ct[1][ict]+2,ct[2][ict]-1:ct[2][ict]+3],\
																							([1,1],[1,1],[1,2]),l,topologyList,totalLabel )
		##Simple pair in y-axis##
		maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0],[0],[0]],[[1],[-1],[0]],[[0],[0],[0]]]))==0)
		ct = np.where(maskFiltered==1)
		for ict in range(len(ct[0])):
			simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],1,l] = isSimplePair6_26Label( Ilabel[ct[0][ict]-1:ct[0][ict]+2,ct[1][ict]-1:ct[1][ict]+3,ct[2][ict]-1:ct[2][ict]+2],\
																							([1,1],[1,2],[1,1]),l,topologyList,totalLabel )
		##Simple pair in z-axis##
		maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0],[1],[0]],[[0],[-1],[0]],[[0],[0],[0]]]))==0)
		ct = np.where(maskFiltered==1)
		for ict in range(len(ct[0])):
			simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],0,l] = isSimplePair6_26Label( Ilabel[ct[0][ict]-1:ct[0][ict]+3,ct[1][ict]-1:ct[1][ict]+2,ct[2][ict]-1:ct[2][ict]+2],\
																							([1,2],[1,1],[1,1]),l,topologyList,totalLabel )

	## Benefit Map ##
	pts=np.where(simplePairMap[:,:,:,0,:]==1)
	if metric=='similarity':
		for ipt in xrange(len(pts[0])):
			benefitPairMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],2,pts[3][ipt]]=computeBenefitPairPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],Sref[pts[0][ipt],pts[1][ipt],\
																												pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric=='distanceMap':
		for ipt in xrange(len(pts[0])):
			benefitPairMap[pts[0][ipt],pts[1][ipt],pts[2][ipt]:pts[2][ipt]+2,2,pts[3][ipt]]=computeBenefitPairPosDistance(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],\
																						distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt]:pts[2][ipt]+2,:],pts[3][ipt])

	pts=np.where(simplePairMap[:,:,:,1,:]==1)
	if metric=='similarity':
		for ipt in xrange(len(pts[0])):
			benefitPairMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],1,pts[3][ipt]]=computeBenefitPairPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],Sref[pts[0][ipt],pts[1][ipt],\
																												pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric=='distanceMap':
		for ipt in xrange(len(pts[0])):
			benefitPairMap[pts[0][ipt],pts[1][ipt]:pts[1][ipt]+2,pts[2][ipt],1,pts[3][ipt]]=computeBenefitPairPosDistance(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],\
																						distanceMap[pts[0][ipt],pts[1][ipt]:pts[1][ipt]+2,pts[2][ipt],:],pts[3][ipt])

	pts=np.where(simplePairMap[:,:,:,2,:]==1)
	if metric=='similarity':
		for ipt in xrange(len(pts[0])):
			benefitPairMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],0,pts[3][ipt]]=computeBenefitPairPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],Sref[pts[0][ipt],pts[1][ipt],\
																												pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric=='distanceMap':
		for ipt in xrange(len(pts[0])):
			benefitPairMap[pts[0][ipt]:pts[0][ipt]+2,pts[1][ipt],pts[2][ipt],0,pts[3][ipt]]=computeBenefitPairPosDistance(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],\
																						distanceMap[pts[0][ipt]:pts[0][ipt]+2,pts[1][ipt],pts[2][ipt],:],pts[3][ipt])

	return simplePairMap,benefitPairMap


def initSimplePairAtlasPos(Ilabel,newSref,simpleMap,atlasPrior,distanceMap,topologyList,topologyAtlasList,totalLabel,metric,padding):

	simplePairMap = np.zeros((Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],3,totalLabel))
	benefitPairMap = np.zeros((Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],3,totalLabel))

	## Simple Pairs ##
	##Binary mask avoiding homogeneous ares, simple points and padding##
	homogeneous = computeHomogeneousArea(Ilabel,totalLabel)
	paddingImage = np.ones_like(Ilabel)
	paddingImage[padding:Ilabel.shape[0]-padding,padding:Ilabel.shape[1]-padding,padding:Ilabel.shape[2]-padding]=0
	for l in xrange(totalLabel):
		##Topological atlas##
		mask = 1*((simpleMap[:,:,:,l] + homogeneous + paddingImage)!=0)
		mask = (1*(atlasPrior!=0))*(1-mask)
		##Simple pair in x-axis##
		maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0,0,0]],[[1,-1,0]],[[0,0,0]]]))==0)
		ct = np.where(maskFiltered==1)
		for ict in range(len(ct[0])):
			simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],2,l] = isSimplePair6_26Label( Ilabel[ct[0][ict]-1:ct[0][ict]+2,ct[1][ict]-1:ct[1][ict]+2,ct[2][ict]-1:ct[2][ict]+3],\
																							([1,1],[1,1],[1,2]),l,topologyAtlasList,totalLabel )
		##Simple pair in y-axis##
		maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0],[0],[0]],[[1],[-1],[0]],[[0],[0],[0]]]))==0)
		ct = np.where(maskFiltered==1)
		for ict in range(len(ct[0])):
			simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],1,l] = isSimplePair6_26Label( Ilabel[ct[0][ict]-1:ct[0][ict]+2,ct[1][ict]-1:ct[1][ict]+3,ct[2][ict]-1:ct[2][ict]+2],\
																							([1,1],[1,2],[1,1]),l,topologyAtlasList,totalLabel )
		##Simple pair in z-axis##
		maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0],[1],[0]],[[0],[-1],[0]],[[0],[0],[0]]]))==0)
		ct = np.where(maskFiltered==1)
		for ict in range(len(ct[0])):
			simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],0,l] = isSimplePair6_26Label( Ilabel[ct[0][ict]-1:ct[0][ict]+3,ct[1][ict]-1:ct[1][ict]+2,ct[2][ict]-1:ct[2][ict]+2],\
																							([1,2],[1,1],[1,1]),l,topologyAtlasList,totalLabel )

		##Rest of Ilabel##
		mask = 1*((simpleMap[:,:,:,l] + homogeneous + paddingImage)!=0)
		mask = (1*(atlasPrior==0))*(1-mask)
		##Simple pair in x-axis##
		maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0,0,0]],[[1,-1,0]],[[0,0,0]]]))==0)
		ct = np.where(maskFiltered==1)
		for ict in range(len(ct[0])):
			simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],2,l] = isSimplePair6_26Label( Ilabel[ct[0][ict]-1:ct[0][ict]+2,ct[1][ict]-1:ct[1][ict]+2,ct[2][ict]-1:ct[2][ict]+3],\
																							([1,1],[1,1],[1,2]),l,topologyList,totalLabel )
		##Simple pair in y-axis##
		maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0],[0],[0]],[[1],[-1],[0]],[[0],[0],[0]]]))==0)
		ct = np.where(maskFiltered==1)
		for ict in range(len(ct[0])):
			simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],1,l] = isSimplePair6_26Label( Ilabel[ct[0][ict]-1:ct[0][ict]+2,ct[1][ict]-1:ct[1][ict]+3,ct[2][ict]-1:ct[2][ict]+2],\
																							([1,1],[1,2],[1,1]),l,topologyList,totalLabel )
		##Simple pair in z-axis##
		maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0],[1],[0]],[[0],[-1],[0]],[[0],[0],[0]]]))==0)
		ct = np.where(maskFiltered==1)
		for ict in range(len(ct[0])):
			simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],0,l] = isSimplePair6_26Label( Ilabel[ct[0][ict]-1:ct[0][ict]+3,ct[1][ict]-1:ct[1][ict]+2,ct[2][ict]-1:ct[2][ict]+2],\
																							([1,2],[1,1],[1,1]),l,topologyList,totalLabel )

	## Benefit Map ##
	pts=np.where(simplePairMap[:,:,:,0,:]==1)
	if metric=='similarity':
		for ipt in xrange(len(pts[0])):
			benefitPairMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],2,pts[3][ipt]]=computeBenefitPairPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],\
																					Sref[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric=='distanceMap':
		for ipt in xrange(len(pts[0])):
			benefitPairMap[pts[0][ipt],pts[1][ipt],pts[2][ipt]:pts[2][ipt]+2,2,pts[3][ipt]]=computeBenefitPairPosDistance(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],\
																					distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt]:pts[2][ipt]+2,:],pts[3][ipt])

	pts=np.where(simplePairMap[:,:,:,1,:]==1)
	if metric=='similarity':
		for ipt in xrange(len(pts[0])):
			benefitPairMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],1,pts[3][ipt]]=computeBenefitPairPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],\
																					Sref[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric=='distanceMap':
		for ipt in xrange(len(pts[0])):
			benefitPairMap[pts[0][ipt],pts[1][ipt]:pts[1][ipt]+2,pts[2][ipt],1,pts[3][ipt]]=computeBenefitPairPosDistance(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],\
																					distanceMap[pts[0][ipt],pts[1][ipt]:pts[1][ipt]+2,pts[2][ipt],:],pts[3][ipt])

	pts=np.where(simplePairMap[:,:,:,2,:]==1)
	if metric=='similarity':
		for ipt in xrange(len(pts[0])):
			benefitPairMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],1,pts[3][ipt]]=computeBenefitPairPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],\
																					Sref[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric=='distanceMap':
		for ipt in xrange(len(pts[0])):
			benefitPairMap[pts[0][ipt]:pts[0][ipt]+2,pts[1][ipt],pts[2][ipt],1,pts[3][ipt]]=computeBenefitPairPosDistance(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],\
																					distanceMap[pts[0][ipt]:pts[0][ipt]+2,pts[1][ipt],pts[2][ipt],:],pts[3][ipt])

	return simplePairMap,benefitPairMap







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
		for it in xrange (n_iter):
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
		for it in xrange (100):
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



def computeCostMap(Ilabel,newSref,distanceMap,thicknessPriors,totalLabel,terms):
	costMap=np.zeros(np.concatenate((newSref.shape,[len(terms)])))
	index=0
	if 'classification' in terms:
		for l in xrange(totalLabel):
			tmp=np.zeros(newSref.shape)
			tmp[:,:,:,l]=1*(Ilabel==l)
			costMap[:,:,:,l,index]=np.linalg.norm(tmp-newSref,axis=3)
		index+=1
	if 'distanceMap' in terms:
		for l in xrange(totalLabel):
			costMap[:,:,:,l,index]=distanceMap[:,:,:,l]*(Ilabel==l)
		index+=1
	if 'geometry' in terms:
		costMap[:,:,:,:,index]=0
		index+=1

	return  costMap



'''##########################################'''
'''#################NEGATIVE#################'''
'''##########################################'''


def computeProportionLabel(Ilabel,Sref,xref,yref,zref,omega,totalLabel):
	'''
	xref,yref,zref are in Sref space
	'''
	proprotionLabel=np.zeros(totalLabel)
	im=Ilabel[int(xref/(2**omega)):int((xref+1)/(2**omega)),int(yref/(2**omega)):int((yref+1)/(2**omega)),int(zref/(2**omega)):int((zref+1)/(2**omega))]

	###x, y, z = xref*2**omega, yref*2**omega, zref*2**omega
	###im=Ilabel[x-(2**omega)/2:x+(2**omega)/2+1,y-(2**omega)/2:y+(2**omega)/2+1,z-(2**omega)/2:z+(2**omega)/2
	for l in xrange(totalLabel):
		proprotionLabel[l]=len(im[im==l])
	proprotionLabel/=im.size
	return proprotionLabel


def computeBenefitNeg(Ilabel,Sref,proportionLabelMap,x,y,z,newLabel,totalLabel):
	'''
	x,y,z are in Ilabel space
	xref,yref,zref are in Sref space
	'''
	xref,yref,zref=int(x*(2**omega)),int(y*(2**omega)),int(z*(2**omega))
	newIlabel=Ilabel.copy()
	newIlabel[x,y,z]=newLabel
	nextProportionalLabelValue=computeProportionLabel(newIlabel,Sref,xref,yref,zref,omega,totalLabel)
	#return np.linalg.norm(newProportionalLabelValue)-np.linalg.norm(proportionLabelMap[x,y,z])
	#return np.linalg.norm(currentVal-Sref[x,y,z,:])-np.linalg.norm(nextVal-Sref[x,y,z,:])
	#return the benefit of a point in Ilabel
	return np.linalg.norm(proportionLabelMap[xref,yref,zref,:]-Sref[xref,yref,zref,:])-np.linalg.norm(nextProportionalLabelValue-Sref[xref,yref,zref,:])





def initNeg(Ilabel,Sref,distanceMap,topologyList,totalLabel,metric,padding,omega):
	simpleMap = np.zeros([Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],totalLabel],dtype=int)
	proportionLabelMap = np.zeros(Sref.shape)
	benefitMap = np.zeros([Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],totalLabel])

	resolutionGap = int((2**abs(omega))**3)

	'''######################'''
	# opath='/home/carlos/Test/Topology'
	# IlabelFull = nibabel.load(opath+'/examples/Ilabel_3l.nii.gz')
	'''######################'''

	## Simple Points ##
	##Binary mask avoiding homogeneous ares, simple points and padding##
	heterogeneous = np.zeros([Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],totalLabel])
	paddingImage = np.zeros_like(Ilabel)
	paddingImage[padding*(2**abs(omega)):Ilabel.shape[0]-padding*(2**abs(omega)),padding*(2**abs(omega)):Ilabel.shape[1]-padding*(2**abs(omega)),padding*(2**abs(omega)):Ilabel.shape[2]-padding*(2**abs(omega))] = 1
	for l in xrange(totalLabel):
		binIm = 1*(Ilabel==l)
		heterogeneous[:,:,:,l] = paddingImage*(1-1*(ndimage.convolve(binIm,np.array([[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,-26,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]))==0))

	# for l in xrange(totalLabel):
	# 	nibabel.save(nibabel.Nifti1Image(heterogeneous[:,:,:,l].astype(float),IlabelFull.affine),opath+'/heterogeneous_label'+str(l)+'_omega'+str(omega)+'.nii.gz')

	# print str(np.count_nonzero(heterogeneous))+' of '+str(heterogeneous.size)+' --> '+str(100*np.count_nonzero(heterogeneous)/heterogeneous.size)+' %'+str(resolutionGap)

	pts = np.where(heterogeneous==1)
	x = chunkIt(pts[0],resolutionGap)
	y = chunkIt(pts[1],resolutionGap)
	z = chunkIt(pts[2],resolutionGap)

	for part in xrange(resolutionGap):
		for ipt in xrange(len(x[part])):
			simpleMap[x[part][ipt],y[part][ipt],z[part][ipt],:] = isSimple6_26(Ilabel[x[part][ipt]-1:x[part][ipt]+2,y[part][ipt]-1:y[part][ipt]+2,z[part][ipt]-1:z[part][ipt]+2],topologyList,totalLabel)

	# for l in xrange(totalLabel):
	# 	nibabel.save(nibabel.Nifti1Image(simpleMap[:,:,:,l].astype(float),IlabelFull.affine),opath+'/simpleMap_label'+str(l)+'_omega'+str(omega)+'.nii.gz')

	# #padding/=int(2**np.abs(omega))
	for xref in xrange(padding,Sref.shape[0]-padding):
		for yref in xrange(padding,Sref.shape[1]-padding):
			for zref in xrange(padding,Sref.shape[2]-padding):
					proportionLabelMap[xref,yref,zref,:]=computeProportionLabel(Ilabel,Sref,xref,yref,zref,omega,totalLabel)

	# for l in xrange(totalLabel):
	# 	nibabel.save(nibabel.Nifti1Image(proportionLabelMap[:,:,:,l].astype(float),IlabelFull.affine),opath+'/proportionLabelMap_label'+str(l)+'_omega'+str(omega)+'.nii.gz')


	#######LOAD VERSION######
	# nibabel.save(nibabel.Nifti1Image(Ilabel.astype(float),IlabelFull.affine),opath+'/IlabelUpsampled_omega'+str(omega)+'.nii.gz')
	# for l in xrange(totalLabel):
	# 	proportionLabelMap[:,:,:,l] = nibabel.load(opath+'/proportionLabelMap_label'+str(l)+'_omega'+str(omega)+'.nii.gz').get_data()
	# 	simpleMap[:,:,:,l] = nibabel.load(opath+'/simpleMap_label'+str(l)+'_omega'+str(omega)+'.nii.gz').get_data()
	#######LOAD VERSION######

	###########TEST POINT#############
	# pts=np.where(simpleMap==1)
	# ipt=100
	# xref,yref,zref = int(pts[0][ipt]*(2**omega)),int(pts[1][ipt]*(2**omega)),int(pts[2][ipt]*(2**omega))
	# print '\nIlabel point '+str(pts[0][ipt]),str(pts[1][ipt]),str(pts[2][ipt]),str(pts[3][ipt])
	# print 'Sref point '+str(xref),str(yref),str(zref)
	# currentProportionIlabelPoint = proportionLabelMap[xref,yref,zref,:]
	# newProportionIlabelPoint = currentProportionIlabelPoint.copy()
	# newProportionIlabelPoint[ Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]] ] -= 1./resolutionGap #delete new element
	# newProportionIlabelPoint[ pts[3][ipt] ] += 1./resolutionGap #add new element
	# distanceMapPoint = distanceMap[xref,yref,zref,:]
	# #benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitNegDistance(currentProportionIlabelPoint,newProportionIlabelPoint,distanceMapPoint)
	# print 'Current norm '+str(np.linalg.norm(currentProportionIlabelPoint*distanceMapPoint))
	# print '\tcurrent proportion '+str(currentProportionIlabelPoint)
	# print 'New norm '+str(np.linalg.norm(newProportionIlabelPoint*distanceMapPoint))
	# print '\tnew proportion '+str(newProportionIlabelPoint)
	# print 'DistanceMap array '+str(distanceMapPoint)
	# if np.linalg.norm(currentProportionIlabelPoint*distanceMapPoint) - np.linalg.norm(newProportionIlabelPoint*distanceMapPoint) > 0:
	# 	benefit = np.linalg.norm(currentProportionIlabelPoint*distanceMapPoint)
	# else:
	# 	benefit = 0
	# print 'Result: '+str(benefit)
	# sys.exit()
	###########TEST POINT#############

	## Benefit Map ##
	pts=np.where(simpleMap==1)
	print str(len(pts[0]))+' simple points detected'
	if metric=='similarity':
		for ipt in xrange(len(pts[0])):
			xref,yref,zref = int(pts[0][ipt]*(2**omega)),int(pts[1][ipt]*(2**omega)),int(pts[2][ipt]*(2**omega))
			currentProportionIlabelPoint = proportionLabelMap[xref,yref,zref,:]
			newProportionIlabelPoint = currentProportionIlabelPoint.copy()
			newProportionIlabelPoint[ Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]] ] -= 1./resolutionGap #delete new element
			newProportionIlabelPoint[ pts[3][ipt] ] += 1./resolutionGap #add new element
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitNegSimilarity(currentProportionIlabelPoint,newProportionIlabelPoint,Sref[xref,yref,zref,:])
	elif metric=='distanceMap':
		for ipt in xrange(len(pts[0])):
			xref,yref,zref = int(pts[0][ipt]*(2**omega)),int(pts[1][ipt]*(2**omega)),int(pts[2][ipt]*(2**omega))
			currentProportionIlabelPoint = proportionLabelMap[xref,yref,zref,:]
			newProportionIlabelPoint = currentProportionIlabelPoint.copy()
			newProportionIlabelPoint[ Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]] ] -= 1./resolutionGap #delete new element
			newProportionIlabelPoint[ pts[3][ipt] ] += 1./resolutionGap #add new element
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitNegDistance(currentProportionIlabelPoint,newProportionIlabelPoint,distanceMap[xref,yref,zref,:])

	# for l in xrange(totalLabel):
	# 	nibabel.save(nibabel.Nifti1Image(benefitMap[:,:,:,l].astype(float),IlabelFull.affine),opath+'/benefitMap_label'+str(l)+'_omega'+str(omega)+'.nii.gz')

	return simpleMap,proportionLabelMap,benefitMap


def initAtlasNeg(Ilabel,Sref,atlasPrior,distanceMap,topologyList,topologyAtlasList,totalLabel,metric,padding,omega):
	simpleMap = np.zeros([Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],totalLabel],dtype=int)
	proportionLabelMap = np.zeros(Sref.shape)
	benefitMap = np.zeros([Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],totalLabel])

	resolutionGap = int((2**abs(omega))**3)

	'''######################'''
	# opath='/home/carlos/Test/Topology'
	# IlabelFull = nibabel.load(opath+'/examples/Ilabel_3l.nii.gz')
	'''######################'''

	## Simple Points ##
	##Binary mask avoiding homogeneous ares, simple points and padding##
	heterogeneous = np.zeros([Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],totalLabel])
	paddingImage = np.zeros_like(Ilabel)
	paddingImage[padding*(2**abs(omega)):Ilabel.shape[0]-padding*(2**abs(omega)),padding*(2**abs(omega)):Ilabel.shape[1]-padding*(2**abs(omega)),padding*(2**abs(omega)):Ilabel.shape[2]-padding*(2**abs(omega))] = 1
	for l in xrange(totalLabel):
		binIm = 1*(Ilabel==l)
		heterogeneous[:,:,:,l] = paddingImage*(1-1*(ndimage.convolve(binIm,np.array([[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,-26,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]))==0))

	# for l in xrange(totalLabel):
	# 	nibabel.save(nibabel.Nifti1Image(heterogeneous[:,:,:,l].astype(float),IlabelFull.affine),opath+'/homogeneous_label'+str(l)+'_omega'+str(omega)+'.nii.gz')

	pts = np.where(heterogeneous==1)
	print str(len(pts[0]))+' of '+str(heterogeneous.size)+' --> '+str(100*len(pts[0])/heterogeneous.size)+' %'

	for ipt in xrange(len(pts[0])):
		if atlasPrior[pts[0][ipt],pts[1][ipt],pts[2][ipt]]==0:
			simpleMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],:]=isSimple6_26(Ilabel[pts[0][ipt]-1:pts[0][ipt]+2,pts[1][ipt]-1:pts[1][ipt]+2,pts[2][ipt]-1:pts[2][ipt]+2],topologyList,totalLabel)
		else:
			simpleMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],:]=isSimple6_26(Ilabel[pts[0][ipt]-1:pts[0][ipt]+2,pts[1][ipt]-1:pts[1][ipt]+2,pts[2][ipt]-1:pts[2][ipt]+2],topologyAtlasList,totalLabel)

	# for x in xrange(padding,Ilabel.shape[0]-padding):
	# 	for y in xrange(padding,Ilabel.shape[1]-padding):
	# 		for z in xrange(padding,Ilabel.shape[2]-padding):
	# 			if heterogeneous[x,y,z]==1:
	# 				if atlasPrior[x,y,z]==0:
	# 					simpleMap[x,y,z,:]=isSimple6_26(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2],topologyList,totalLabel)
	# 				else:
	# 					simpleMap[x,y,z,:]=isSimple6_26(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2],topologyAtlasList,totalLabel)

	# for l in xrange(totalLabel):
	# 	nibabel.save(nibabel.Nifti1Image(simpleMap[:,:,:,l].astype(float),IlabelFull.affine),opath+'/simpleMap_atlas_label'+str(l)+'_omega'+str(omega)+'.nii.gz')

	# padding/=int(2**np.abs(omega))
	for xref in xrange(padding,Sref.shape[0]-padding):
		for yref in xrange(padding,Sref.shape[1]-padding):
			for zref in xrange(padding,Sref.shape[2]-padding):
					proportionLabelMap[xref,yref,zref,:]=computeProportionLabel(Ilabel,Sref,xref,yref,zref,omega,totalLabel)

	# for l in xrange(totalLabel):
	# 	nibabel.save(nibabel.Nifti1Image(proportionLabelMap[:,:,:,l].astype(float),IlabelFull.affine),opath+'/proportionLabelMap_atlas_label'+str(l)+'_omega'+str(omega)+'.nii.gz')

	## Benefit Map ##
	pts=np.where(simpleMap==1)
	if metric=='similarity':
		for ipt in xrange(len(pts[0])):
			xref,yref,zref = int(pts[0][ipt]*(2**omega)),int(pts[1][ipt]*(2**omega)),int(pts[2][ipt]*(2**omega))
			currentProportionIlabelPoint = proportionLabelMap[int(pts[0][ipt]*(2**omega)),int(pts[1][ipt]*(2**omega)),int(pts[2][ipt]*(2**omega))]
			newProportionIlabelPoint = currentProportionIlabelPoint.copy()
			newProportionIlabelPoint[ Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]] ] -= 1./resolutionGap #delete new element
			newProportionIlabelPoint[ pts[3][ipt] ] += 1./resolutionGap #add new element
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitNegSimilarity(currentProportionIlabelPoint,newProportionIlabelPoint,Sref[pts[0][ipt],pts[1][ipt],pts[2][ipt],:])
	elif metric=='distanceMap':
		for ipt in xrange(len(pts[0])):
			xref,yref,zref = int(pts[0][ipt]*(2**omega)),int(pts[1][ipt]*(2**omega)),int(pts[2][ipt]*(2**omega))
			currentProportionIlabelPoint = proportionLabelMap[xref,yref,zref]
			newProportionIlabelPoint = currentProportionIlabelPoint.copy()
			newProportionIlabelPoint[ Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]] ] -= 1./resolutionGap #delete new element
			newProportionIlabelPoint[ pts[3][ipt] ] += 1./resolutionGap #add new element
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitNegDistance(currentProportionIlabelPoint,newProportionIlabelPoint,distanceMap[xref,yref,zref,:])

	# for l in xrange(totalLabel):
	# 	nibabel.save(nibabel.Nifti1Image(benefitMap[:,:,:,l].astype(float),IlabelFull.affine),opath+'/benefitMap_atlas_label'+str(l)+'_omega'+str(omega)+'.nii.gz')

	return simpleMap,proportionLabelMap,benefitMap


def updateMapsNeg(Ilabel,Sref,simpleMap,proportionLabelMap,benefitMap,x,y,z,newLabel,topologyList,totalLabel,omega):
	neighbours=getNeighbourCoordonates(x,y,z)
	for nb in neighbours:
		## update simpleMap ##
		simpleMap[nb[0],nb[1],nb[2],:]=isSimple(Ilabel,topologyList,totalLabel,nb[0],nb[1],nb[2])
		## update proportionLabelMap ##
		xref,yref,zref=int(nb[0]*(2**omega)),int(nb[1]*(2**omega)),int(nb[2]*(2**omega))
		proportionLabelMap[xref,yref,zref,:]=computeProportionLabel(Ilabel,Sref,xref,yref,zref,omega,totalLabel)
		## update benefitMap ##
		for l in xrange(totalLabel):
			if simpleMap[nb[0],nb[1],nb[2],l]==1:
				benefitMap[nb[0],nb[1],nb[2],l]=computeBenefitNeg(Ilabel,Sref,proportionLabelMap,x,y,z,l,totalLabel)
			else:
				benefitMap[nb[0],nb[1],nb[2],l]=0

	return simpleMap,proportionLabelMap,benefitMap


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def computeHomogeneousArea(im,totalLabel):
	homogeneous = np.zeros_like(im)
	for l in xrange(totalLabel):
		binIm = 1*(im==l)
		##Binary mask avoiding homogeneous ares, simple points and padding##
		homogeneous += 1*(ndimage.convolve(binIm,np.array([[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,-26,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]))==0)
	homogeneous[homogeneous!=0] = 1
	return homogeneous
