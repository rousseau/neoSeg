# coding=utf-8
import sys
import nibabel
import numpy as np
import skfmm
from skimage import measure
from scipy import ndimage
from .binarisation import computeMajorityVoting, computeMajorityVotingThreshold, computeMajorityVotingFuzzy



''' Thickness map '''

def computeThicknessDistanceMap(Ilabel,label,neighborLabels):
	'''
    Compute the thickness of a label in an image of labels given two neighbor
	labels that surround it using their distance transforms.
    -------------------------------
    Input:  Ilabel 			--> Image of labels
            label   		--> Value of label in 'Ilabel' to compute the thickness
            neighborLabels  --> Values of labels neighbors to 'label'
    Output: thicknessMap    --> Thickness map
    '''
	Ilabel = np.unique(Ilabel)
	thicknessMap = np.zeros(Ilabel.shape)
	thicknessMap1 = ndimage.morphology.distance_transform_edt(1-(1*(Ilabel==neighborLabels[0])))
	thicknessMap2 = ndimage.morphology.distance_transform_edt(1-(1*(Ilabel==neighborLabels[1])))
	thicknessMap = (Ilabel==label)*((thicknessMap1+thicknessMap2)-1)

	return thicknessMap


def computeThicknessLaplaceProfiles(Ilabel,label,neighborLabels,milimeterSize=[1,1,1]):
	'''
    Compute the thickness of a label in an image of labels given two neighbor
	labels that surround it using Laplace profiles method(*).
	(*) Makki, K., Borotikar, B., Garetier, M., Acosta, O., Brochard, S.,
		Ben Salem, D., and Rousseau, F. (2019). 4D in vivo quantification of
		ankle joint space width using dynamic MRI. Presented in EMBC 2019.
    -------------------------------
    Input:  Ilabel 			--> Image of labels
            label   		--> Value of label in 'Ilabel' to compute the thickness
            neighborLabels  --> Values of labels neighbors to 'label'
			milimeterSize	--> Array containing voxel resolution in milimeters
    Output: thicknessMap    --> Thickness map
    '''
	nx, ny, nz = Ilabel.shape
	thicknessMap = np.zeros([nx,ny,nz])

	r = 1*(Ilabel==label) #structure
	S = 1*(Ilabel==neighborLabels[0]) #1st boundary
	S_prime = 1*(Ilabel==neighborLabels[1]) #2nd boundary
	R = np.where(r != 0)
	r[...,0] = 0
	r[...,nz-1] = 0
	#Initial conditions
	dx,dy,dz = milimeterSize[0],milimeterSize[1],milimeterSize[2]

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
	thicknessMap[R[0],R[1],R[2]] = L0[R[0],R[1],R[2]] + L1[R[0],R[1],R[2]]

	return thicknessMap



''' Distance transform map '''

def computeDistanceMap(fuzzyLabels,padding,totalLabel,mode='threshold',threshold=0):
	'''
    Compute the distance transform of a label given a 4D image of probabilistic labels and
	a mode of binarisation.
    -------------------------------
    Input:  fuzzyLabels 	--> 4D image of probabilistic labels
            padding   		--> Margin to be ignored in the image (considered
								as padding)
            totalLabel      --> Number of labels
			mode			--> Method of binarisation ('majorityVoting',
								'threshold' or 'fuzzy')
			threshold		--> Additional parameter for 'threshold' or 'fuzzy'
								modes for ignoring lower or higher value respectively
    Output: distanceMap    --> Distance transform map
    '''
	distanceMap=np.zeros(fuzzyLabels.shape)
	fuzzyLabelsWithoutPad = fuzzyLabels[padding:-padding,padding:-padding,padding:-padding,:]
	if mode=='majorityVoting':
		threshold=''
		binSref=computeMajorityVoting(fuzzyLabelsWithoutPad)
	elif mode=='threshold':
		binSref=computeMajorityVotingThreshold(fuzzyLabelsWithoutPad,threshold)
	elif mode=='fuzzy':
		threshold = np.max( [np.round(np.min(np.max(fuzzyLabelsWithoutPad,axis=-1))*100)/100, threshold] )
		fuzzySref,binSref=computeMajorityVotingFuzzy(fuzzyLabelsWithoutPad,threshold)
		fuzzySref=np.pad(fuzzySref,[(padding,padding),(padding,padding),(padding,padding),(0,0)],mode='constant', constant_values=0)
	binSref=np.pad(binSref,[(padding,padding),(padding,padding),(padding,padding)],mode='constant', constant_values=0)

	for l in range(totalLabel):
		if mode=='fuzzy':
				distanceMap[:,:,:,l]=ndimage.morphology.distance_transform_edt(1-(1*(fuzzySref[:,:,:,l]==1)))
		else:
			distanceMap[:,:,:,l]=ndimage.morphology.distance_transform_edt(1-(1*(binSref==l)))

	return distanceMap


def computeGaussianDistanceMap(fuzzyLabels,stdGaussian,padding,omega,base,totalLabel,mode='threshold',threshold=0):
	'''
    Compute the distance transform of a label given a 4D image of probabilistic labels and
	a mode of binarisation, and resample using a Gaussian filter.
    -------------------------------
    Input:  fuzzyLabels 	--> Image of labels
			stdGaussian		--> Standard deviation of the Gaussian filter
            padding   		--> Padding added to the result
			omega			--> Factor of downsampling, 2^{omega} (omega is
								considered positive)
			base			--> Base parameter of increase the scale in each step
            totalLabel      --> Number of labels
			mode			--> Method of binarisation ('majorityVoting',
								'threshold' or 'fuzzy')
			threshold		--> Additional parameter for 'threshold' or 'fuzzy'
								modes for ignoring lower or higher value respectively
    Output: distanceMap    --> Distance transform map
    '''
	##Binarisation##
	if mode=='majorityVoting':
		threshold=''
		binfuzzyLabels=computeMajorityVoting(fuzzyLabels)
	elif mode=='threshold':
		binfuzzyLabels=computeMajorityVotingThreshold(fuzzyLabels,threshold)
	elif mode=='fuzzy':
		threshold = np.max( [np.round(np.min(np.max(fuzzyLabels,axis=-1))*100)/100, threshold] )
		fuzzyfuzzyLabels,binfuzzyLabels=computeMajorityVotingFuzzy(fuzzyLabels,threshold)
	distanceMap=np.zeros([binfuzzyLabels.shape[0],binfuzzyLabels.shape[1],binfuzzyLabels.shape[2],fuzzyLabels.shape[3]])

	##Distance Map##
	for l in range(totalLabel):
		if mode=='fuzzy':
			distanceMap[:,:,:,l]=ndimage.morphology.distance_transform_edt(1-(1*(fuzzyfuzzyLabels[:,:,:,l]==1)))
		else:
			distanceMap[:,:,:,l]=ndimage.morphology.distance_transform_edt(1-(1*(binfuzzyLabels==l)))

	if omega>0:
		##Downsampling##
		for o in range(omega):
			## Filtering for next downsampling ##
			gaussDistanceMap = np.zeros([distanceMap.shape[0],distanceMap.shape[1],distanceMap.shape[2],totalLabel])
			for l in range(totalLabel):
				gaussDistanceMap[:,:,:,l] = ndimage.gaussian_filter(distanceMap[:,:,:,l],sigma=stdGaussian)

			newSizeimage=np.array(gaussDistanceMap.shape)
			while (newSizeimage[0]%(base) != 0):
				newSizeimage[0]+=1
			while (newSizeimage[1]%(base) != 0):
				newSizeimage[1]+=1
			while (newSizeimage[2]%(base) != 0):
				newSizeimage[2]+=1
			distanceMap=np.zeros(newSizeimage)
			sizeimage=gaussDistanceMap.shape
			distanceMap[:sizeimage[0],:sizeimage[1],:sizeimage[2],:sizeimage[3]]=gaussDistanceMap
			gaussDistanceMap=distanceMap.copy()
			newSizeimage=[newSizeimage[0]/(base),newSizeimage[1]/(base),newSizeimage[2]/(base),totalLabel]
			distanceMap=np.zeros(newSizeimage)

			for ii in range(newSizeimage[0]):
				for jj in range(newSizeimage[1]):
					for kk in range(newSizeimage[2]):
						for ll in range(totalLabel):
							## For each point, compute the mean of the corresponding square in image ##
							distanceMap[ii,jj,kk,ll] = gaussDistanceMap[(ii)*base,(jj)*base,(kk)*base,ll]
	else:
		## Gaussian Filter ##
		gaussDistanceMap = np.zeros([distanceMap.shape[0],distanceMap.shape[1],distanceMap.shape[2],totalLabel])
		for l in range(totalLabel):
			gaussDistanceMap[:,:,:,l] = ndimage.gaussian_filter(distanceMap[:,:,:,l],sigma=stdGaussian)
		distanceMap = gaussDistanceMap.copy()

	## Adding pad to the result ##
	distanceMap = np.pad(distanceMap,[(padding,padding),(padding,padding),(padding,padding),(0,0)],mode='constant', constant_values=0)

	return distanceMap


def computeFastMarchingGaussianDistanceMap(fuzzyLabels,stdGaussian,padding,omega,base,totalLabel,mode='threshold',threshold=0,erosion=0):
	'''
    Compute the distance transform of a label given a 4D image of probabilistic labels and
	a mode of binarisation, and resample using a Gaussian filter. The distance transform
	is computed by the scikit-fmm library which avoid the diagonal neighbor problem.
    -------------------------------
    Input:  fuzzyLabels 	--> Image of labels
			stdGaussian		--> Standard deviation of the Gaussian filter
            padding   		--> Padding added to the result
			omega			--> Factor of downsampling, {base}^{omega} (omega is
								considered positive)
			base			--> Base parameter of increase the scale in each step
            totalLabel      --> Number of labels
			mode			--> Method of binarisation ('majorityVoting',
								'threshold' or 'fuzzy')
			threshold		--> Additional parameter for 'threshold' or 'fuzzy'
								modes for ignoring lower or higher value respectively
    Output: distanceMap    --> Distance transform map
    '''
	##Binarisation##
	if mode=='majorityVoting':
		threshold=''
		binSref=computeMajorityVoting(fuzzyLabels)
	elif mode=='threshold':
		binfuzzyLabels=computeMajorityVotingThreshold(fuzzyLabels,threshold)
	elif mode=='fuzzy':
		threshold = np.max( [np.round(np.min(np.max(fuzzyLabels,axis=-1))*100)/100, threshold] )
		fuzzyfuzzyLabels,binfuzzyLabels=computeMajorityVotingFuzzy(fuzzyLabels,threshold)
	distanceMap=np.zeros([binfuzzyLabels.shape[0],binfuzzyLabels.shape[1],binfuzzyLabels.shape[2],fuzzyLabels.shape[3]])

	if erosion!=0:
		tmp = np.zeros_like(binfuzzyLabels)
		for l in range(totalLabel):
			tmp += l * ndimage.morphology.binary_erosion(1*(binfuzzyLabels==l),iterations=erosion)
		binfuzzyLabels = tmp.copy()

	##Distance Map##
	for l in range(totalLabel):
		if mode=='fuzzy':
			distanceMap[:,:,:,l] = skfmm.distance(1-(1*(fuzzyfuzzyLabels[:,:,:,l]==1)))
		else:
			distanceMap[:,:,:,l] = skfmm.distance(1-(1*(binfuzzyLabels==l)))

	if omega>0:
		##Downsampling##
		for o in range(omega):
			## Filtering for next downsampling ##
			gaussDistanceMap = np.zeros((distanceMap.shape[0],distanceMap.shape[1],distanceMap.shape[2],totalLabel))
			for l in range(totalLabel):
				gaussDistanceMap[:,:,:,l] = ndimage.gaussian_filter(distanceMap[:,:,:,l],sigma=stdGaussian)

			newSizeimage=np.array(gaussDistanceMap.shape)
			while (newSizeimage[0]%(base) != 0):
				newSizeimage[0]+=1
			while (newSizeimage[1]%(base) != 0):
				newSizeimage[1]+=1
			while (newSizeimage[2]%(base) != 0):
				newSizeimage[2]+=1
			distanceMap=np.zeros(newSizeimage)
			sizeimage=gaussDistanceMap.shape
			distanceMap[:sizeimage[0],:sizeimage[1],:sizeimage[2],:sizeimage[3]]=gaussDistanceMap
			gaussDistanceMap=distanceMap.copy()
			newSizeimage=[int(newSizeimage[0]/(base)),int(newSizeimage[1]/(base)),int(newSizeimage[2]/(base)),totalLabel]
			distanceMap=np.zeros(newSizeimage)

			for ii in range(newSizeimage[0]):
				for jj in range(newSizeimage[1]):
					for kk in range(newSizeimage[2]):
						for ll in range(totalLabel):
							## For each point, compute the mean of the corresponding square in image ##
							distanceMap[ii,jj,kk,ll] = gaussDistanceMap[(ii)*base,(jj)*base,(kk)*base,ll]

	## Adding pad to the result ##
	distanceMap = np.pad(distanceMap,[(padding,padding),(padding,padding),(padding,padding),(0,0)],mode='constant', constant_values=0)

	return distanceMap



''' Similarities '''

	## For positive omegas (dimension lower than the reference) ##

		## Simple Points ##

def computeBenefitPosDistance(currentILabelPoint,distanceMapPoint,newLabel):
	'''
    Compute the benefit of a point for a new label with respect to the distance
	transform in a particular point (the simplicity of the point has to be
	previously checked).
    -------------------------------
    Input:  currentILabelPoint 	--> Current label of the point
			distanceMapPoint	--> Distance transform maps in this point
            newLabel   			--> Label to be evaluated
    Output: benefit    --> Benefit of the point for the evaluated label
    '''
	if distanceMapPoint[currentILabelPoint]-distanceMapPoint[newLabel] > 0:
		benefit = distanceMapPoint[currentILabelPoint]
	else:
		benefit = 0
	return  benefit


def computeBenefitPosSimilarity(currentILabelPoint,currentfuzzyLabelsPoint,newLabel,totalLabel):
	'''
    Compute the benefit (positive) or loss (negative) of a point for a new label
	with respect to the similarity with the probabilistic reference labels (the
	simplicity of the point has to be previously checked).
    -------------------------------
    Input:  currentILabelPoint 		--> Current label of the point
			currentfuzzyLabelsPoint	--> Label of the point in the reference label
            newLabel   				--> Label to be evaluated
			totalLabel				--> Number of labels
    Output: benefit    --> Benefit of the point for the evaluated label
    '''
	currentVal = np.zeros([totalLabel])
	currentVal[currentILabelPoint] = 1
	nextVal = np.zeros([totalLabel])
	nextVal[newLabel] = 1
	benefit = np.linalg.norm(currentVal-currentfuzzyLabelsPoint) - np.linalg.norm(nextVal-currentfuzzyLabelsPoint)
	return  benefit


def computeBenefitPosGradientDistance(currentILabelPoint,distanceMapPoint,newLabel):
	'''
    Compute the benefit (positive) or loss (negative) of a point for a new label
	with respect to the distance transform in a particular point (the simplicity
	of the point has to be previously checked).
    -------------------------------
    Input:  currentILabelPoint 	--> Current label of the point
			distanceMapPoint	--> Distance transform maps in this point
            newLabel   			--> Label to be evaluated
    Output: benefit    --> Benefit of the point for the evaluated label
    '''
	benefit = distanceMapPoint[currentILabelPoint] - distanceMapPoint[newLabel]
	return benefit


def computeBenefitPairPosDistance(currentILabelPair,distanceMapPair,newLabel):
	'''
    Compute the benefit (positive) or loss (negative) of a pair for a new label
	with respect to the distance transform in a particular pair (the simplicity
	of the pair has to be previously checked).
    -------------------------------
    Input:  currentILabelPair 	--> Current label of the pair
			distanceMapPair		--> Distance transform maps in this pair
            newLabel   			--> Label to be evaluated
    Output: benefit    --> Benefit of the pair for the evaluated label
    '''
	benefit = np.sum(distanceMapPair[:,currentILabelPair] - distanceMapPair[:,newLabel])
	return benefit


		## Simple Pairs ##

def computeBenefitPairPosSimilarity(currentILabelPair,currentfuzzyLabelsPair,newLabel,totalLabel):
	'''
    Compute the benefit (positive) or loss (negative) of a pair for a new label
	with respect to the similarity with the probabilistic reference labels (the
	simplicity of the pair has to be previously checked).
    -------------------------------
    Input:  currentILabelPair 		--> Current label of the pair
			currentfuzzyLabelsPair	--> Label of the pair in the reference label
            newLabel   				--> Label to be evaluated
			totalLabel				--> Number of labels
    Output: benefit    --> Benefit of the pair for the evaluated label
    '''
	currentVal = np.zeros([2,totalLabel])
	currentVal[:,currentILabelPair] = 1
	nextVal = np.zeros([2,totalLabel])
	nextVal[:,newLabel] = 1
	benefit = np.linalg.norm(currentVal-currentfuzzyLabelsPair) - np.linalg.norm(nextVal-currentfuzzyLabelsPair)
	return  benefit



	## For negative omegas (dimension higer than the reference) ##

def computeBenefitNegDistance(currentProportionIlabelPoint,newProportionIlabelPoint,distanceMapPoint):
	'''
    Compute the benefit of a point for a new label with respect to the distance
	transform in a particular point (the simplicity of the point has to be
	previously checked).
    -------------------------------
    Input:  currentILabelPoint 	--> Current label of the point
			distanceMapPoint	--> Distance transform maps in this point
            newLabel   			--> Label to be evaluated
    Output: benefit    --> Benefit of the point for the evaluated label
    '''
	if np.linalg.norm(currentProportionIlabelPoint*distanceMapPoint) - np.linalg.norm(newProportionIlabelPoint*distanceMapPoint) > 0:
		benefit = np.linalg.norm(currentProportionIlabelPoint*distanceMapPoint)
	else:
		benefit = 0
	return  benefit


def computeBenefitNegSimilarity(currentProportionIlabelPoint,newProportionIlabelPoint,currentSrefPoint):
	'''
    Compute the benefit (positive) or loss (negative) of a point for a new label
	with respect to the similarity with the probabilistic reference labels (the
	simplicity of the point has to be previously checked).
    -------------------------------
    Input:  currentILabelPoint 		--> Current label of the point
			currentfuzzyLabelsPoint	--> Label of the point in the reference label
            newLabel   				--> Label to be evaluated
			totalLabel				--> Number of labels
    Output: benefit    --> Benefit of the point for the evaluated label
    '''
	benefit = np.linalg.norm(currentProportionIlabelPoint-currentSrefPoint)-np.linalg.norm(newProportionIlabelPoint-currentSrefPoint)
	return benefit
