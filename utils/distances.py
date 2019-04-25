# coding=utf-8

import numpy as np
import skfmm
from scipy import ndimage
from utils.binarisation import computeMajorityVoting, computeMajorityVotingThreshold, computeMajorityVotingFuzzy


def computeThicknessDistance(Ilabel,label,totalLabel):
	thicknessLabelMap=np.zeros(Ilabel.shape)
	thicknessMap1=ndimage.morphology.distance_transform_edt(1-(1*(Ilabel==(label+1)%totalLabel)))
	thicknessMap2=ndimage.morphology.distance_transform_edt(1-(1*(Ilabel==(label-1)%totalLabel)))
	thicknessLabelMap=(Ilabel==label)*((thicknessMap1+thicknessMap2)-1)

	return thicknessLabelMap


def computeDistanceMap(newSref,padding,totalLabel,mode='threshold',threshold=0):
	distanceMap=np.zeros(newSref.shape)
	Sref = newSref[padding:-padding,padding:-padding,padding:-padding,:]
	if mode=='standard':
		threshold=''
		binSref=computeMajorityVoting(Sref)
	elif mode=='threshold':
		#threshold=0.6
		binSref=computeMajorityVotingThreshold(Sref,threshold)
	elif mode=='fuzzy':
		#threshold=0.30
		threshold = np.max( [np.round(np.min(np.max(Sref,axis=-1))*100)/100, threshold] )
		fuzzySref,binSref=computeMajorityVotingFuzzy(Sref,threshold)
		fuzzySref=np.pad(fuzzySref,[(padding,padding),(padding,padding),(padding,padding),(0,0)],mode='constant', constant_values=0)
	binSref=np.pad(binSref,[(padding,padding),(padding,padding),(padding,padding)],mode='constant', constant_values=0)

	for l in xrange(totalLabel):
		if mode=='fuzzy':
				distanceMap[:,:,:,l]=ndimage.morphology.distance_transform_edt(1-(1*(fuzzySref[:,:,:,l]==1)))
		else:
			distanceMap[:,:,:,l]=ndimage.morphology.distance_transform_edt(1-(1*(binSref==l)))

	return distanceMap


def computeGaussianDistanceMap(Sref,stdGaussian,padding,omega,base,totalLabel,mode='threshold',threshold=0):
	##Binarisation##
	if mode=='standard':
		threshold=''
		binSref=computeMajorityVoting(Sref)
	elif mode=='threshold':
		binSref=computeMajorityVotingThreshold(Sref,threshold)
	elif mode=='fuzzy':
		threshold = np.max( [np.round(np.min(np.max(Sref,axis=-1))*100)/100, threshold] )
		fuzzySref,binSref=computeMajorityVotingFuzzy(Sref,threshold)
	distanceMap=np.zeros([binSref.shape[0],binSref.shape[1],binSref.shape[2],Sref.shape[3]])

	##Distance Map##
	for l in xrange(totalLabel):
		if mode=='fuzzy':
			distanceMap[:,:,:,l]=ndimage.morphology.distance_transform_edt(1-(1*(fuzzySref[:,:,:,l]==1)))
		else:
			distanceMap[:,:,:,l]=ndimage.morphology.distance_transform_edt(1-(1*(binSref==l)))


	if omega>0:
		##Downsampling##
		for o in xrange(omega):
			## Filtering for next downsampling ##
			gaussDistanceMap = np.zeros([distanceMap.shape[0],distanceMap.shape[1],distanceMap.shape[2],totalLabel])
			for l in xrange(totalLabel):
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

			for ii in xrange(newSizeimage[0]):
				for jj in xrange(newSizeimage[1]):
					for kk in xrange(newSizeimage[2]):
						for ll in xrange(totalLabel):
							## For each point, compute the mean of the corresponding square in image ##
							distanceMap[ii,jj,kk,ll] = gaussDistanceMap[(ii)*base,(jj)*base,(kk)*base,ll]
	else:
		## Gaussian Filter ##
		gaussDistanceMap = np.zeros([distanceMap.shape[0],distanceMap.shape[1],distanceMap.shape[2],totalLabel])
		for l in xrange(totalLabel):
			gaussDistanceMap[:,:,:,l] = ndimage.gaussian_filter(distanceMap[:,:,:,l],sigma=stdGaussian)
		distanceMap = gaussDistanceMap.copy()

	## Adding pad to the result ##
	distanceMap = np.pad(distanceMap,[(padding,padding),(padding,padding),(padding,padding),(0,0)],mode='constant', constant_values=0)

	return distanceMap


def computeFastMarchingGaussianDistanceMap(Sref,stdGaussian,padding,omega,base,totalLabel,mode='threshold',threshold=0):
	##Binarisation##
	if mode=='standard':
		threshold=''
		binSref=computeMajorityVoting(Sref)
	elif mode=='threshold':
		binSref=computeMajorityVotingThreshold(Sref,threshold)
	elif mode=='fuzzy':
		threshold = np.max( [np.round(np.min(np.max(Sref,axis=-1))*100)/100, threshold] )
		fuzzySref,binSref=computeMajorityVotingFuzzy(Sref,threshold)
	distanceMap=np.zeros([binSref.shape[0],binSref.shape[1],binSref.shape[2],Sref.shape[3]])

	##Distance Map##
	for l in xrange(totalLabel):
		if mode=='fuzzy':
			distanceMap[:,:,:,l] = skfmm.distance(1-(1*(fuzzySref[:,:,:,l]==1)))
		else:
			distanceMap[:,:,:,l] = skfmm.distance(1-(1*(binSref==l)))

	if omega>0:
		##Downsampling##
		for o in xrange(omega):
			## Filtering for next downsampling ##
			gaussDistanceMap = np.zeros([distanceMap.shape[0],distanceMap.shape[1],distanceMap.shape[2],totalLabel])
			for l in xrange(totalLabel):
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

			for ii in xrange(newSizeimage[0]):
				for jj in xrange(newSizeimage[1]):
					for kk in xrange(newSizeimage[2]):
						for ll in xrange(totalLabel):
							## For each point, compute the mean of the corresponding square in image ##
							distanceMap[ii,jj,kk,ll] = gaussDistanceMap[(ii)*base,(jj)*base,(kk)*base,ll]
	else:
		## Gaussian Filter ##
		gaussDistanceMap = np.zeros([distanceMap.shape[0],distanceMap.shape[1],distanceMap.shape[2],totalLabel])
		for l in xrange(totalLabel):
			gaussDistanceMap[:,:,:,l] = ndimage.gaussian_filter(distanceMap[:,:,:,l],sigma=stdGaussian)
		distanceMap = gaussDistanceMap.copy()

	## Adding pad to the result ##
	distanceMap = np.pad(distanceMap,[(padding,padding),(padding,padding),(padding,padding),(0,0)],mode='constant', constant_values=0)

	return distanceMap



def computeDilationFastMarchingGaussianDistanceMap(Sref,stdGaussian,dilationIter,padding,omega,base,totalLabel,mode='threshold',threshold=0):
	##Binarisation##
	if mode=='standard':
		threshold=''
		binSref=computeMajorityVoting(Sref)
	elif mode=='threshold':
		binSref=computeMajorityVotingThreshold(Sref,threshold)
	elif mode=='fuzzy':
		threshold = np.max( [np.round(np.min(np.max(Sref,axis=-1))*100)/100, threshold] )
		fuzzySref,binSref=computeMajorityVotingFuzzy(Sref,threshold)
		#fuzzySref=np.pad(fuzzySref,[(padding,padding),(padding,padding),(padding,padding),(0,0)],mode='constant', constant_values=0)
	#binSref=np.pad(binSref,[(padding,padding),(padding,padding),(padding,padding)],mode='constant', constant_values=0)
	distanceMap=np.zeros([binSref.shape[0],binSref.shape[1],binSref.shape[2],Sref.shape[3]])

	##Distance Map##
	for l in xrange(totalLabel):
		if mode=='fuzzy':
			distanceMap[:,:,:,l] = skfmm.distance(1-(ndimage.morphology.binary_dilation(1*(fuzzySref[:,:,:,l]==1),iterations=dilationIter) ))
		else:
			distanceMap[:,:,:,l] = skfmm.distance(1-(ndimage.morphology.binary_dilation(1*(binSref==l),iterations=dilationIter) ))

	if omega>0:
		##Downsampling##
		for o in xrange(omega):
			## Filtering for next downsampling ##
			gaussDistanceMap = np.zeros([distanceMap.shape[0],distanceMap.shape[1],distanceMap.shape[2],totalLabel])
			for l in xrange(totalLabel):
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

			for ii in xrange(newSizeimage[0]):
				for jj in xrange(newSizeimage[1]):
					for kk in xrange(newSizeimage[2]):
						for ll in xrange(totalLabel):
							## For each point, compute the mean of the corresponding square in image ##
							distanceMap[ii,jj,kk,ll] = gaussDistanceMap[(ii)*base,(jj)*base,(kk)*base,ll]
	else:
		## Gaussian Filter ##
		gaussDistanceMap = np.zeros([distanceMap.shape[0],distanceMap.shape[1],distanceMap.shape[2],totalLabel])
		for l in xrange(totalLabel):
			gaussDistanceMap[:,:,:,l] = ndimage.gaussian_filter(distanceMap[:,:,:,l],sigma=stdGaussian)
		distanceMap = gaussDistanceMap.copy()

	## Adding pad to the result ##
	distanceMap = np.pad(distanceMap,[(padding,padding),(padding,padding),(padding,padding),(0,0)],mode='constant', constant_values=0)

	return distanceMap


def computeGrowthFMMGaussianDistanceMap(Sref,stdGaussian,growth,padding,omega,base,totalLabel,struct=1,mode='threshold',threshold=0):
	##Binarisation##
	if mode=='standard':
		threshold=''
		binSref=computeMajorityVoting(Sref)
	elif mode=='threshold':
		binSref=computeMajorityVotingThreshold(Sref,threshold)
	elif mode=='fuzzy':
		threshold = np.max( [np.round(np.min(np.max(Sref,axis=-1))*100)/100, threshold] )
		fuzzySref,binSref=computeMajorityVotingFuzzy(Sref,threshold)
		#fuzzySref=np.pad(fuzzySref,[(padding,padding),(padding,padding),(padding,padding),(0,0)],mode='constant', constant_values=0)
	#binSref=np.pad(binSref,[(padding,padding),(padding,padding),(padding,padding)],mode='constant', constant_values=0)
	distanceMap=np.zeros([binSref.shape[0],binSref.shape[1],binSref.shape[2],Sref.shape[3]])

	##Distance Map##
	for l in xrange(totalLabel):
		#print l
		count=0
		if growth[l]>0:
			initialNumPoints = np.count_nonzero(1*(binSref==l))
			numPoints = np.count_nonzero(1*(binSref==l))
			label = 1*(binSref==l)
			#
			label = ndimage.morphology.binary_dilation(label,iterations=struct)
			#while (np.count_nonzero(label)<1.5*initialNumPoints) & (np.count_nonzero(label)<0.9*Sref[:,:,:,l].size):
			#	#print str(count)+' '+str(np.count_nonzero(label))+' '+str(0.9*Sref[:,:,:,l].size)
			#	label = ndimage.morphology.binary_dilation(label,iterations=1)
			#	count += 1
			distanceMap[:,:,:,l] = skfmm.distance( 1-label )
		elif growth[l]<0:
			initialNumPoints = np.count_nonzero(1*(binSref==l))
			numPoints = np.count_nonzero(1*(binSref==l))
			label = 1*(binSref==l)
			#
			label = ndimage.morphology.binary_erosion(label,iterations=struct)
			#while (np.count_nonzero(label)>0.5*initialNumPoints):
			#	label = ndimage.morphology.binary_erosion(label,iterations=1)
			#	count += 1
			distanceMap[:,:,:,l] = skfmm.distance( 1-label )
		else:
			distanceMap[:,:,:,l] = skfmm.distance( 1-(1*(binSref==l)) )

	if omega>0:
		##Downsampling##
		for o in xrange(omega):
			## Filtering for next downsampling ##
			gaussDistanceMap = np.zeros([distanceMap.shape[0],distanceMap.shape[1],distanceMap.shape[2],totalLabel])
			for l in xrange(totalLabel):
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

			for ii in xrange(newSizeimage[0]):
				for jj in xrange(newSizeimage[1]):
					for kk in xrange(newSizeimage[2]):
						for ll in xrange(totalLabel):
							## For each point, compute the mean of the corresponding square in image ##
							distanceMap[ii,jj,kk,ll] = gaussDistanceMap[(ii)*base,(jj)*base,(kk)*base,ll]
	else:
		## Gaussian Filter ##
		gaussDistanceMap = np.zeros([distanceMap.shape[0],distanceMap.shape[1],distanceMap.shape[2],totalLabel])
		for l in xrange(totalLabel):
			gaussDistanceMap[:,:,:,l] = ndimage.gaussian_filter(distanceMap[:,:,:,l],sigma=stdGaussian)
		distanceMap = gaussDistanceMap.copy()

	## Adding pad to the result ##
	distanceMap = np.pad(distanceMap,[(padding,padding),(padding,padding),(padding,padding),(0,0)],mode='constant', constant_values=0)

	return distanceMap


## Similarities ##
def computeBenefitPosDistance(currentILabelPoint,distanceMapPoint,newLabel):
	if distanceMapPoint[currentILabelPoint]-distanceMapPoint[newLabel] > 0:
		benefit = distanceMapPoint[currentILabelPoint]
	else:
		benefit = 0

	return  benefit


def computeBenefitPairPosDistance(currentILabelPair,distanceMapPair,newLabel):
	if np.sum(distanceMapPair[:,currentILabelPair]-distanceMapPair[:,newLabel]) > 0:
		benefit = distanceMapPair[:,currentILabelPair]
	else:
		benefit = 0

	return  benefit


'''NEGATIVE'''
def computeBenefitNegDistance(currentProportionIlabelPoint,newProportionIlabelPoint,distanceMapPoint):
	if np.linalg.norm(currentProportionIlabelPoint*distanceMapPoint) - np.linalg.norm(newProportionIlabelPoint*distanceMapPoint) > 0:
		benefit = np.linalg.norm(currentProportionIlabelPoint*distanceMapPoint)
	else:
		benefit = 0

	return  benefit


def computeBenefitPosGradientDistance(currentILabelPoint,distanceMapPoint,newLabel):
	return distanceMapPoint[currentILabelPoint]-distanceMapPoint[newLabel]



def computeBenefitPosSimilarity(currentILabelPoint,currentSrefPoint,newLabel,totalLabel):
	currentVal=np.zeros([totalLabel])
	currentVal[currentILabelPoint]=1
	nextVal=np.zeros([totalLabel])
	nextVal[newLabel]=1
	benefit=np.linalg.norm(currentVal-currentSrefPoint)-np.linalg.norm(nextVal-currentSrefPoint)

	return  benefit


'''NEGATIVE'''
def computeBenefitNegSimilarity(currentProportionIlabelPoint,newProportionIlabelPoint,currentSrefPoint):
	return np.linalg.norm(currentProportionIlabelPoint-currentSrefPoint)-np.linalg.norm(newProportionIlabelPoint-currentSrefPoint)


def computeBenefitPairPosSimilarity(currentILabelPair,currentSrefPair,newLabel,totalLabel):
	currentVal=np.zeros([totalLabel])
	currentVal[currentILabelPair]=1
	nextVal=np.zeros([totalLabel])
	nextVal[newLabel]=1
	benefit=np.linalg.norm(currentVal-currentSrefPair)-np.linalg.norm(nextVal-currentSrefPair)

	return  benefit


## OLD ##
'''
def computeDistanceMapBenefit(Ilabel,distanceMap,newLabel,x,y,z):
	neighbours=get6NeighbourCoordinates(x,y,z)
	currentLabel=Ilabel[x,y,z]
	nbCurrent=[nb for nb in neighbours if Ilabel[nb[0],nb[1],nb[2]]==currentLabel]
	nbCurrentMin=np.min([distanceMap[nb[0],nb[1],nb[2],currentLabel] for nb in nbCurrent])
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
'''
