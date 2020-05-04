# coding=utf-8
import numpy as np
from scipy import ndimage



def computeMajorityVoting(Sref):
	'''
	Compute the binarisation of a stack of label maps (Sref) using
	majority voting.
	-------------------------------
    Input:  Sref	--> Segmentation of reference (one map for each label)
    Output: binSref --> Segmentation of reference (binary image)
	'''
	binSref = np.argmax(Sref,axis=3)
	return binSref

def computeMajorityVotingThreshold(Sref,threshold=0.5):
	'''
	Compute the binarisation of a stack of label maps (Sref) using
	majority voting and ignoring probabilities less that threshold. These
	criteria lead to leave some points unclassified.
	-------------------------------
    Input:  Sref		--> Segmentation of reference (one map for each label)
			threshold	--> Threshold for ignoring probabilities
    Output: binSref --> Segmentation of reference (binary image)
	'''
	binSref = np.argmax(Sref,axis=3)
	binSref[np.max(Sref,axis=3) < threshold] = -1
	return binSref

def computeMajorityVotingFuzzy(Sref,threshold=0.25):
	'''
	Compute the binarisation of a stack of label maps (Sref) using
	majority voting and adding probabilities less that threshold. These
	criteria lead to have some points belonging to several labels.
	-------------------------------
    Input:  Sref		--> Segmentation of reference (one map for each label)
			threshold	--> Threshold for ignoring probabilities
    Output: fuzzySrefMV	-->	Segmentation of reference (binary with several label
							in some points)
			binSrefMV 	--> Segmentation of reference (binary image creating
							a new label for multi-label points)
	'''
	fuzzySrefMV = np.zeros(Sref.shape)
	binSrefMV = np.zeros(Sref[:,:,:,0].shape)
	for l in xrange(Sref.shape[-1]):
		fuzzySrefMV[:,:,:,l] = 1*(Sref[:,:,:,l]>=threshold)
		binSrefMV += l*(fuzzySrefMV[:,:,:,l])
	binSrefMV[np.sum(fuzzySrefMV,axis=-1)>1] = Sref.shape[-1]
	binSrefMV[np.sum(fuzzySrefMV,axis=-1)==0] = -1
	return fuzzySrefMV,binSrefMV
