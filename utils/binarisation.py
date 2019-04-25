# coding=utf-8

import numpy as np
from scipy import ndimage



def computeMajorityVoting(Sref):
	return np.argmax(Sref,axis=3)

def computeMajorityVotingThreshold(Sref,threshold=0.5):
	binSref = np.argmax(Sref,axis=3)
	binSref[np.max(Sref,axis=3) < threshold] = -1
	return binSref

def computeMajorityVotingFuzzy(Sref,threshold=0.25):
	fuzzySrefMV = np.zeros(Sref.shape)
	binSrefMV = np.zeros(Sref[:,:,:,0].shape)
	for l in xrange(Sref.shape[-1]):
		fuzzySrefMV[:,:,:,l] = 1*(Sref[:,:,:,l]>=threshold)
		binSrefMV += l*(fuzzySrefMV[:,:,:,l])
	binSrefMV[np.sum(fuzzySrefMV,axis=-1)>1] = Sref.shape[-1]
	binSrefMV[np.sum(fuzzySrefMV,axis=-1)==0] = -1
	return fuzzySrefMV,binSrefMV
