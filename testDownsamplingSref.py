# coding=utf-8

import os,sys
import numpy as np
import math
import nibabel
from numba import jit
from skimage import measure

@jit
def downsampling(Sref,omega):
	## Check if Sref size is divisible by 2^omega ##
	newSizeSref=Sref.shape
	while (newSizeSref[0]%(2**omega) != 0):
		newSizeSref[0]+=1
	while (newSizeSref[1]%(2**omega) != 0):
		newSizeSref[1]+=1
	while (newSizeSref[2]%(2**omega) != 0):
		newSizeSref[2]+=1
	## Create Sref with the new size (divisible by 2^omega) ##
	newSref=np.zeros(newSizeSref)
	sizeSref=Sref.shape
	newSref[:sizeSref[0],:sizeSref[1],:sizeSref[2],:]=Sref
	Sref=newSref.copy()
	## Switch to the small size ##
	newSizeSref=[newSizeSref[0]/(2**omega),newSizeSref[1]/(2**omega),newSizeSref[2]/(2**omega),newSizeSref[3]]
	newSref=np.zeros(newSizeSref)
	for ii in xrange(newSizeSref[0]):
		for jj in xrange(newSizeSref[1]):
			for kk in xrange(newSizeSref[2]):
				for l in xrange(newSizeSref[3]):
					## For each point, compute the mean of the corresponding square in Sref ##
					newSref[ii,jj,kk,l]=np.mean(Sref[ii*(2**omega):(ii+1)*(2**omega),jj*(2**omega):(jj+1)*(2**omega),kk*(2**omega):(kk+1)*(2**omega),l])
	return newSref




if __name__ == '__main__':
	'''
	Function to test:
		downsampling(Sref,omega)
				Input:
					Sref	=> 	Image to be downsampled
					omega	=>	Number of times to be downsampled
				Return:
				 	newSref	=>	Image downsampled
	'''



	## Test downsampling a segmentation of reference (Sref) ##
	ipath = '/home/carlos/Code/neoSeg'
	inputImage = nibabel.load(ipath+'/examples/Sref.nii.gz')

	Sref = inputImage.get_data()
	omega = 2
	print '\nSize of input Sref = '+str(Sref.shape)+'\n'

	newSref=downsampling(Sref,omega)

	nibabel.save(nibabel.Nifti1Image(newSref,inputImage.affine),ipath+'/Sref_downsampled.nii.gz')
	print 'Size of result = '+str(newSref.shape)+'\n'
