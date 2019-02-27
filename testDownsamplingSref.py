# coding=utf-8

import os,sys
import numpy as np
import math
import nibabel
from numba import jit
from skimage import measure
from scipy import ndimage



@jit
def downsamplingMean4D(Sref,omega,base=2):
	## Check if Sref size is divisible by 2^omega ##
	newSizeSref=np.array(Sref.shape)
	while (newSizeSref[0]%(base**omega) != 0):
		newSizeSref[0]+=1
	while (newSizeSref[1]%(base**omega) != 0):
		newSizeSref[1]+=1
	while (newSizeSref[2]%(base**omega) != 0):
		newSizeSref[2]+=1
	## Create Sref with the new size (divisible by 2^omega) ##
	newSref=np.zeros(newSizeSref)
	sizeSref=Sref.shape
	newSref[:sizeSref[0],:sizeSref[1],:sizeSref[2],:]=Sref
	Sref=newSref.copy()
	## Switch to the small size ##
	newSizeSref=[newSizeSref[0]/(base**omega),newSizeSref[1]/(base**omega),newSizeSref[2]/(base**omega),newSizeSref[3]]
	newSref=np.zeros(newSizeSref)
	for ii in xrange(newSizeSref[0]):
		for jj in xrange(newSizeSref[1]):
			for kk in xrange(newSizeSref[2]):
				for l in xrange(newSizeSref[3]):
					## For each point, compute the mean of the corresponding square in Sref ##
					newSref[ii,jj,kk,l]=np.mean(Sref[ii*(base**omega):(ii+1)*(base**omega),jj*(base**omega):(jj+1)*(base**omega),kk*(base**omega):(kk+1)*(base**omega),l])
	return newSref



@jit
def downsamplingMean3D(image,omega,base=2):
	## Check if image size is divisible by 2^omega ##
	newSizeimage=np.array(image.shape)
	while (newSizeimage[0]%(base**omega) != 0):
		newSizeimage[0]+=1
	while (newSizeimage[1]%(base**omega) != 0):
		newSizeimage[1]+=1
	while (newSizeimage[2]%(base**omega) != 0):
		newSizeimage[2]+=1
	## Create image with the new size (divisible by 2^omega) ##
	newimage=np.zeros(newSizeimage)
	sizeimage=image.shape
	newimage[:sizeimage[0],:sizeimage[1],:sizeimage[2]]=image
	image=newimage.copy()
	## Switch to the small size ##
	newSizeimage=[newSizeimage[0]/(base**omega),newSizeimage[1]/(base**omega),newSizeimage[2]/(base**omega)]
	newimage=np.zeros(newSizeimage)
	for ii in xrange(newSizeimage[0]):
		for jj in xrange(newSizeimage[1]):
			for kk in xrange(newSizeimage[2]):
				## For each point, compute the mean of the corresponding square in image ##
				newimage[ii,jj,kk]=np.mean(image[ii*(base**omega):(ii+1)*(base**omega),jj*(base**omega):(jj+1)*(base**omega),kk*(base**omega):(kk+1)*(base**omega)])
	return newimage



def downsamplingGauss3D(image,omega,stdGaussian,base=2):
	## Compute the guassian filter ##
	gaussimage = ndimage.gaussian_filter(image,sigma=stdGaussian)
	## Check if image size is divisible by 2^omega ##
	newSizeimage=np.array(gaussimage.shape)
	while (newSizeimage[0]%(base) != 0):
		newSizeimage[0]+=1
	while (newSizeimage[1]%(base) != 0):
		newSizeimage[1]+=1
	while (newSizeimage[2]%(base) != 0):
		newSizeimage[2]+=1
	## Create image with the new size (divisible by 2^omega) ##
	newimage=np.zeros(newSizeimage)
	sizeimage=gaussimage.shape
	newimage[:sizeimage[0],:sizeimage[1],:sizeimage[2]]=gaussimage
	gaussimage=newimage.copy()
	## Switch to the small size ##
	newSizeimage=[newSizeimage[0]/(base),newSizeimage[1]/(base),newSizeimage[2]/(base)]
	newimage=np.zeros(newSizeimage)
	for ii in xrange(newSizeimage[0]):
		for jj in xrange(newSizeimage[1]):
			for kk in xrange(newSizeimage[2]):
				## For each point, compute the mean of the corresponding square in image ##
				newimage[ii,jj,kk] = gaussimage[(ii)*base,(jj)*base,(kk)*base]
	return newimage


def downsamplingGauss4D(image,omega,stdGaussian,base=2):
	## Compute the guassian filter ##
	gaussimage = np.zeros([image.shape[0],image.shape[1],image.shape[2],totalLabel])
	for l in xrange(totalLabel):
		gaussimage[:,:,:,l] = ndimage.gaussian_filter(image[:,:,:,l],sigma=stdGaussian)
	## Check if image size is divisible by 2^omega ##
	newSizeimage=np.array(gaussimage.shape)
	while (newSizeimage[0]%(base) != 0):
		newSizeimage[0]+=1
	while (newSizeimage[1]%(base) != 0):
		newSizeimage[1]+=1
	while (newSizeimage[2]%(base) != 0):
		newSizeimage[2]+=1
	## Create image with the new size (divisible by 2^omega) ##
	newimage=np.zeros(newSizeimage)
	sizeimage=gaussimage.shape
	newimage[:sizeimage[0],:sizeimage[1],:sizeimage[2],:sizeimage[3]]=gaussimage
	gaussimage=newimage.copy()
	## Switch to the small size ##
	newSizeimage=[newSizeimage[0]/(base),newSizeimage[1]/(base),newSizeimage[2]/(base),totalLabel]
	newimage=np.zeros(newSizeimage)
	for ii in xrange(newSizeimage[0]):
		for jj in xrange(newSizeimage[1]):
			for kk in xrange(newSizeimage[2]):
				for ll in xrange(totalLabel):
					## For each point, compute the mean of the corresponding square in image ##
					newimage[ii,jj,kk,ll] = gaussimage[(ii)*base,(jj)*base,(kk)*base,ll]
	return newimage



if __name__ == '__main__':
	'''
	Function to test:
		downsamplingMean(Sref,omega)
				Input:
					Sref	=> 	Image to be downsampled
					omega	=>	Number of times to be downsampled
				Return:
				 	newSref	=>	Image downsampled
	'''



	## Test downsamplingMean a segmentation of reference (Sref) ##
	ipath = '/home/carlos/Code/neoSeg'
	inputImage = nibabel.load(ipath+'/examples/Sref.nii.gz')

	Sref = inputImage.get_data()
	omega = 2
	print '\nSize of input Sref = '+str(Sref.shape)+'\n'

	newSref=downsamplingMean(Sref,omega)

	nibabel.save(nibabel.Nifti1Image(newSref,inputImage.affine),ipath+'/Sref_downsampled.nii.gz')
	print 'Size of result = '+str(newSref.shape)+'\n'
