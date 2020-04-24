# coding=utf-8
import os,sys
import numpy as np
import math
import nibabel
from numba import jit
from skimage import measure
from scipy import ndimage



def changeVoxelSizeInAffineMatrix(affineMatrix,factor):
	'''
	Change the voxel size in an affine matrix 4 x 4
    -------------------------------
    Input:  affineMatrix 	--> affine matrix to be changed
			factor			--> multiply factor for new resolution
    Output: newAffineMatrix --> new changed affine matrix
	'''
	newAffineMatrix = affineMatrix.copy()
	newAffineMatrix.flat[::5] = np.r_[np.diag(affineMatrix)[:-1]*factor,affineMatrix[-1,-1]]
	return newAffineMatrix


def upsampling3D(im,base=2):
	'''
	Upsample a 3D image by a specific factor
    -------------------------------
    Input:  im 		--> 3D image
			base	--> factor of the upsampling
    Output: newIm --> 3D upsampled image
	'''
	sizeIm=im.shape
	newIm=np.zeros([im.shape[0]*base,im.shape[1]*base,im.shape[2]*base])
	for ii in range(sizeIm[0]):
		for jj in range(sizeIm[1]):
			for kk in range(sizeIm[2]):
				newIm[ii*base:(ii+1)*base,jj*base:(jj+1)*base,kk*base:(kk+1)*base]=im[ii,jj,kk]

	return newIm


def upsampling4D(im,base=2):
	'''
	Upsample a 4D image by a specific factor
    -------------------------------
    Input:  im 		--> 4D image
			base	--> factor of the upsampling
    Output: newIm --> 4D upsampled image
	'''
	sizeIm=im.shape
	newIm=np.zeros([im.shape[0]*base,im.shape[1]*base,im.shape[2]*base,im.shape[3]*base])
	for ii in range(sizeIm[0]):
		for jj in range(sizeIm[1]):
			for kk in range(sizeIm[2]):
				for ll in range(sizeIm[3]):
					newIm[ii*base:(ii+1)*base,jj*base:(jj+1)*base,kk*base:(kk+1)*base,ll]=im[ii,jj,kk,ll]

	return newIm


def downsamplingMean4D(image,omega,base=2):
	'''
	Downsample a 4D image by a specific factor using the average intensity value
    -------------------------------
    Input:  image	--> 4D image
			omega	-->	Factor of downsampling, {base}^{omega} (omega is
								considered positive)
			base	--> Base parameter of increase the scale in each step
    Output: newIm --> 4D downsampled image
	'''
	## Check if image size is divisible by 2^omega ##
	newSizeSref = np.array(image.shape).astype(int)
	while (newSizeSref[0]%(base**omega) != 0):
		newSizeSref[0] += 1
	while (newSizeSref[1]%(base**omega) != 0):
		newSizeSref[1] += 1
	while (newSizeSref[2]%(base**omega) != 0):
		newSizeSref[2] += 1
	## Create image with the new size (divisible by 2^omega) ##
	newSref = np.zeros(newSizeSref)
	sizeSref = image.shape
	newSref[:sizeSref[0],:sizeSref[1],:sizeSref[2],:] = image
	image = newSref.copy()
	## Switch to the small size ##
	newSizeSref = [int(newSizeSref[0]/(base**omega)),int(newSizeSref[1]/(base**omega)),int(newSizeSref[2]/(base**omega)),newSizeSref[3]]
	newSref = np.zeros(newSizeSref)
	for ii in range(newSizeSref[0]):
		for jj in range(newSizeSref[1]):
			for kk in range(newSizeSref[2]):
				for l in range(newSizeSref[3]):
					## For each point, compute the mean of the corresponding square in image ##
					newSref[ii,jj,kk,l] = np.mean(image[ii*(base**omega):(ii+1)*(base**omega),jj*(base**omega):(jj+1)*(base**omega),kk*(base**omega):(kk+1)*(base**omega),l])
	return newSref


def downsamplingMean3D(image,omega,base=2):
	'''
	Downsample a 3D image by a specific factor using the average intensity value
    -------------------------------
    Input:  Sref	--> 3D image
			omega	-->	Factor of downsampling, {base}^{omega} (omega is
								considered positive)
			base	--> Base parameter of increase the scale in each step
    Output: newIm --> 3D downsampled image
	'''
	## Check if image size is divisible by 2^omega ##
	newSizeimage = np.array(image.shape)
	while (newSizeimage[0]%(base**omega) != 0):
		newSizeimage[0] += 1
	while (newSizeimage[1]%(base**omega) != 0):
		newSizeimage[1] += 1
	while (newSizeimage[2]%(base**omega) != 0):
		newSizeimage[2] += 1
	## Create image with the new size (divisible by 2^omega) ##
	newimage = np.zeros(newSizeimage)
	sizeimage = image.shape
	newimage[:sizeimage[0],:sizeimage[1],:sizeimage[2]] = image
	image = newimage.copy()
	## Switch to the small size ##
	newSizeimage = [newSizeimage[0]/(base**omega),newSizeimage[1]/(base**omega),newSizeimage[2]/(base**omega)]
	newimage = np.zeros(newSizeimage)
	for ii in range(newSizeimage[0]):
		for jj in range(newSizeimage[1]):
			for kk in range(newSizeimage[2]):
				## For each point, compute the mean of the corresponding square in image ##
				newimage[ii,jj,kk] = np.mean(image[ii*(base**omega):(ii+1)*(base**omega),jj*(base**omega):(jj+1)*(base**omega),kk*(base**omega):(kk+1)*(base**omega)])
	return newimage


def downsamplingGauss3D(image,omega,stdGaussian,base=2):
	'''
	Downsample a 3D image by a specific factor using a Gaussian filter
    -------------------------------
    Input:  image		--> 3D image
			omega		-->	Factor of downsampling, {base}^{omega} (omega is
								considered positive)
			stdGaussian	-->	Standard deviation of the Gaussian filter
			base		--> Base parameter of increase the scale in each step
    Output: newIm --> 3D downsampled image
	'''
	## Compute the guassian filter ##
	gaussimage = ndimage.gaussian_filter(image,sigma=stdGaussian)
	## Check if image size is divisible by 2^omega ##
	newSizeimage = np.array(gaussimage.shape)
	while (newSizeimage[0]%(base) != 0):
		newSizeimage[0] += 1
	while (newSizeimage[1]%(base) != 0):
		newSizeimage[1] += 1
	while (newSizeimage[2]%(base) != 0):
		newSizeimage[2] += 1
	## Create image with the new size (divisible by 2^omega) ##
	newimage = np.zeros(newSizeimage)
	sizeimage = gaussimage.shape
	newimage[:sizeimage[0],:sizeimage[1],:sizeimage[2]] = gaussimage
	gaussimage = newimage.copy()
	## Switch to the small size ##
	newSizeimage = [int(newSizeimage[0]/(base)),int(newSizeimage[1]/(base)),int(newSizeimage[2]/(base))]
	newimage = np.zeros(newSizeimage)
	for ii in range(newSizeimage[0]):
		for jj in range(newSizeimage[1]):
			for kk in range(newSizeimage[2]):
				## For each point, compute the mean of the corresponding square in image ##
				newimage[ii,jj,kk] = gaussimage[(ii)*base,(jj)*base,(kk)*base]
	return newimage


def downsamplingGauss4D(image,omega,stdGaussian,base=2):
	'''
	Downsample a 4D image by a specific factor using a Gaussian filter
    -------------------------------
    Input:  image		--> 3D image
			omega		-->	Factor of downsampling, {base}^{omega} (omega is
								considered positive)
			stdGaussian	-->	Standard deviation of the Gaussian filter
			base		--> Base parameter of increase the scale in each step
    Output: newIm --> 4D downsampled image
	'''
	## Compute the guassian filter ##
	gaussimage = np.zeros([image.shape[0],image.shape[1],image.shape[2],image.shape[3]])
	for l in range(image.shape[3]):
		gaussimage[:,:,:,l] = ndimage.gaussian_filter(image[:,:,:,l],sigma=stdGaussian)
	## Check if image size is divisible by 2^omega ##
	newSizeimage = np.array(gaussimage.shape)
	while (newSizeimage[0]%(base) != 0):
		newSizeimage[0] += 1
	while (newSizeimage[1]%(base) != 0):
		newSizeimage[1] += 1
	while (newSizeimage[2]%(base) != 0):
		newSizeimage[2] += 1
	## Create image with the new size (divisible by 2^omega) ##
	newimage = np.zeros(newSizeimage)
	sizeimage = gaussimage.shape
	newimage[:sizeimage[0],:sizeimage[1],:sizeimage[2],:sizeimage[3]] = gaussimage
	gaussimage = newimage.copy()
	## Switch to the small size ##
	newSizeimage = [newSizeimage[0]/(base),newSizeimage[1]/(base),newSizeimage[2]/(base),image.shape[3]]
	newimage = np.zeros(newSizeimage)
	for ii in range(newSizeimage[0]):
		for jj in range(newSizeimage[1]):
			for kk in range(newSizeimage[2]):
				for ll in range(image.shape[3]):
					## For each point, compute the mean of the corresponding square in image ##
					newimage[ii,jj,kk,ll] = gaussimage[(ii)*base,(jj)*base,(kk)*base,ll]
	return newimage
