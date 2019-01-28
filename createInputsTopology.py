#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

  This software is governed by the CeCILL-B license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL-B
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited
  liability.

  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.

  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL-B license and that you accept its terms.

  #####################################################################
  Example:
  python patchAnalysis.py -i ~/Data/ALBERT_makropoulos/T2/ALBERT_01.nii.gz
        -s ~/Data/ALBERT_makropoulos/gm-posteriors-v3/ALBERT_01.nii.gz
        -x 10 -y 76 -z 65
  #####################################################################
"""

import argparse,sys,os
import nibabel
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from time import time
import itertools
from skimage import transform,measure
import math
from numba import jit
import nipype.algorithms.metrics as nipalg
import random
import imp
sendemail=imp.load_source('sendemail', '/home/carlos/Code/sendemail.py')





def addBackgroundToSref(S):
    Sref=np.zeros([S.shape[0],S.shape[1],S.shape[2],S.shape[3]+1])
    Sref[:,:,:,1:]=S
    #Sref[:,:,:,0]=1-np.max(S,axis=-1)
    Sref[:,:,:,0]=1-np.sum(S,axis=-1)
    return Sref

def constructSref(segList,delta,addBackground=1):
    Sref=np.zeros([segList[0].shape[0]+delta[0],segList[0].shape[1]+delta[1],segList[0].shape[2]+delta[2],len(segList)])
    for l,S in enumerate(segList):
        Sref[:segList[0].shape[0],:segList[0].shape[1],:segList[0].shape[2],l]=S
    ##Normalise##
    Sref/=np.max(np.sum(Sref,axis=-1))
    if addBackground==1:
        Sref=addBackgroundToSref(Sref)

    ###normalize##
    #max=np.sum(Sref,axis=3)
    #for l in xrange(len(segList)+1):
    #    Sref[:,:,:,l]/=max
    return Sref


def tracePerimeter(im):
    distance = ndimage.distance_transform_cdt(im,metric='taxicab')
    #distance[distance!=1] = 0
    #distance = ndimage.distance_transform_edt(im)
    distance[distance>3] = 0
    distance[distance<1] = 0
    return distance

def getNeighbourCoordonates(x,y,z,sizeIm,connectivity):
	minx=x-1 if x!=0 else 0
	miny=y-1 if y!=0 else 0
	minz=z-1 if z!=0 else 0
	maxx=x+2 if x!=sizeIm[0] else sizeIm[0]+1
	maxy=y+2 if y!=sizeIm[1] else sizeIm[1]+1
	maxz=z+2 if z!=sizeIm[2] else sizeIm[2]+1

	coordinates=list()
	for ii in range(minx,maxx):
		for jj in range(miny,maxy):
			for kk in range(minz,maxz):
				if connectivity==26:
					coordinates.append((ii,jj,kk))
				elif 1*(ii==x) + 1*(jj==y) + 1*(kk==z) >=2:
					coordinates.append((ii,jj,kk))

	return list(set(coordinates))


def constructSyntheticIlabel(sizeRef,omega0):
    newSizeSref=list(sizeRef)
    while (newSizeSref[0]%(2**omega0) != 0):
        newSizeSref[0]+=1
    while (newSizeSref[1]%(2**omega0) != 0):
        newSizeSref[1]+=1
    while (newSizeSref[2]%(2**omega0) != 0):
        newSizeSref[2]+=1
    sizeIlabel=[newSizeSref[0]/(2**omega0),newSizeSref[1]/(2**omega0),newSizeSref[2]/(2**omega0)]
    Ilabel=np.zeros(sizeIlabel)

    centroid=(int(sizeIlabel[0]/2),int(sizeIlabel[1]/2),int(sizeIlabel[2]/2))

    rWm,rCortex,rCsf=int(np.min(centroid)*0.7),int(np.min(centroid)*0.8),int(np.min(centroid)*0.9)

    x,y,z = np.ogrid[-centroid[0]:sizeIlabel[0]-centroid[0], -centroid[1]:sizeIlabel[1]-centroid[1], -centroid[2]:sizeIlabel[2]-centroid[2]]
    wm = x**2 + y**2 + z**2 <= rWm**2
    cortex = x**2 + y**2 + z**2 <= rCortex**2
    csf = x**2 + y**2 + z**2 <= rCsf**2

    if omega0>2:
        Ilabel[wm]=1
    else:
        Ilabel[csf]=3
        Ilabel[cortex]=2
        Ilabel[wm]=1

    return Ilabel,newSizeSref




def constructIlabelBySref(Sref,omega0):
    newSizeSref=Sref.shape
    sizeIlabel=[newSizeSref[0]/(2**omega0),newSizeSref[1]/(2**omega0),newSizeSref[2]/(2**omega0)]
    Ilabel=np.zeros(sizeIlabel)

    newSref=downsampling(Sref,omega)
    Ilabel[newSref[:,:,:,0]<0.5]=1

    ## Fill holes ##
    Ilabel=ndimage.morphology.binary_fill_holes(Ilabel)
    Ilabel=ndimage.morphology.binary_closing(Ilabel,iterations=5)
    Ilabel=Ilabel.astype(np.int)

    if omega0<=2:
        ## Perimeters ##
        csf=tracePerimeter(Ilabel)
        cortexWm=Ilabel.copy()
        cortexWm[csf==1]=0
        cortex=tracePerimeter(cortexWm)
        wm=cortexWm.copy()
        wm[cortex==1]=0


        Ilabel[csf==1]=3
        Ilabel[cortex==1]=2
        Ilabel[wm==1]=1

    return Ilabel




def constructIlabelByCentroidSref(Sref,omega0,threshold=1):
    newSizeSref=Sref.shape
    sizeIlabel=[newSizeSref[0]/(2**omega0),newSizeSref[1]/(2**omega0),newSizeSref[2]/(2**omega0)]
    Ilabel=np.zeros(sizeIlabel)

    ## Threshold the downsampled Sref##
    newSref=downsampling(Sref,omega0)
    Ilabel[newSref[:,:,:,0]<threshold]=1

    ## Fill holes ##
    Ilabel=ndimage.morphology.binary_fill_holes(Ilabel)

    ## Keep the biggest part ##
    labels,numLabels=measure.label(Ilabel,connectivity=1,return_num=True)
    max=0
    for l in range(1,numLabels+1):
        if (len(np.where(labels==l)[0]) > max):
            max=len(np.where(labels==l)[0])
            label=l

    Ilabel[labels!=label]=0
    Ilabel[labels==label]=1
    Ilabel=Ilabel.astype(np.int)

    centroid=ndimage.measurements.center_of_mass(Ilabel)
    centroid=[int(c) for c in centroid]
    x,y,z = np.ogrid[-centroid[0]:sizeIlabel[0]-centroid[0], -centroid[1]:sizeIlabel[1]-centroid[1], -centroid[2]:sizeIlabel[2]-centroid[2]]
    points=np.where(Ilabel==1)
    rx,ry,rz=int(np.min([centroid[0]-np.min(points[0]),np.max(points[0])-centroid[0]])),int(np.min([centroid[1]-np.min(points[1]),np.max(points[1])-centroid[1]])),int(np.min([centroid[2]-np.min(points[2]),np.max(points[2])-centroid[2]]))
    r=int(np.min([rx,ry,rz]))
    brain = x**2 + y**2 + z**2 <= r**2


    #4labels#Ilabel=np.zeros(sizeIlabel)
    #4labels#Ilabel[brain]=1
    #4labels#points=np.where(Ilabel==1)
    #4labels#for ipt in range(len(points[0])):
    #4labels#    x,y,z=points[0][ipt],points[1][ipt],points[2][ipt]
    #4labels#    neighbours26=getNeighbourCoordonates(x,y,z,Ilabel.shape,connectivity=26)
    #4labels#    if (len([nb for nb in neighbours26 if Ilabel[nb[0],nb[1],nb[2]]==0]) > 0):
    #4labels#        Ilabel[x,y,z]=3
    #4labels#points=np.where(Ilabel==1)
    #4labels#for ipt in range(len(points[0])):
    #4labels#    x,y,z=points[0][ipt],points[1][ipt],points[2][ipt]
    #4labels#    neighbours26=getNeighbourCoordonates(x,y,z,Ilabel.shape,connectivity=26)
    #4labels#    if (len([nb for nb in neighbours26 if Ilabel[nb[0],nb[1],nb[2]]==3]) > 0):
    #4labels#        Ilabel[x,y,z]=2

    Ilabel=np.zeros(sizeIlabel)
    Ilabel[brain]=1
    points=np.where(Ilabel==1)
    for ipt in range(len(points[0])):
        x,y,z=points[0][ipt],points[1][ipt],points[2][ipt]
        neighbours26=getNeighbourCoordonates(x,y,z,Ilabel.shape,connectivity=26)
        if (len([nb for nb in neighbours26 if Ilabel[nb[0],nb[1],nb[2]]==0]) > 0):
            Ilabel[x,y,z]=2


    return Ilabel


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



def buildInputs(labelsPath,opath,omega,threshold=1):
    header=nibabel.load(labelsPath[0]).affine
    S=list()
    for labPath in labelsPath:
        S.append(nibabel.load(labPath).get_data())
    sizeRef=S[0].shape

    newSizeSref=list(sizeRef)
    while (newSizeSref[0]%(2**omega) != 0):
		newSizeSref[0]+=1
    while (newSizeSref[1]%(2**omega) != 0):
    	newSizeSref[1]+=1
    while (newSizeSref[2]%(2**omega) != 0):
    	newSizeSref[2]+=1
    sizeIlabel=[newSizeSref[0]/(2**omega),newSizeSref[1]/(2**omega),newSizeSref[2]/(2**omega)]

    delta=[newSizeSref[0]-sizeRef[0],newSizeSref[1]-sizeRef[1],newSizeSref[2]-sizeRef[2]]

    Sref=constructSref(S,delta,1)
    nibabel.save(nibabel.Nifti1Image(Sref,header),opath+'/Sref_3l.nii.gz')

    Ilabel=constructIlabelByCentroidSref(Sref,omega,threshold)
    nibabel.save(nibabel.Nifti1Image(Ilabel.astype(np.float),header),opath+'/Ilabel_3l.nii.gz')



def buildTestFarSpheres2D(sizeIm,numLabels,header,opath,omega):

    Sref=np.zeros([sizeIm[0],sizeIm[1],numLabels])
    Ilabel=np.zeros([sizeIm[0],sizeIm[1]])
    centroid=(int(sizeIm[0]/2),int(sizeIm[1]/2))
    rWm,rCortex,rCsf=int(np.min(centroid)*0.7),int(np.min(centroid)*0.8),int(np.min(centroid)*0.9)
    x,y = np.ogrid[-centroid[0]:sizeIm[0]-centroid[0], -centroid[1]:sizeIm[1]-centroid[1]]
    wm = x**2 + y**2 <= rWm**2
    cortex = x**2 + y**2 <= rCortex**2
    #csf = x**2 + y**2 + z**2 <= rCsf**2
    #Ilabel[csf]=3
    Ilabel[cortex]=2
    Ilabel[wm]=1
    for i in xrange(numLabels):
        Sref[:,:,i]=1*Ilabel==i
    nibabel.save(nibabel.Nifti1Image(Sref,header),opath+'/testFarSpheres2D_Sref.nii.gz')

    Ilabel=np.zeros([sizeIm[0],sizeIm[1]])
    rWm,rCortex,rCsf=int(np.min(centroid)*0.3),int(np.min(centroid)*0.4),int(np.min(centroid)*0.5)
    centroid=(int(sizeIm[0]/2),int(sizeIm[1]/2))
    x,y = np.ogrid[-centroid[0]:sizeIm[0]-centroid[0], -centroid[1]:sizeIm[1]-centroid[1]]
    wm = x**2 + y**2 <= rWm**2
    cortex = x**2 + y**2 <= rCortex**2
    #csf = x**2 + y**2 + z**2 <= rCsf**2
    #Ilabel[csf]=3
    Ilabel[cortex]=2
    Ilabel[wm]=1
    nibabel.save(nibabel.Nifti1Image(Ilabel.astype(np.float),header),opath+'/testFarSpheres2D_Ilabel.nii.gz')


def buildTestFarSpheres(sizeIm,numLabels,header,opath,omega):

    Sref=np.zeros([sizeIm[0],sizeIm[1],sizeIm[2],numLabels])
    Ilabel=np.zeros([sizeIm[0],sizeIm[1],sizeIm[2]])
    centroid=(int(sizeIm[0]/2),int(sizeIm[1]/2),int(sizeIm[2]/2))
    rWm,rCortex,rCsf=int(np.min(centroid)*0.7),int(np.min(centroid)*0.8),int(np.min(centroid)*0.9)
    x,y,z = np.ogrid[-centroid[0]:sizeIm[0]-centroid[0], -centroid[1]:sizeIm[1]-centroid[1], -centroid[2]:sizeIm[2]-centroid[2]]
    wm = x**2 + y**2 + z**2 <= rWm**2
    cortex = x**2 + y**2 + z**2 <= rCortex**2
    #csf = x**2 + y**2 + z**2 <= rCsf**2
    #Ilabel[csf]=3
    Ilabel[cortex]=2
    Ilabel[wm]=1
    for i in xrange(numLabels):
        Sref[:,:,:,i]=1*Ilabel==i
    nibabel.save(nibabel.Nifti1Image(Sref,header),opath+'/testFarSpheres_Sref.nii.gz')

    Ilabel=np.zeros([sizeIm[0],sizeIm[1],sizeIm[2]])
    rWm,rCortex,rCsf=int(np.min(centroid)*0.3),int(np.min(centroid)*0.4),int(np.min(centroid)*0.5)
    centroid=(int(sizeIm[0]/2),int(sizeIm[1]/2),int(sizeIm[2]/2))
    x,y,z = np.ogrid[-centroid[0]:sizeIm[0]-centroid[0], -centroid[1]:sizeIm[1]-centroid[1], -centroid[2]:sizeIm[2]-centroid[2]]
    wm = x**2 + y**2 + z**2 <= rWm**2
    cortex = x**2 + y**2 + z**2 <= rCortex**2
    #csf = x**2 + y**2 + z**2 <= rCsf**2
    #Ilabel[csf]=3
    Ilabel[cortex]=2
    Ilabel[wm]=1
    nibabel.save(nibabel.Nifti1Image(Ilabel.astype(np.float),header),opath+'/testFarSpheres_Ilabel.nii.gz')


def buildTestCrossEllipsoids(sizeIm,numLabels,header,opath,omega):

    Sref=np.zeros([sizeIm[0],sizeIm[1],sizeIm[2],numLabels])
    Ilabel=np.zeros([sizeIm[0],sizeIm[1],sizeIm[2]])
    centroid=(int(sizeIm[0]/2),int(sizeIm[1]/2),int(sizeIm[2]/2))
    rWm,rCortex,rCsf=int(np.min(centroid)*0.7),int(np.min(centroid)*0.8),int(np.min(centroid)*0.9)
    x,y,z = np.linspace(0,sizeIm[0]-1,sizeIm[0]),np.linspace(0,sizeIm[1]-1,sizeIm[1])[:,None],np.linspace(0,sizeIm[2]-1,sizeIm[2])[:,None,None]
    Rx,Ry,Rz = [8,10],[16,18],[8,10]
    for i in list(reversed(range(len(Rx)))):
        mask = ((x-centroid[0])/Rx[i])**2 + ((y-centroid[1])/Ry[i])**2 + ((z-centroid[2])/Rz[i])**2 <= 1
        Ilabel[mask] = i+1
    for i in xrange(numLabels):
        Sref[:,:,:,i]=1*Ilabel==i
    nibabel.save(nibabel.Nifti1Image(Sref,header),opath+'/testCrossEllipsoids_Sref.nii.gz')

    Ilabel=np.zeros([sizeIm[0],sizeIm[1],sizeIm[2]])
    centroid=(int(sizeIm[0]/2),int(sizeIm[1]/2),int(sizeIm[2]/2))
    rWm,rCortex,rCsf=int(np.min(centroid)*0.3),int(np.min(centroid)*0.4),int(np.min(centroid)*0.5)
    x,y,z = np.linspace(0,sizeIm[0]-1,sizeIm[0]),np.linspace(0,sizeIm[1]-1,sizeIm[1])[:,None],np.linspace(0,sizeIm[2]-1,sizeIm[2])[:,None,None]
    Rx,Ry,Rz = [8,10],[8,10],[16,18]
    for i in list(reversed(range(len(Rx)))):
        mask = ((x-centroid[0])/Rx[i])**2 + ((y-centroid[1])/Ry[i])**2 + ((z-centroid[2])/Rz[i])**2 <= 1
        Ilabel[mask] = i+1
    nibabel.save(nibabel.Nifti1Image(Ilabel.astype(np.float),header),opath+'/testCrossEllipsoids_Ilabel.nii.gz')


if __name__ == '__main__':

    ipath='/home/carlos/Test/IMAPA/31/results'
    opath='/home/carlos/Code/neoSeg/examples'
    opath='/home/carlos/Test/Topology/examples'
    os.system('mkdir -p '+opath)

    labelsPath=list()
    labelsPath.append(ipath+'/WM/Simapa_wm_atlasdHCP30_K15_hps1_hss3_alpha0.25.nii.gz')
    labelsPath.append(ipath+'/Cortex/Simapa_cortex_atlasdHCP30_K15_hps1_hss3_alpha0.25.nii.gz')
    #labelsPath.append(ipath+'/CSF/Simapa_csf_atlasdHCP30_K15_hps1_hss3_alpha0.25.nii.gz')

    omega=4

    threshold=1

    #buildInputs(labelsPath,opath,omega,threshold)

    sizeIm=[50,50,50]
    numLabels=3
    header=nibabel.load(labelsPath[0]).affine
    buildTestFarSpheres2D(sizeIm,numLabels,header,opath,omega)

    #buildTestFarSpheres(sizeIm,numLabels,header,opath,omega)
    #buildTestCrossEllipsoids(sizeIm,numLabels,header,opath,omega)
