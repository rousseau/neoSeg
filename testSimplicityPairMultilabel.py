# coding=utf-8

import os,sys
import numpy as np
import math
from skimage import measure
from scipy import ndimage
import nibabel
from testSimplicityPairBinary import isBinarySimplePoint,isBinarySimplePair2
from time import time



def isSimplePair(im,center,topologyList,totalLabel):
    simplicity=np.zeros([totalLabel],dtype=int)
    #im=Ilabel[x-1:x+2,y-1:y+2,z-1:z+2].copy()
    im_c=im.copy()
    #centralLabel=Ilabel[x,y,z]
    #im_c[im_c==Ilabel[x,y,z]]=-1
    im_c[im_c==im[center]]=-1
    candidatesLabel=np.unique(im_c)[1:]
    numLabel=len(candidatesLabel)
    ## Loop of labels in the neighbourhood of (x,y,z) ##

    while(numLabel>0):
        #evalLabel=candidatesLabel[numLabel-1]
        simplicityLabel=1
        numTopology=len(topologyList)
        ## Evaluate label simplicity in each topology ##
        while((simplicityLabel==1) & (numTopology>0)):
            ## Discard when current label and evalued label are in a topology group or there are not at all ##
            im_c=im.copy()
            ## Evaluation of all labels ##
            for l in xrange(totalLabel):
                im_c[im==l]=1 if l in topologyList[numTopology-1] else 0
            if len(np.unique(im_c))>1:
                simplicityLabel=isBinarySimplePair2(im_c,center)
            numTopology-=1
        simplicity[candidatesLabel[numLabel-1]]=simplicityLabel
        numLabel-=1
    return simplicity


def showSubplot(im,columns,title):
    numIm = im.shape[2]
    rows = int((numIm/columns))

    plt.figure(title)
    i = 0
    for i in range(numIm):
        plt.subplot(rows,columns,i+1)
        plt.imshow(im[i,:,:],cmap='gray')
        plt.title('z = '+str(i))
        plt.axis('off')




if __name__ == '__main__':
    '''
    Function to test:
        isSimple(Ilabel,topologyList,totalLabel,x,y,z)
                Input:
                    Ilabel			=> 	3D image of labels
                    topologyList 	=>	groups of labels to check its topology
                    totalLabel		=>	number of labels in Ilabel
                    x,y,z 			=>	point to be evaluated
                Return:
                    Binary array with a size equal to the number of labels
                    (totalLabel). Position in this array determines the evalued
                    label and its value is:
                         1 		if (x,y,z) is simple in 6-26 connectivity
                        0		otherwise
    '''



    ## Test Multilabel Simple Points ##
    ipath='/home/carlos/Test/Topology'
    Ilabel = nibabel.load(ipath+'/examples/Ilabel_3l.nii.gz')
    Ilabel = Ilabel.get_data().astype(int)
    x,y,z = 25,36,12 # Cortex point #
    x,y,z = 14,34,25 # Background point #
    x,y,z = 21,36,17 # Cortex point #
    x,y,z = 10,10,10 # Cortex point #
    x,y,z = 51,27,32 # Cortex % WM #
    x,y,z = 26,51,35 # Cortex point #
    x,y,z = 14,36,25 # Cortex point #

    #Ilabel=np.array([[[1,1,1],[1,1,1],[1,1,1]], [[1,1,1],[1,0,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]])
    #x,y,z = 1,1,1
    image = Ilabel[x-2:x+3,y-2:y+3,z-2:z+3]

    print '\nInput neighbourhood to be evaluated:\n\n'+str(image)+'\n'


    topologyList=[[1,2],[2,3],[1,2,3],[1],[0]]
    topologyList=[[1],[2],[3]]
    totalLabel=np.intc(3)


    t=time()


    padding = 1

    simpleMap = np.zeros((image.shape[0],image.shape[1],image.shape[2],totalLabel))
    for i in range(padding,image.shape[0]-padding):
        for j in range(padding,image.shape[1]-padding):
            for k in range(padding,image.shape[2]-padding):
                for l in range(totalLabel):
                    simpleMap[i,j,k,l] = isBinarySimplePoint(image[i-1:i+2,j-1:j+2,k-1:k+2])


    simplePairMap = np.zeros((image.shape[0],image.shape[1],image.shape[2],3,totalLabel))

    for l in xrange(totalLabel):
        binIm = 1*(image==l)

        homogeneous = 1*(ndimage.convolve(binIm,np.array([[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,-27,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]))==0)
        paddingImage = np.ones_like(binIm)
        paddingImage[padding:image.shape[0]-padding,padding:image.shape[1]-padding,padding:image.shape[2]-padding]=0
        mask = 1*((simpleMap[:,:,:,l] + homogeneous + paddingImage)!=0)
        mask = 1-mask

        maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0,0,0]],[[1,-1,0]],[[0,0,0]]]))==0)
        ct = np.where(maskFiltered==1)
        for ict in range(len(ct[0])):
            simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],0,:] = isSimplePair( image[ct[0][ict]-1:ct[0][ict]+2,ct[1][ict]-1:ct[1][ict]+2,ct[2][ict]-1:ct[2][ict]+3],([1,1],[1,1],[1,2]),topologyList,totalLabel )

        maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0],[0],[0]],[[1],[-1],[0]],[[0],[0],[0]]]))==0)
        ct = np.where(maskFiltered==1)
        for ict in range(len(ct[0])):
            simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],1,:] = isSimplePair( image[ct[0][ict]-1:ct[0][ict]+2,ct[1][ict]-1:ct[1][ict]+3,ct[2][ict]-1:ct[2][ict]+2],([1,1],[1,2],[1,1]),topologyList,totalLabel )

        maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0],[1],[0]],[[0],[-1],[0]],[[0],[0],[0]]]))==0)
        ct = np.where(maskFiltered==1)
        for ict in range(len(ct[0])):
            simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],2,:] = isSimplePair( image[ct[0][ict]-1:ct[0][ict]+3,ct[1][ict]-1:ct[1][ict]+2,ct[2][ict]-1:ct[2][ict]+2],([1,2],[1,1],[1,1]),topologyList,totalLabel )

    print 'Computing time: '+str(time()-t)+' s'

    print 'Result '+str(np.count_nonzero(simplePairMap==1))
