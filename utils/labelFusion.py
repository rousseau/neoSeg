# -*- coding: utf-8 -*-

import argparse,sys,os
import numpy as np
import nibabel
import math
from scipy import optimize,ndimage
from time import time
from numba import jit






def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out



def computeSigmaNoise(img):
    hx=np.array([[[0.,0.,0.],[0.,-(1./6.),0.],[0.,0.,0.]],
                    [[0.,-(1./6.),0.],[-(1./6.),1.,-(1./6.)],[0.,-(1./6.),0.]],
                    [[0.,0.,0.],[0.,-(1./6.),0.],[0.,0.,0.]]])
    sigma2=(6.0/7.0) * np.sum(np.square(ndimage.convolve(img,hx))) / np.float(img.shape[0]*img.shape[1]*img.shape[2])
    return sigma2


def computeSigmaNoiseMask(img,mask):
    img[mask==0]=0
    hx=np.array([[[0.,0.,0.],[0.,-(1./6.),0.],[0.,0.,0.]],
                    [[0.,-(1./6.),0.],[-(1./6.),1.,-(1./6.)],[0.,-(1./6.),0.]],
                    [[0.,0.,0.],[0.,-(1./6.),0.],[0.,0.,0.]]])
    sigma2=(6.0/7.0) * np.sum(np.square(ndimage.convolve(img,hx))) / np.float(img.shape[0]*img.shape[1]*img.shape[2])
    return sigma2



@jit
def computeNLMValue(I,I_examples,S_examples,i,j,k,hps,hss,h,num_atlas,num_examples,K): #(i,j,k):
    ##Define the search zone##
    xmin = i-hss[0]
    ymin = j-hss[1]
    zmin = k-hss[2]
    xmax = i+hss[0]+1
    ymax = j+hss[1]+1
    zmax = k+hss[2]+1

    #sumW = np.float(0.0)
    #outputValue = np.float(0.0)
    weight=np.zeros([num_examples]) #list()
    dlabels=np.zeros([num_examples])
    count=0
    for a in range(num_atlas):
        for ii in range(xmin,xmax):
            for jj in range(ymin,ymax):
                for kk in range(zmin,zmax):

                    #dist = np.float(0.0)
                    for iii in range(-hps[0],hps[0]+1):
                        for jjj in range(-hps[1],hps[1]+1):
                            for kkk in range(-hps[2],hps[2]+1):
                                weight[count] += (I[i+iii,j+jjj,k+kkk]-I_examples[a,ii+iii,jj+jjj,kk+kkk])**2#*(input[i+iii,j+jjj,k+kkk]-references[a,ii+iii,jj+jjj,kk+kkk])
                    dlabels[count]=S_examples[a,ii,jj,kk]
                    #weight.append(np.exp(-dist/h))
                    #weight[count]=dist
                    count+=1
                    #sumW += weight
                    #outputValue += weight*S_examples[a,ii,jj,kk]

    #if sum>0:
    #    outputValue/=sum
    selection1=sorted(np.asarray(sorted(range(len(weight)), key=weight.__getitem__)[:K]))
    sumW = np.float(0.0)
    outputValue=np.float(0.0)
    for is1 in selection1:
        w=np.exp(-(weight[is1]/h))
        outputValue+=w*dlabels[is1]
        sumW+=w


    return outputValue/sumW





@jit
def computeNLMWeight(I,I_examples,S_examples,i,j,k,hps,hss,h,num_atlas,K):#(i,j,k):
    ##Define the search zone##
    xmin = i-hss[0]
    ymin = j-hss[1]
    zmin = k-hss[2]
    xmax = i+hss[0]+1
    ymax = j+hss[1]+1
    zmax = k+hss[2]+1

    #sumW = np.float(0.0)

    weight=np.zeros([num_atlas*(2*hss[0]+1)*(2*hss[1]+1)*(2*hss[2]+1)])
    count=0
    for a in range(num_atlas):
        for ii in range(xmin,xmax):
            for jj in range(ymin,ymax):
                for kk in range(zmin,zmax):

                    dist = np.float(0.0)
                    for iii in range(-hps[0],hps[0]+1):
                        for jjj in range(-hps[1],hps[1]+1):
                            for kkk in range(-hps[2],hps[2]+1):
                                dist += (I[i+iii,j+jjj,k+kkk]-I_examples[a,ii+iii,jj+jjj,kk+kkk])**2#*(input[i+iii,j+jjj,k+kkk]-references[a,ii+iii,jj+jjj,kk+kkk])
                    weight[count] = dist
                    #sumW+=weight[count]
                    count+=1
    selection1=sorted(np.asarray(sorted(range(len(weight)), key=weight.__getitem__)[:K]))
    w=np.exp(-(weight[selection1]/h))

    #w=np.zeros([K])
    #sumW = np.float(0.0)
    #for x in range(K):
    #    w[x]=weight[selection1[x]]
    #    sumW+=weight[selection1[x]]
    return [selection1,w/np.sum(w)]

def WNLM_threads(args):
    (input,references,labels,points,hps,hss,h,num_atlas,K)=args
    wnlm=np.zeros([len(points),2,K])
    for iv,vx in enumerate(points):
        wnlm[iv,:,:]=computeNLMWeight(input,references,labels,vx[0],vx[1],vx[2],hps,hss,h,num_atlas,K)
    return wnlm


#@profile
def NLM_threads(args):
    (input,references,labels,points,hps,hss,h,K)=args
    S=np.zeros(input.shape)
    num_atlas=references.shape[0]
    num_examples=(2*hss[0]+1)*(2*hss[1]+1)*(2*hss[2]+1)*(num_atlas)
    for vx in points:
        S[vx[0],vx[1],vx[2]]=computeNLMValue(input,references,labels,vx[0],vx[1],vx[2],hps,hss,h,num_atlas,num_examples,K)
    return S


#@jit
def function_phi(x,C,b):
    return np.linalg.norm(np.dot(C,x)-b)  #(np.dot(C,x)-b)**2




@jit
def computeITERNLMValueTotal(I,I_examples,S_estimated,S_examples,i,j,k,hps,hss,num_atlas,num_examples,patch_size,alpha,beta,h1,h2):  #,Wini,K  #(i,j,k):

    ##Define the search zone##
    xmin = i-hss[0]
    ymin = j-hss[1]
    zmin = k-hss[2]
    xmax = i+hss[0]+1
    ymax = j+hss[1]+1
    zmax = k+hss[2]+1

    outputValue=0.0
    sumW=0.0
    ##Loop on atlas##
    for a in range(num_atlas):
        ##Loop on search zone##
        for ii in range(xmin,xmax):
            for jj in range(ymin,ymax):
                for kk in range(zmin,zmax):

                    ##Loop on patch##
                    distance=0.0
                    distance_segmentation=0.0
                    for iii in range(-hps[0],hps[0]+1):
                        for jjj in range(-hps[1],hps[1]+1):
                            for kkk in range(-hps[2],hps[2]+1):
                                distance+=(I_examples[a,ii+iii,jj+jjj,kk+kkk]-I[i+iii,j+jjj,k+kkk])**2
                                distance_segmentation+=(S_examples[a,ii+iii,jj+jjj,kk+kkk]-S_estimated[i+iii,j+jjj,k+kkk])**2

                    outputValue += np.exp(-(beta*distance/h1+alpha*distance_segmentation/h2))*S_examples[a,ii,jj,kk]
                    sumW += np.exp(-(beta*distance/h1+alpha*distance_segmentation/h2))

    ##Normalise segmentation value avoiding division by 0##
    if sumW==0.0:
        sumW=1.0

    return outputValue/sumW



@jit
def computeITERNLMValueK(I,I_examples,S_estimated,S_examples,i,j,k,hps,hss,num_atlas,K,num_examples,patch_size,alpha,beta,h1,h2,iK,reg=1e-03):  #,Wini,K  #(i,j,k):

    ##Define the search zone##
    xmin = i-hss[0]
    ymin = j-hss[1]
    zmin = k-hss[2]
    xmax = i+hss[0]+1
    ymax = j+hss[1]+1
    zmax = k+hss[2]+1

    count=0
    distance=np.zeros([num_examples])
    distance_segmentation=np.zeros([num_examples])
    ##Loop on atlas##
    for a in range(num_atlas):
        ##Loop on search zone##
        for ii in range(xmin,xmax):
            for jj in range(ymin,ymax):
                for kk in range(zmin,zmax):

                    ##Loop on patch##
                    for iii in range(-hps[0],hps[0]+1):
                        for jjj in range(-hps[1],hps[1]+1):
                            for kkk in range(-hps[2],hps[2]+1):
                                distance[count]+=(I_examples[a,ii+iii,jj+jjj,kk+kkk]-I[i+iii,j+jjj,k+kkk])**2
                                distance_segmentation[count]+=(S_examples[a,ii+iii,jj+jjj,kk+kkk]-S_estimated[i+iii,j+jjj,k+kkk])**2
                    count+=1
    #distance=np.exp(-distance/h1)
    #distance_segmentation=np.exp(-distance_segmentation/h2)
    #dist=beta*distance+alpha*distance_segmentation
    dist=np.exp(-((beta*distance/h1)+(alpha*distance_segmentation/h2)))
    #dist=np.exp(-(distance/h1))

    #KNN=sorted(np.asarray(sorted(range(len(dist)), key=dist.__getitem__)[-K:]))
    KNN=iK

    weights1=np.zeros([K])
    for ix,x in enumerate(KNN):
        weights1[ix]=dist[x]

    count=0
    count1=0
    sumW1=np.float(0.0)
    outputValue1 = np.float(0.0)
    for a in range(num_atlas):
        for ii in range(xmin,xmax):
            for jj in range(ymin,ymax):
                for kk in range(zmin,zmax):
                    if count in KNN:
                        outputValue1 += weights1[count1]*S_examples[a,ii,jj,kk]
                        sumW1+=weights1[count1]
                        count1+=1
                    count+=1

    ##Normalise segmentation value avoiding division by 0##
    if sumW1==0.:
        sumW1=1.

    return outputValue1/sumW1



@jit
def computeITERValueAutoK(I, I_examples, S_estimated, S_examples, i, j, k, hps, hss, num_atlas, num_examples, patch_size, sqrt_alpha, sqrt_1_alpha, mu1, sigma1, reg=1e-03, epsilon=1e-06): #, mean_patch, var_patch, reg=1e-03, epsilon=1e-06

    ##Define the search zone##
    xmin = i-hss[0]
    ymin = j-hss[1]
    zmin = k-hss[2]
    xmax = i+hss[0]+1
    ymax = j+hss[1]+1
    zmax = k+hss[2]+1

    mean_patch=np.mean(I[i-hps[0]:i+hps[0]+1,j-hps[1]:j+hps[1]+1,k-hps[2]:k+hps[2]+1])
    var_patch=np.var(I[i-hps[0]:i+hps[0]+1,j-hps[1]:j+hps[1]+1,k-hps[2]:k+hps[2]+1])

    loop_examples=0
    K=0
    distance=np.ones([num_examples])*num_examples*2.
    Z=np.zeros([num_examples,2*patch_size])
    ##Loop on atlas##
    for a in range(num_atlas):
        ##Loop on search zone##
        for ii in range(xmin,xmax):
            for jj in range(ymin,ymax):
                for kk in range(zmin,zmax):

                    ##Loop on patch##
                    loop_patch = 0
                    ##Thresholding examples##
                    if exampleThreshold(mu1,sigma1,mean_patch,var_patch,I_examples[a,ii-hps[0]:ii+hps[0]+1,jj-hps[1]:jj+hps[1]+1,kk-hps[2]:kk+hps[2]+1]):
                        for iii in range(-hps[0],hps[0]+1):
                            for jjj in range(-hps[1],hps[1]+1):
                                for kkk in range(-hps[2],hps[2]+1):
                                    ##Compute LLE input##
                                    Z[loop_examples,loop_patch]=sqrt_1_alpha*(I_examples[a,ii+iii,jj+jjj,kk+kkk]-I[i+iii,j+jjj,k+kkk])
                                    Z[loop_examples,loop_patch+patch_size]=sqrt_alpha*(S_examples[a,ii+iii,jj+jjj,kk+kkk]-S_estimated[i+iii,j+jjj,k+kkk])
                                    ##Compute distance for initialise LLE##
                                    distance[loop_examples]+=Z[loop_examples,loop_patch]**2
                                    distance[loop_examples]+=Z[loop_examples,loop_patch+patch_size]**2
                                    loop_patch += 1
                        K+=1
                    loop_examples+=1


    ##Selection of KNN##
    KNN=sorted(np.asarray(sorted(range(len(distance)), key=distance.__getitem__)[:K]))
    distance=(distance+epsilon)**-0.5


    weights_ini=np.zeros([K])
    Z_knn=np.zeros([K,2*patch_size])
    for ix,x in enumerate(KNN):
        weights_ini[ix]=distance[x]
        Z_knn[ix,:]=Z[x,:]
    ##KNN from distance for initialise LLE##
    weights_ini/=np.max(weights_ini)


    ##Compute LLE solution##
    ##Define bounds of weights##
    bounds=list()
    for x in xrange(K):
        bounds.append( (0.,1.) )#np.finfo(float).eps
    ones=np.ones(K)
    C_knn=np.dot(Z_knn,np.transpose(Z_knn))
    ##Add a identity matrix with a constant regularisation (reg)
    trace = np.trace(C_knn)
    R = reg * trace if trace>0 else reg
    C_knn.flat[::K + 1] += R
    ##Compute LLE weights##
    weights = optimize.minimize(function_phi, weights_ini, args=(C_knn,ones), method='L-BFGS-B', bounds=bounds)['x']


    ##Estimate segmentation with new weights##
    loop_examples=0
    loop_K=0
    sum_w=0.0
    S_new = 0.0
    ##Loop on atlas x search zone##
    for a in range(num_atlas):
        for ii in range(xmin,xmax):
            for jj in range(ymin,ymax):
                for kk in range(zmin,zmax):
                    ##Select KNN##
                    if loop_examples in KNN:
                        S_new += weights[loop_K]*S_examples[a,ii,jj,kk]
                        sum_w+=weights[loop_K]
                        loop_K+=1
                    loop_examples+=1

    ##Normalise segmentation value avoiding division by 0##
    if sum_w==0.0:
        sum_w=1.0

    return S_new/sum_w





@jit
def computeNormL1(input,examples,vx,hps,hss,num_examples,patch_size):

    xmin = vx[0]-hss[0]
    ymin = vx[1]-hss[1]
    zmin = vx[2]-hss[2]
    xmax = vx[0]+hss[0]+1
    ymax = vx[1]+hss[1]+1
    zmax = vx[2]+hss[2]+1
    normL1=np.zeros([num_examples,patch_size])
    loop_examples=0
    for a in range(num_atlas):
        ##Loop on search zone##
        for ii in range(xmin,xmax):
            for jj in range(ymin,ymax):
                for kk in range(zmin,zmax):
                    ##Loop on patch##
                    loop_patch = 0
                    for iii in range(-hps[0],hps[0]+1):
                        for jjj in range(-hps[1],hps[1]+1):
                            for kkk in range(-hps[2],hps[2]+1):
                                ##Compute LLE input##
                                normL1[loop_examples,loop_patch]=(examples[a,ii+iii,jj+jjj,kk+kkk]-input[vx[0]+iii,vx[1]+jjj,vx[2]+kkk])
                                loop_patch += 1
                    loop_examples+=1

    return normL1

#@jit
def computeOptimalWeights(data,weights_ini,K,reg=1e-03):
    ##KNN from distance for initialise LLE##
    weights_ini/=np.max(weights_ini)


    ##Compute LLE solution##
    ##Define bounds of weights##
    #bounds=list()
    bounds=((0.,1.),)*K
    #for x in xrange(K):
    #    bounds.append( (0.,1.) )#np.finfo(float).eps
    ones=np.ones(K)
    C=np.dot(data,np.transpose(data))
    ##Add a identity matrix with a constant regularisation (reg)
    trace = np.trace(C)
    R = reg * trace if trace>0 else reg
    C.flat[::K + 1] += R
    ##Compute LLE weights##
    weights = optimize.minimize(function_phi, weights_ini, args=(C,ones), method='L-BFGS-B', bounds=bounds)['x']

    return weights

@jit
def computeMeanPatch(examples,vx,hps,hss,num_examples):

    xmin = vx[0]-hss[0]
    ymin = vx[1]-hss[1]
    zmin = vx[2]-hss[2]
    xmax = vx[0]+hss[0]+1
    ymax = vx[1]+hss[1]+1
    zmax = vx[2]+hss[2]+1
    meanPatch=np.zeros([num_examples])
    loop_examples=0
    for a in range(num_atlas):
        ##Loop on search zone##
        for ii in range(xmin,xmax):
            for jj in range(ymin,ymax):
                for kk in range(zmin,zmax):
                    ##Loop on patch##
                    meanPatch[loop_examples]=np.mean(examples[a,ii-hps[0]:ii+hps[0]+1,jj-hps[1]:jj+hps[1]+1,kk-hps[2]:kk+hps[2]+1])
                    loop_examples+=1

    return meanPatch


@jit
def computeIMAPAValue(I, I_examples, S_estimated, S_examples, vx, hps, hss, patch_size, num_examples, num_atlas, K, alpha1_sqrt, alpha_sqrt, reg=1e-03, epsilon=1e-06): #, mean_patch, var_patch, reg=1e-03, epsilon=1e-06

    ##Define the search zone##
    i=vx[0]
    j=vx[1]
    k=vx[2]
    hssi=hss[0]
    hssj=hss[1]
    hssk=hss[2]
    hpsi=hps[0]
    hpsj=hps[1]
    hpsk=hps[2]

    xmin = i-hssi
    ymin = j-hssj
    zmin = k-hssk
    xmax = i+hssi+1
    ymax = j+hssj+1
    zmax = k+hssk+1

    #alpha1_sqrt=(1-alpha)**0.5
    #alpha_sqrt=(alpha)**0.5
    #patch_size = (2*hpsi+1)*(2*hpsj+1)*(2*hpsk+1)
    #num_examples = num_atlas*(2*hssi+1)*(2*hssj+1)*(2*hssk+1)

    loop_examples=0
    S_order=np.zeros([num_examples])
    distance=epsilon*np.ones([num_examples])
    Z=np.zeros([num_examples,2*patch_size])
    '''##Loop on atlas##'''
    for a in range(num_atlas):
        '''##Loop on search zone##'''
        for ii in range(xmin,xmax):
            for jj in range(ymin,ymax):
                for kk in range(zmin,zmax):

                    '''##Loop on patch##'''
                    loop_patch = 0
                    for iii in range(-hpsi,hpsi+1):
                        for jjj in range(-hpsj,hpsj+1):
                            for kkk in range(-hpsk,hpsk+1):
                                '''##Compute LLE input##'''
                                Z[loop_examples,loop_patch] = alpha1_sqrt * (I_examples[a,ii+iii,jj+jjj,kk+kkk]-I[i+iii,j+jjj,k+kkk])
                                Z[loop_examples,loop_patch+patch_size] = alpha_sqrt * (S_examples[a,ii+iii,jj+jjj,kk+kkk]-S_estimated[i+iii,j+jjj,k+kkk])
                                '''##Compute distance for initialise LLE##'''
                                distance[loop_examples]+=Z[loop_examples,loop_patch]**2
                                distance[loop_examples]+=Z[loop_examples,loop_patch+patch_size]**2
                                #distance[loop_examples] += Z[loop_examples,loop_patch]**2 + Z[loop_examples,loop_patch+patch_size]**2
                                loop_patch += 1
                    #MeanPatch#
                    #S_order[loop_examples]=np.mean(S_examples[a,ii-hpsi:ii+hpsi+1,jj-hpsj:jj+hpsj+1,kk-hpsk:kk+hpsk+1])
                    #CentralPoint#
                    S_order[loop_examples]=S_examples[a,ii,jj,kk]

                    loop_examples+=1

    #Z[:,:patch_size] = (1-alpha)**0.5 * computeNormL1(I,I_examples,vx,hps,hss,num_examples,patch_size)
    #Z[:,patch_size:] = (alpha)**0.5 * computeNormL1(S_estimated,S_examples,vx,hps,hss,num_examples,patch_size)
    #Z[:,:patch_size] *= (1-alpha)**0.5
    #Z[:,patch_size:] *= (alpha)**0.5
    #distance = np.sum(Z[:,:patch_size]**2 + Z[:,patch_size:]**2,axis=1)

    #CentralPoint#
    #S_order = np.reshape(S_examples[:,vxi-hssi:vxi+hssi+1,vxj-hssj:vxj+hssj+1,vxk-hssk:vxk+hssk+1],(num_examples))
    #MeanPatch#
    #S_order = computeMeanPatch(S_examples,vx,hps,hss,num_examples)
    ##

    '''##Selection of KNN##'''
    KNN=sorted(np.asarray(sorted(range(len(distance)), key=distance.__getitem__)[:K]))
    distance=(distance)**-0.5


    #
    weights_ini=distance[KNN]
    Z_knn=Z[KNN,:]
    #weights_ini=np.zeros([K])
    #Z_knn=np.zeros([K,2*patch_size])
    #for ix,x in enumerate(KNN):
    #    weights_ini[ix]=distance[x]
    #    Z_knn[ix,:]=Z[x,:]
    #

    ##KNN from distance for initialise LLE##
    ##weights_ini/=np.max(weights_ini)


    '''##Compute LLE solution##'''
    '''##Define bounds of weights##'''
    bounds=((0.,1.),)*K
    #bounds=list()
    #for x in xrange(K):
    #    bounds.append( (0.,1.) )#np.finfo(float).eps
    ones=np.ones(K)
    C_knn=np.dot(Z_knn,np.transpose(Z_knn))
    '''##Add a identity matrix with a constant regularisation (reg)'''
    trace = np.trace(C_knn)
    R = reg * trace if trace>0 else reg
    C_knn.flat[::K + 1] += R
    '''##Compute LLE weights##'''
    weights = optimize.minimize(function_phi, weights_ini, args=(C_knn,ones), method='L-BFGS-B', bounds=bounds)['x']

    #weights = computeOptimalWeights(Z[KNN,:],distance[KNN],K)


    '''##Estimate segmentation with new weights##'''
    S_new=np.sum(np.dot(weights,S_order[KNN]))
    sum_w=np.sum(weights)


    #loop_examples=0
    #loop_K=0
    #sum_w=0.0
    #S_new = 0.0
    ###Loop on atlas x search zone##
    #for a in range(num_atlas):
    #    for ii in range(xmin,xmax):
    #        for jj in range(ymin,ymax):
    #            for kk in range(zmin,zmax):
    #                ##Select KNN##
    #                if loop_examples in KNN:
    #                    S_new += weights[loop_K]*S_examples[a,ii,jj,kk]
    #                    sum_w+=weights[loop_K]
    #                    loop_K+=1
    #                loop_examples+=1

    '''##Normalise segmentation value avoiding division by 0##'''
    if sum_w==0.0:
        sum_w=1.0

    return S_new/sum_w


@jit
def computeIMAPAT1Value(I, I_examples, I2, I2_examples, S_estimated, S_examples, vx, hps, hss, patch_size, num_examples, num_atlas, K, alpha1_sqrt, alpha_sqrt, reg=1e-03, epsilon=1e-06): #, mean_patch, var_patch, reg=1e-03, epsilon=1e-06

    ##Define the search zone##
    i=vx[0]
    j=vx[1]
    k=vx[2]
    hssi=hss[0]
    hssj=hss[1]
    hssk=hss[2]
    hpsi=hps[0]
    hpsj=hps[1]
    hpsk=hps[2]

    xmin = i-hssi
    ymin = j-hssj
    zmin = k-hssk
    xmax = i+hssi+1
    ymax = j+hssj+1
    zmax = k+hssk+1

    loop_examples=0
    S_order=np.zeros([num_examples])
    distance=epsilon*np.ones([num_examples])
    Z=np.zeros([num_examples,(2+1)*patch_size])
    '''##Loop on atlas##'''
    for a in range(num_atlas):
        '''##Loop on search zone##'''
        for ii in range(xmin,xmax):
            for jj in range(ymin,ymax):
                for kk in range(zmin,zmax):

                    '''##Loop on patch##'''
                    loop_patch = 0
                    for iii in range(-hpsi,hpsi+1):
                        for jjj in range(-hpsj,hpsj+1):
                            for kkk in range(-hpsk,hpsk+1):
                                '''##Compute LLE input##'''
                                Z[loop_examples,loop_patch] = alpha1_sqrt * (0.5**0.5) * ( (I_examples[a,ii+iii,jj+jjj,kk+kkk]-I[i+iii,j+jjj,k+kkk]) )
                                Z[loop_examples,loop_patch+patch_size] = alpha1_sqrt * (0.5**0.5) * ( (I2_examples[a,ii+iii,jj+jjj,kk+kkk]-I2[i+iii,j+jjj,k+kkk]) )
                                Z[loop_examples,loop_patch+2*patch_size] = alpha_sqrt * (S_examples[a,ii+iii,jj+jjj,kk+kkk]-S_estimated[i+iii,j+jjj,k+kkk])
                                '''##Compute distance for initialise LLE##'''
                                distance[loop_examples]+=Z[loop_examples,loop_patch]**2
                                distance[loop_examples]+=Z[loop_examples,loop_patch+patch_size]**2
                                distance[loop_examples]+=Z[loop_examples,loop_patch+2*patch_size]**2
                                #distance[loop_examples] += Z[loop_examples,loop_patch]**2 + Z[loop_examples,loop_patch+patch_size]**2
                                loop_patch += 1
                    #MeanPatch#
                    #S_order[loop_examples]=np.mean(S_examples[a,ii-hpsi:ii+hpsi+1,jj-hpsj:jj+hpsj+1,kk-hpsk:kk+hpsk+1])
                    #CentralPoint#
                    S_order[loop_examples]=S_examples[a,ii,jj,kk]

                    loop_examples+=1

    '''##Selection of KNN##'''
    KNN=sorted(np.asarray(sorted(range(len(distance)), key=distance.__getitem__)[:K]))
    distance=(distance)**-0.5

    weights_ini=distance[KNN]
    Z_knn=Z[KNN,:]

    ##KNN from distance for initialise LLE##

    '''##Compute LLE solution##'''
    '''##Define bounds of weights##'''
    bounds=((0.,1.),)*K
    ones=np.ones(K)
    C_knn=np.dot(Z_knn,np.transpose(Z_knn))
    '''##Add a identity matrix with a constant regularisation (reg)'''
    trace = np.trace(C_knn)
    R = reg * trace if trace>0 else reg
    C_knn.flat[::K + 1] += R
    '''##Compute LLE weights##'''
    weights = optimize.minimize(function_phi, weights_ini, args=(C_knn,ones), method='L-BFGS-B', bounds=bounds)['x']


    '''##Estimate segmentation with new weights##'''
    S_new=np.sum(np.dot(weights,S_order[KNN]))
    sum_w=np.sum(weights)


    '''##Normalise segmentation value avoiding division by 0##'''
    if sum_w==0.0:
        sum_w=1.0

    return S_new/sum_w



@jit
def computeIMAPAValue_SR(I, I_examples, S_estimated, S_examples, vx, hps, hss, patch_size, num_examples, num_atlas, K, alpha1_sqrt, alpha_sqrt, reg=1e-03, epsilon=1e-06): #, mean_patch, var_patch, reg=1e-03, epsilon=1e-06

    ##Define the search zone##
    i=vx[0]
    j=vx[1]
    k=vx[2]
    hssi=hss[0]
    hssj=hss[1]
    hssk=hss[2]
    hpsi=hps[0]
    hpsj=hps[1]
    hpsk=hps[2]

    xmin = i-hssi
    ymin = j-hssj
    zmin = k-hssk
    xmax = i+hssi+1
    ymax = j+hssj+1
    zmax = k+hssk+1

    #alpha1_sqrt=(1-alpha)**0.5
    #alpha_sqrt=(alpha)**0.5
    #patch_size = (2*hpsi+1)*(2*hpsj+1)*(2*hpsk+1)
    #num_examples = num_atlas*(2*hssi+1)*(2*hssj+1)*(2*hssk+1)

    loop_examples=0
    S_order=np.zeros([num_examples,3])
    distance=epsilon*np.ones([num_examples])
    Z=np.zeros([num_examples,2*patch_size])
    '''##Loop on atlas##'''
    for a in range(num_atlas):
        '''##Loop on search zone##'''
        for ii in range(xmin,xmax):
            for jj in range(ymin,ymax):
                for kk in range(zmin,zmax):

                    '''##Loop on patch##'''
                    loop_patch = 0
                    for iii in range(-hpsi,hpsi+1):
                        for jjj in range(-hpsj,hpsj+1):
                            for kkk in range(-hpsk,hpsk+1):
                                '''##Compute LLE input##'''
                                Z[loop_examples,loop_patch] = alpha1_sqrt * (I_examples[a,ii+iii,jj+jjj,kk+kkk]-I[i+iii,j+jjj,k+kkk])
                                Z[loop_examples,loop_patch+patch_size] = alpha_sqrt * (S_examples[a,ii+iii,jj+jjj,3*(kk+kkk)]-S_estimated[i+iii,j+jjj,3*(k+kkk)])
                                '''##Compute distance for initialise LLE##'''
                                distance[loop_examples]+=Z[loop_examples,loop_patch]**2
                                distance[loop_examples]+=Z[loop_examples,loop_patch+patch_size]**2
                                #distance[loop_examples] += Z[loop_examples,loop_patch]**2 + Z[loop_examples,loop_patch+patch_size]**2
                                loop_patch += 1
                    #MeanPatch#
                    #S_order[loop_examples]=np.mean(S_examples[a,ii-hpsi:ii+hpsi+1,jj-hpsj:jj+hpsj+1,kk-hpsk:kk+hpsk+1])
                    #CentralPoint#
                    S_order[loop_examples,:]=S_examples[a,ii,jj,3*kk-1:3*kk+2]

                    loop_examples+=1

    #Z[:,:patch_size] = (1-alpha)**0.5 * computeNormL1(I,I_examples,vx,hps,hss,num_examples,patch_size)
    #Z[:,patch_size:] = (alpha)**0.5 * computeNormL1(S_estimated,S_examples,vx,hps,hss,num_examples,patch_size)
    #Z[:,:patch_size] *= (1-alpha)**0.5
    #Z[:,patch_size:] *= (alpha)**0.5
    #distance = np.sum(Z[:,:patch_size]**2 + Z[:,patch_size:]**2,axis=1)

    #CentralPoint#
    #S_order = np.reshape(S_examples[:,vxi-hssi:vxi+hssi+1,vxj-hssj:vxj+hssj+1,vxk-hssk:vxk+hssk+1],(num_examples))
    #MeanPatch#
    #S_order = computeMeanPatch(S_examples,vx,hps,hss,num_examples)
    ##

    '''##Selection of KNN##'''
    KNN=sorted(np.asarray(sorted(range(len(distance)), key=distance.__getitem__)[:K]))
    distance=(distance)**-0.5


    #
    weights_ini=distance[KNN]
    Z_knn=Z[KNN,:]
    #weights_ini=np.zeros([K])
    #Z_knn=np.zeros([K,2*patch_size])
    #for ix,x in enumerate(KNN):
    #    weights_ini[ix]=distance[x]
    #    Z_knn[ix,:]=Z[x,:]
    #

    ##KNN from distance for initialise LLE##
    ##weights_ini/=np.max(weights_ini)


    '''##Compute LLE solution##'''
    '''##Define bounds of weights##'''
    bounds=((0.,1.),)*K
    #bounds=list()
    #for x in xrange(K):
    #    bounds.append( (0.,1.) )#np.finfo(float).eps
    ones=np.ones(K)
    C_knn=np.dot(Z_knn,np.transpose(Z_knn))
    '''##Add a identity matrix with a constant regularisation (reg)'''
    trace = np.trace(C_knn)
    R = reg * trace if trace>0 else reg
    C_knn.flat[::K + 1] += R
    '''##Compute LLE weights##'''
    weights = optimize.minimize(function_phi, weights_ini, args=(C_knn,ones), method='L-BFGS-B', bounds=bounds)['x']

    #weights = computeOptimalWeights(Z[KNN,:],distance[KNN],K)


    '''##Estimate segmentation with new weights##'''
    S_new=np.sum(np.dot(weights,S_order[KNN,:]),axis=-1)
    sum_w=np.sum(weights)

    '''##Normalise segmentation value avoiding division by 0##'''
    if sum_w==0.0:
        sum_w=1.0

    return S_new/sum_w









#@jit
def IMAPA_threads(args):
    (input,references,estimatedInputLabel,labels,points,hps,hss,K,alpha)=args

    ##Define parameters##
    patch_size=(2*hps[0]+1)*(2*hps[1]+1)*(2*hps[2]+1)
    num_atlas=references.shape[0]
    num_examples=(2*hss[0]+1)*(2*hss[1]+1)*(2*hss[2]+1)*(num_atlas)
    alpha1_sqrt=(1-alpha)**0.5
    alpha_sqrt=(alpha)**0.5
    #bounds=[(0.,1.) for x in xrange(K)]
    ####bounds=list()
    ####for x in xrange(K):
    ####    #bounds.append((np.finfo(float).eps,1.))
    ####    bounds.append((0.,1.))
    ##sqrt_alpha=alpha**0.5
    ##sqrt_1_alpha=(1-alpha)**0.5


    ##Process every voxel##
    S=np.zeros(input.shape)
    #for iv,vx in enumerate(points):
    for vx in points:
        #mean_patch=np.mean(input[vx[0]-hps[0]:vx[0]+hps[0]+1,vx[1]-hps[1]:vx[1]+hps[1]+1,vx[2]-hps[2]:vx[2]+hps[2]+1])
        #var_patch=np.var(input[vx[0]-hps[0]:vx[0]+hps[0]+1,vx[1]-hps[1]:vx[1]+hps[1]+1,vx[2]-hps[2]:vx[2]+hps[2]+1])
        #S[vx[0],vx[1],vx[2]]=computeITERValueAutoK(input,references,estimatedInputLabel,labels,vx[0],vx[1],vx[2],hps,hss,num_atlas,K,num_examples,patch_size,sqrt_alpha,sqrt_beta,mu1,sigma1)
        S[vx[0],vx[1],vx[2]]=computeIMAPAValue(input,references,estimatedInputLabel,labels,vx,hps,hss,patch_size,num_examples,num_atlas,K,alpha1_sqrt,alpha_sqrt)
        #S[vx[0],vx[1],vx[2]]=computeIMAPAValue(input,references,estimatedInputLabel,labels, vx[0],vx[1],vx[2], hps[0],hps[1],hps[2], hss[0],hss[1],hss[2], num_atlas,K,alpha)

    return S


def IMAPA_SR_threads(args):
    (input,references,estimatedInputLabel,labels,points,hps,hss,K,alpha,resolutionGain)=args

    ##Define parameters##
    patch_size=(2*hps[0]+1)*(2*hps[1]+1)*(2*hps[2]+1)
    num_atlas=references.shape[0]
    num_examples=(2*hss[0]+1)*(2*hss[1]+1)*(2*hss[2]+1)*(num_atlas)
    alpha1_sqrt=(1-alpha)**0.5
    alpha_sqrt=(alpha)**0.5
    #bounds=[(0.,1.) for x in xrange(K)]
    ####bounds=list()
    ####for x in xrange(K):
    ####    #bounds.append((np.finfo(float).eps,1.))
    ####    bounds.append((0.,1.))
    ##sqrt_alpha=alpha**0.5
    ##sqrt_1_alpha=(1-alpha)**0.5


    ##Process every voxel##
    S=np.zeros(estimatedInputLabel.shape)
    #for iv,vx in enumerate(points):
    for vx in points:
        #mean_patch=np.mean(input[vx[0]-hps[0]:vx[0]+hps[0]+1,vx[1]-hps[1]:vx[1]+hps[1]+1,vx[2]-hps[2]:vx[2]+hps[2]+1])
        #var_patch=np.var(input[vx[0]-hps[0]:vx[0]+hps[0]+1,vx[1]-hps[1]:vx[1]+hps[1]+1,vx[2]-hps[2]:vx[2]+hps[2]+1])
        #S[vx[0],vx[1],vx[2]]=computeITERValueAutoK(input,references,estimatedInputLabel,labels,vx[0],vx[1],vx[2],hps,hss,num_atlas,K,num_examples,patch_size,sqrt_alpha,sqrt_beta,mu1,sigma1)
        S[vx[0],vx[1],3*vx[2]-1:3*vx[2]+2]=computeIMAPAValue_SR(input,references,estimatedInputLabel,labels,vx,hps,hss,patch_size,num_examples,num_atlas,K,alpha1_sqrt,alpha_sqrt)
        #S[vx[0],vx[1],vx[2]]=computeIMAPAValue(input,references,estimatedInputLabel,labels, vx[0],vx[1],vx[2], hps[0],hps[1],hps[2], hss[0],hss[1],hss[2], num_atlas,K,alpha)

    return S



def IMAPA_t1_threads(args):
    (input,references,input2,references2,hatS,labels,points,hps,hss,K,alpha)=args

    ##Define parameters##
    patch_size=(2*hps[0]+1)*(2*hps[1]+1)*(2*hps[2]+1)
    num_atlas=references.shape[0]
    num_examples=(2*hss[0]+1)*(2*hss[1]+1)*(2*hss[2]+1)*(num_atlas)
    alpha1_sqrt=(1-alpha)**0.5
    alpha_sqrt=(alpha)**0.5

    ##Process every voxel##
    S=np.zeros(input.shape)
    for vx in points:
        S[vx[0],vx[1],vx[2]]=computeIMAPAT1Value(input,references,input2,references2,hatS,labels,vx,hps,hss,patch_size,num_examples,num_atlas,K,alpha1_sqrt,alpha_sqrt)

    return S
