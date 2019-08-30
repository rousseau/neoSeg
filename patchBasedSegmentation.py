#!/usr/bin/env python3
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

"""

import argparse,sys,os
import nibabel
import numpy as np
from scipy import ndimage,optimize
from time import time
import itertools
import multiprocessing
from numba import jit




#--Methods--#

def LP_threads(args):
    '''
    Execute Label Propagation (LP) method(*).
    (*) F. Rousseau, P. A. Habas, and C. Studholme, “A supervised patch-based approach for human
        brain labeling,” IEEE Transactions on Medical Imaging, vol. 30, pp. 1852–1862, Oct. 2011.
    -------------------------------
    Input:  points --> voxels to be processed
    Output: W      --> optimized weigths
            I_i    --> estimation of the intensity input image
            S_s    --> estimation of the label input map
    '''
    (points)=args
    I_i=np.zeros([input.shape[0],input.shape[1],input.shape[2]])
    I_s=np.zeros([input.shape[0],input.shape[1],input.shape[2]])
    W=np.zeros([len(points),num_atlasexamples])
    for iv,vx in enumerate(points):
        Pi=extractCentralPatch(input,vx,hps)
        Pa_j=extractExamples(atlas,vx,hss,hps)
        Pl_j=extractExamples(label,vx,hss,hps)
        W[iv,:]=computeLPKNNWeights(Pi,Pa_j)
        I_i[vx[0],vx[1],vx[2]]=np.sum([W[iv,j]*Pa_j[j,center] for j in range(num_atlasexamples)], axis=0)
        I_s[vx[0],vx[1],vx[2]]=np.sum([W[iv,j]*Pl_j[j,center] for j in range(num_atlasexamples)], axis=0)
    return W,I_i,I_s


def S_opt_threads(args):
    '''
    Execute Segmentation optimized (S_opt) method.
    -------------------------------
    Input:  points --> voxels to be processed
            Wini   --> initialization of weights
    Output: I_i    --> estimation of the intensity input image
            I_s    --> estimation of the label input map
    '''
    (points,Wini)=args
    I_i=np.zeros(input_size)
    I_s=np.zeros(input_size)
    for iv,vx in enumerate(points):
        Ps=extractCentralPatch(seg,vx,hps)
        Pa_j=extractExamples(atlas,vx,hss,hps)
        Pl_j=extractExamples(label,vx,hss,hps)
        Ws_opt=computeOptKNNWeights(Ps,Pl_j,Wini[iv,:])
        I_i[vx[0],vx[1],vx[2]]=np.sum([Ws_opt[j]*Pa_j[j,center] for j in range(num_atlasexamples)], axis=0)
        I_s[vx[0],vx[1],vx[2]]=np.sum([Ws_opt[j]*Pl_j[j,center] for j in range(num_atlasexamples)], axis=0)
    return I_i,I_s


def I_opt_threads(args):
    '''
    Execute Intensity optimized (I_opt) method.
    -------------------------------
    Input:  points --> voxels to be processed
            Wini   --> initialization of weights
    Output: I_i    --> estimation of the intensity input image
            I_s    --> estimation of the label input map
    '''
    (points,Wini)=args
    I_i=np.zeros(input_size)
    I_s=np.zeros(input_size)
    for iv,vx in enumerate(points):
        Pi=extractCentralPatch(input,vx,hps)
        Pa_j=extractExamples(atlas,vx,hss,hps)
        Pl_j=extractExamples(label,vx,hss,hps)
        Wi_opt=computeOptKNNWeights(Pi,Pa_j,Wini[iv,:])
        I_i[vx[0],vx[1],vx[2]]=np.sum([Wi_opt[j]*Pa_j[j,center] for j in range(num_atlasexamples)], axis=0)
        I_s[vx[0],vx[1],vx[2]]=np.sum([Wi_opt[j]*Pl_j[j,center] for j in range(num_atlasexamples)], axis=0)
    return I_i,I_s


def IS_opt_threads(args):
    '''
    Execute Intensity & Segmentation optimized (IS_opt) method.
    -------------------------------
    Input:  points --> voxels to be processed
            Wini   --> initialization of weights
    Output: Witer  --> optimized weights
            I_i    --> estimation of the intensity input image
            I_s    --> estimation of the label input map
    '''
    (points,Wini)=args
    I_i=np.zeros([input.shape[0],input.shape[1],input.shape[2]])
    I_s=np.zeros([input.shape[0],input.shape[1],input.shape[2]])
    Witer=np.zeros([len(points),num_atlasexamples])
    for iv,vx in enumerate(points):
        Pi=extractCentralPatch(input,vx,hps)
        Ps=extractCentralPatch(seg,vx,hps)
        Pa_j=extractExamples(atlas,vx,hss,hps)
        Pl_j=extractExamples(label,vx,hss,hps)
        Pis=np.concatenate((Pi,Ps))
        Pis_j=np.concatenate((Pa_j,Pl_j),axis=1)
        Witer[iv,:]=computeOptKNNWeights(Pis,Pis_j,Wini[iv,:])
        I_i[vx[0],vx[1],vx[2]]=np.sum([Witer[iv,j]*Pa_j[j,center] for j in range(num_atlasexamples)], axis=0)
        I_s[vx[0],vx[1],vx[2]]=np.sum([Witer[iv,j]*Pl_j[j,center] for j in range(num_atlasexamples)], axis=0)
    return Witer,I_i,I_s




def IMAPA_threads(points):
    '''
    Execute IMAPA method(*).
    (*) C. Tor-Díez, N. Passat, I. Bloch, S. Faisan, N. Bednarek and F. Rousseau,
        “An iterative multi-atlas patch-based approach for cortex segmentation from
        neonatal MRI,” Computerized Medical Imaging and Graphics, 70:73–82,
        2018, hal-01761063.
    -------------------------------
    Input:  points --> voxels to be processed
    Output: I      --> estimation of the intensity input image
            S      --> estimation of the label input map
    '''
    ##Process every voxel##
    I = np.zeros(input.shape)
    S = np.zeros(input.shape)
    for vx in points:
        distance, Z, I_order, L_order = computeIMAPADistance(\
            input[vx[0]-hps[0]:vx[0]+hps[0]+1,\
                    vx[1]-hps[1]:vx[1]+hps[1]+1,\
                    vx[2]-hps[2]:vx[2]+hps[2]+1],\
            atlas[:,vx[0]-hss[0]-hps[0]:vx[0]+hss[0]+hps[0]+1,\
                    vx[1]-hss[1]-hps[1]:vx[1]+hss[1]+hps[1]+1,\
                    vx[2]-hss[2]-hps[2]:vx[2]+hss[2]+hps[2]+1],\
            hatLabel[vx[0]-hps[0]:vx[0]+hps[0]+1,\
                    vx[1]-hps[1]:vx[1]+hps[1]+1,\
                    vx[2]-hps[2]:vx[2]+hps[2]+1],\
            label[:,vx[0]-hss[0]-hps[0]:vx[0]+hss[0]+hps[0]+1,\
                    vx[1]-hss[1]-hps[1]:vx[1]+hss[1]+hps[1]+1,\
                    vx[2]-hss[2]-hps[2]:vx[2]+hss[2]+hps[2]+1],\
            hps, hss, patch_size, num_examples, num_atlas, K, alpha1_sqrt, alpha_sqrt)

        I[vx[0],vx[1],vx[2]], S[vx[0],vx[1],vx[2]] = optimizeIMAPAWeights(distance, Z, I_order, L_order)

    return I, S



#--Tools--#

def extractCentralPatch(data,vx,hps):
    '''
    Extract a patch from a 3D image.
    -------------------------------
    Input:  data --> 3D image
            vx   --> central voxel of the patch
            hps  --> half patch size
    Output: P    --> central patch
    '''
    patch_size=(2*hps[0]+1)*(2*hps[1]+1)*(2*hps[2]+1)
    P=data[vx[0]-hps[0]:vx[0]+hps[0]+1,vx[1]-hps[1]:vx[1]+hps[1]+1,vx[2]-hps[2]:vx[2]+hps[2]+1].reshape([patch_size])
    return P


def extractExamples(data4D,vx,hss,hps):
    '''
    Extract patches from a stack 3D images.
    -------------------------------
    Input:  data4D --> stack of 3D images
            vx   --> central voxel of the patches
            hss  --> half search window size
            hps  --> half patch size
    Output: P_j    --> stack of central patches
    '''
    P_j=np.zeros([num_atlasexamples,patch_size])
    count=0
    for ind in range(num_atlas):
        data=data4D[ind,:,:,:]
        xmin=vx[0]-hss[0]
        ymin=vx[1]-hss[1]
        zmin=vx[2]-hss[2]
        xmax=vx[0]+hss[0]
        ymax=vx[1]+hss[1]
        zmax=vx[2]+hss[2]
        for ii in range(xmin,xmax+1):
            for jj in range(ymin,ymax+1):
                for kk in range(zmin,zmax+1):
                    P_j[count,:]=data[ii-hps[0]:ii+hps[0]+1,jj-hps[1]:jj+hps[1]+1,kk-hps[2]:kk+hps[2]+1].reshape([patch_size])
                    count+=1
    return P_j


@jit(nopython=True)
def computeIMAPADistance(I, I_examples, L, L_examples, hps, hss, patch_size, num_examples, num_atlas, K, alpha1_sqrt, alpha_sqrt, epsilon=1e-06):
    '''
    First step of the IMAPA method for computing the weights:
    it computes the distance between patches.
    -------------------------------
    Input:  I           --> input image to be labeled
            I_examples  --> intensity images from the atlas set
            L           --> estimation of input label map
            L_examples  --> label images from the atlas set
            hps         --> half patch size
            hss         --> half search window size
            patch_size  --> number of voxels in a patch
            num_examples--> number of patches in the atlas set
            num_atlas   --> number of images in the atlas set
            K           --> k-Nearest Neighbors (kNN)
            alpha1_sqrt --> square root of (1-alpha)
            alpha_sqrt  --> square root of (alpha)
            epsilon     --> small term for inversion purpose
    Output: distance    --> distance L2 between input patch and example patches
            Z           --> distance L1 between input patch and example patches
            I_order     --> sequence of intensity examples
            L_order     --> sequence of label examples
    '''
    loop_examples=0
    I_order=np.zeros(num_examples)
    L_order=np.zeros(num_examples)
    distance=epsilon*np.ones(num_examples)
    Z=np.zeros((num_examples,2*patch_size))
    '''##Loop on atlas##'''
    for a in range(num_atlas):
        '''##Loop on search zone##'''
        for ii in range(2*hss[0]+1):
            for jj in range(2*hss[1]+1):
                for kk in range(2*hss[2]+1):

                    '''##Loop on patch##'''
                    loop_patch = 0
                    for iii in range(2*hps[0]+1):
                        for jjj in range(2*hps[1]+1):
                            for kkk in range(2*hps[2]+1):
                                '''##Compute LLE input##'''
                                Z[loop_examples,loop_patch] = alpha1_sqrt * (I_examples[a,ii+iii,jj+jjj,kk+kkk]-I[iii,jjj,kkk])
                                Z[loop_examples,loop_patch+patch_size] = alpha_sqrt * (L_examples[a,ii+iii,jj+jjj,kk+kkk]-L[iii,jjj,kkk])
                                '''##Compute distance for initialise LLE##'''
                                distance[loop_examples] += Z[loop_examples,loop_patch]**2 + Z[loop_examples,loop_patch+patch_size]**2
                                loop_patch += 1
                    I_order[loop_examples] = I_examples[a,ii+hps[0],jj+hps[1],kk+hps[2]]
                    L_order[loop_examples] = L_examples[a,ii+hps[0],jj+hps[1],kk+hps[2]]

                    loop_examples += 1

    return distance, Z, I_order, L_order


def optimizeIMAPAWeights(distance, Z, I_order,L_order, reg=1e-03):
    '''
    Second step of the IMAPA method for computing the weights:
    it optimize the weights given distances between patches.
    -------------------------------
    Input:  distance  --> distance L2 between input patch and example patches
            Z         --> distance L1 between input patch and example patches
            I_order   --> sequence of intensity examples
            L_order   --> sequence of label examples
            reg       --> regularization term for ensuring a non-invertible covariance matrix
    Output: I_new     --> estimation of the intensity input image
            L_new     --> estimation of the label input map
    '''
    '''##Selection of KNN##'''
    KNN = sorted(np.asarray(sorted(range(len(distance)), key=distance.__getitem__)[:K]))
    distance = (distance)**-0.5

    '''##Compute LLE solution##'''
    '''##Define bounds of weights##'''
    C_knn = np.dot(Z[KNN,:],np.transpose(Z[KNN,:]))
    '''##Add a identity matrix with a constant regularisation (reg)'''
    R = reg * np.trace(C_knn) if np.trace(C_knn)>0 else reg
    C_knn.flat[::K + 1] += R
    '''##Compute LLE weights##'''
    weights = optimize.minimize(function_phi, distance[KNN], args=(C_knn,np.ones(K)), method='L-BFGS-B', bounds=((0.,1.),)*K)['x']

    '''##Estimate outputs with new weights##'''
    I_new = np.sum(np.dot(weights,I_order[KNN]))/np.sum(weights)
    L_new = np.sum(np.dot(weights,L_order[KNN]))/np.sum(weights)

    return I_new, L_new


def computeOptKNNWeights(Pw,Pw_j,initialization,reg=1e-03):
    '''
    Computating the k weights of the kNN atlas patches by following the
    Locally Linear Embedding (LLE) algorithm optimization(*):

        weights_{opt}  =  argmin_w  \| Pw - \sum_j( w * Pw_j ) \|

    (*) S. T. Roweis and L. K. Saul, “Nonlinear dimensionality reduction by locally linear
        embedding,” Science, vol. 290, pp. 2323–2326, Dec. 2000.
    -------------------------------
    Input:  Pw             --> input patch
            Pw_j           --> stack of patches from the atlas set
            initialization --> initialization for the opitmization weights
            reg            --> regularization term for ensuring a non-invertible covariance matrix
    Output: weights        --> optimized k weights for the kNN patches
    '''
    Z=np.zeros([K,Pw_j.shape[1]])
    dist_j=np.array([np.linalg.norm(Pw-Pw_j[j,:]) for j in range(num_atlasexamples)]) #distance between P and P_j
    order=np.asarray(sorted(range(len(dist_j)), key=lambda lam: dist_j[lam])[:K])#les plus petits
    b=np.ones(len(order))
    w=np.array([initialization[o] for o in order])
    Z=(Pw_j[order,:]-Pw)
    C=np.dot(Z,np.transpose(Z))
    trace = np.trace(C)
    R = reg * trace if trace>0 else reg
    C.flat[::num_atlasexamples + 1] += R
    fmin = lambda x: np.linalg.norm(np.dot(C,x)-b)
    sol = optimize.minimize(fmin, w, method='L-BFGS-B', bounds=[(0.,1.) for x in range(len(order))])#, constraints=cons)
    w = sol['x']
    weights=np.zeros([num_atlasexamples])
    if w.sum()!=0:
        weights[order]=w/np.sum(w)
    return weights


def computeLPKNNWeights(Pw,Pw_j):
    '''
    Computating the k weights of the kNN atlas patches by following the Label Propagation method(*).
    (*) F. Rousseau, P. A. Habas, and C. Studholme, “A supervised patch-based approach for human
        brain labeling,” IEEE Transactions on Medical Imaging, vol. 30, pp. 1852–1862, Oct. 2011.
    -------------------------------
    Input:  Pw       --> input patch
            Pw_j     --> stack of patches from the atlas set
    Output: weights  --> optimized k weights for the kNN patches
    '''
    dist_j=np.array([np.sum((Pw-Pw_j[j,:])**2) for j in range(num_atlasexamples)]) #distance between Pw and Pw_j
    order=sorted(range(len(dist_j)), key=lambda lam: dist_j[lam])[:K]#les plus petits
    w=np.exp(-(dist_j[order]/h))
    weights=np.zeros([num_atlasexamples])
    if np.sum(w)!=0:
        weights[order]=w/np.sum(w)
    return weights


def computeSigmaNoise(img):
    '''
    Computating the variation of the image noise by following an optimization of NLM for MR images(*).
    (*) P. Coupé, P. Yger, S. Prima, P. Hellier, C. Kervrann, and C. Barillot, “An
        Optimized Blockwise Nonlocal Means Denoising Filter for 3-D Magnetic Resonance
        Images,” IEEE Transactions on Medical Imaging, vol. 27, pp. 425–441, Apr. 2008.
    -------------------------------
    Input:  img     --> 3D image
    Output: sigma2  --> variation of image noise
    '''
    hx=np.array([[[0.,0.,0.],[0.,-(1./6.),0.],[0.,0.,0.]],
                    [[0.,-(1./6.),0.],[-(1./6.),1.,-(1./6.)],[0.,-(1./6.),0.]],
                    [[0.,0.,0.],[0.,-(1./6.),0.],[0.,0.,0.]]])

    sigma2=(6.0/7.0) * np.sum(np.square(ndimage.convolve(img,hx))) / np.float(img.shape[0]*img.shape[1]*img.shape[2])
    return sigma2



def function_phi(x,C,b):
    '''
    Linear system to be solved:
        C * x = b
    -------------------------------
    Input:  x --> weights
            C --> covariance matrix
            b --> array of all ones
    Output: norm 2 of the product of C and x minus b
    '''
    return np.linalg.norm(np.dot(C,x)-b)



def chunkIt(seq, num):
    '''
    Split a sequence in blocks of sequences.
    -------------------------------
    Input:  seq --> sequence to be splitted
            num --> number of blocks
    Output: out --> sequence containing blocks of sequences
    '''
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out





if __name__ == '__main__':
    '''
    Python script consecrated to patch-based image segmentation. The methods
    used are the Label Propagation (LP) algorithm (inspirated in the classical
    Non-Local Means), an optimization regarding the real intensity (S_opt),
    input image and an optimization regarding the real segmentation image
    (I_opt), an optimization regarding both real intensity (IS_opt) and
    segmentation and, a new one, an iterative optimization (IMAPA).


    Thus, parameter --method can be multiple and adopt the next set of values:
          -LP
          -S_opt
          -I_opt
          -IS_opt
          -IMAPA


    Example of utilisation in your terminal:

        python neoSeg/patchBasedSegmentation.py  -i  brain.nii.gz
            -a  atlas1_registered_HM.nii.gz  atlas2_registered_HM.nii.gz
            -l  label1_propagated.nii.gz  label2_propagated.nii.gz
            -mask mask.nii.gz  -m IMAPA  -hss 3  -hps 1  -k 15  -alphas 0 0.25  -t 4

        Note:   We recommend to previously register the intensity image from the atlas set
                to the input image, apply a histogram matching algorithm and propagate
                the transformations to the label maps.
    '''

    t0=time()
    np.seterr(divide='ignore', invalid='ignore')
    parser = argparse.ArgumentParser(prog='patchAnalysis')

    parser.add_argument('-i', '--input', help='Input anatomical image (required)', type=str, required=True)
    parser.add_argument('-s', '--seg', help='Input segmentation image (ground truth)', type=str, required=False, default='')
    parser.add_argument('-o', '--output', help='Output name', type=str, default='output', required=False)
    parser.add_argument('-a', '--atlas', help='Anatomical atlas images in the input space (required)', type=str, nargs='*', required=True)
    parser.add_argument('-l', '--label', help='Label atlas images in the input space (required)', type=str, nargs='*', required=True)
    parser.add_argument('-m', '--method', help='Segmentation method chosen (LP, S_opt, I_opt, IS_opt or IMAPA)', type=str, nargs='*', default=['IMAPA'], required=False)
    parser.add_argument('-mask', '--mask', help='Binary image for input', type=str, required=False)
    parser.add_argument('-hss', '--hss', help='Half search window input_size', type=int, default=1, required=False)
    parser.add_argument('-hps', '--hps', help='Half patch input_size', type=int, default=1, required=False)
    parser.add_argument('-k', '--k', help='k-Nearest Neighbors (kNN)', type=int, default=20, required=False)
    parser.add_argument('-alphas', '--alphas', help='Alphas parameter for IS_opt and IMAPA methods. The number of values determines \
                                                    the iterations. Example: [0.0, 0.25] -> 2 iterations, first with alpha = 0.0 and \
                                                    alpha = 0.25.', type=float, nargs='*', default=[0.0, 0.25], required=False)
    parser.add_argument('-t', '--threads', help='Number of threads (0 for the maximum number of cores available)', type=int, default=4, required=False)


    args = parser.parse_args()

    if len(args.atlas)!=len(args.label):
        print('Number of atlas and label images have to be equal.')
        sys.exit()

    ###--Load input data--###

    try:
        input=np.float32(nibabel.load(args.input).get_data())
        header=nibabel.load(args.input).affine
    except:
        print('Input anatomical image file not found.')
        sys.exit()
    if args.seg != '':
        try:
            seg=np.float32(nibabel.load(args.seg).get_data())
        except:
            print('Input segmentation image file not found.')
            sys.exit()

    hss=np.array([args.hss,args.hss,args.hss])
    hps=np.array([args.hps,args.hps,args.hps])
    beta=1.0
    h=np.float(2.0 * beta * (computeSigmaNoise(input)) * np.float((2*hps[0]+1)*(2*hps[1]+1)*(2*hps[2]+1)) )
    patch_size=(2*hps[0]+1)*(2*hps[1]+1)*(2*hps[2]+1)

    num_atlas=len(args.atlas)
    num_atlasexamples=(2*hss[0]+1)*(2*hss[1]+1)*(2*hss[2]+1)*(num_atlas)
    try:
        atlas=np.array([nibabel.load(a).get_data() for a in args.atlas])
        label=np.array([nibabel.load(a).get_data() for a in args.label])
    except:
        print('An anatomical or segmentation atlas file not found.')
        sys.exit()
    #Check if the mask is given by the user, if not it built one from atlas segmentations#
    if args.mask!=None:
        try:
            mask=np.float32(nibabel.load(args.mask).get_data())
        except:
            print('Mask image file not found.')
            sys.exit()
    elif args.mask=='atlas':
        mask=np.zeros(input.shape)
        for a in range(len(atlas)):
            mask[label[a,:,:,:]>(np.max(label)*0.10)]=1
    else:
        mask=np.ones(input.shape)
    K=args.k if hss[0]!=0 else num_atlas
    #Correct repetitions or outliers from methods introduced by the user#
    available_method=['LP','S_opt','I_opt','IS_opt','IMAPA']
    method=[m for m in set(args.method) if m not in list(set(args.method)-set(available_method))]
    method=set(args.method)
    #Assign a dictionary for the outputs#
    output={}
    for m in method:
        if m=='LP':
            output['LP']=args.output+'_LP.nii.gz'
        elif m=='S_opt':
            if args.seg==None:
                print('Input segmentation image (ground truth) is required for this method.')
                sys.exit()
            output['S_opt']=args.output+'_S_opt.nii.gz'
        elif m=='I_opt':
            output['I_opt']=args.output+'_I_opt.nii.gz'
        elif m=='IS_opt':
            if args.seg==None:
                print('Input segmentation image (ground truth) is required for this method.')
                sys.exit()
            if args.alphas==None:
                print('Alpha parameter is required for this method.')
                sys.exit()
            output['IS_opt']=args.output+'_IS_opt'
        elif m=='IMAPA':
            if args.alphas==None:
                print('Alpha parameter is required for this method.')
                sys.exit()
            output['IMAPA']=args.output+'_IMAPA'
        else:
            print('The method ',m,' is not found. Please use one of the available methods (LP, S_opt, I_opt, IS_opt or IMAPA).')
            sys.exit()
    alphas=args.alphas
    #Compare the number of cores available against the user parameter#
    threads=multiprocessing.cpu_count()
    if (args.threads!=0) & (args.threads<threads):
        threads=args.threads


    print(args.method)



    ###--Print Input Information--###

    print('Input anatomical image: ',args.input)
    print('Input segmentation image (probability map): ',args.seg)
    print('Anatomical exemple images: ',args.atlas)
    print('Label exemple images: ',args.label)
    print('Methods selected: ',method)

    print('Points to be processed: ',str(len(np.where(mask!=0))))

    print('Half search window size: ',str(args.hss),' voxels')
    print('Half patch size: ',str(args.hps),' voxels')
    print('K: ',str(K))
    if 'IMAPA' in method:
        print('Alpha parameter: ',str(alphas))
    print('Number of threads: ',str(threads))


    #--Normalisation--#
    maxinput = np.max(input)
    maxlabel = np.max(label)
    input /= maxinput
    atlas /= maxinput
    label /= maxlabel

    padding=[(2*(hps[0]+hss[0]),2*(hps[0]+hss[0])),(2*(hps[1]+hss[1]),2*(hps[1]+hss[1])),(2*(hps[2]+hss[2]),2*(hps[2]+hss[2]))]

    input = np.pad(input,padding,mode='constant', constant_values=0)
    atlas = np.array([np.pad(a,padding,mode='constant', constant_values=0) for a in atlas])
    label = np.array([np.pad(a,padding,mode='constant', constant_values=0) for a in label])
    mask = np.pad(mask,padding,mode='constant', constant_values=0)

    ###--Compute analysis--###

    center = hps[0] + (2*hps[1]+1) * ( hps[1] + (2*hps[2]+1) * hps[2] )

    eps=np.finfo(float).eps
    input_size=input.shape

    ptx,pty,ptz = np.where(mask==1)
    points=[np.array([ptx[pti],pty[pti],ptz[pti]]) for pti in range(len(ptx))]

    num_partitions=threads
    pointSplit=chunkIt(points,num_partitions)




    #--Compute LP--#
    if 'LP' in method:
        pool=multiprocessing.Pool(threads)
        tmp=pool.map(LP_threads, pointSplit)
        pool.close()
        pool.join()

        Inlm_i=np.zeros(input_size)
        Inlm_s=np.zeros(input_size)
        Wnlm=np.zeros((len(points),num_atlasexamples))
        count=0
        for part in range(num_partitions):
            Wnlm[count:count+len(pointSplit[part]),:]=tmp[part][0]
            Inlm_i+=tmp[part][1]
            Inlm_s+=tmp[part][2]
            count+=len(pointSplit[part])

        try:
            nibabel.save(nibabel.Nifti1Image(Inlm_i, header),output['LP']+'_I.nii.gz')
            nibabel.save(nibabel.Nifti1Image(Inlm_s, header),output['LP']+'_S.nii.gz')
        except:
            print('Error saving IMAPA result')
            sys.exit()


    #--Compute S_opt--#
    if 'S_opt' in method:
        pool=multiprocessing.Pool(threads)
        tmp=pool.map(S_opt_threads, itertools.izip(pointSplit,chunkIt(Wnlm,num_partitions)))
        pool.close()
        pool.join()

        Is_opt_i=np.zeros([input.shape[0],input.shape[1],input.shape[2]])
        Is_opt_s=np.zeros([input.shape[0],input.shape[1],input.shape[2]])
        for part in range(num_partitions):
            Is_opt_i+=tmp[part][0]
            Is_opt_s+=tmp[part][1]
        try:
            nibabel.save(nibabel.Nifti1Image(Is_opt_i, header),output['S_opt']+'_I.nii.gz')
            nibabel.save(nibabel.Nifti1Image(Is_opt_s, header),output['S_opt']+'_S.nii.gz')
        except:
            print('Error saving S_opt result')
            sys.exit()

    #--Compute I_opt--#
    if 'I_opt' in method:
        pool=multiprocessing.Pool(threads)
        tmp=pool.map(I_opt_threads, itertools.izip(pointSplit,chunkIt(Wnlm,num_partitions)))
        pool.close()
        pool.join()

        Ii_opt_i=np.zeros([input.shape[0],input.shape[1],input.shape[2]])
        Ii_opt_s=np.zeros([input.shape[0],input.shape[1],input.shape[2]])
        for part in range(num_partitions):
            Ii_opt_i+=tmp[part][0]
            Ii_opt_s+=tmp[part][1]
        try:
            nibabel.save(nibabel.Nifti1Image(Ii_opt_i, header),output['I_opt']+'_I.nii.gz')
            nibabel.save(nibabel.Nifti1Image(Ii_opt_s, header),output['I_opt']+'_S.nii.gz')
        except:
            print('Error saving I_opt result')
            sys.exit()


    #--Compute IS_opt--#
    if 'IS_opt' in method:
        Wis_opt=Wnlm

        for alpha in alphas:
            W_prev=chunkIt(Wis_opt,num_partitions)
            Iis_opt_s=None
            Iis_opt_i=None
            pool=multiprocessing.Pool(threads)
            tmp=pool.map(IS_opt_threads,itertools.izip(pointSplit,W_prev))
            pool.close()
            pool.join()

            Iis_opt_i=np.zeros([input.shape[0],input.shape[1],input.shape[2]])
            Iis_opt_s=np.zeros([input.shape[0],input.shape[1],input.shape[2]])
            Wis_opt=np.zeros((len(points),num_atlasexamples))
            count=0
            for part in range(num_partitions):
                Wis_opt[count:count+len(pointSplit[part]),:]=tmp[part][0]
                Iis_opt_i+=tmp[part][1]
                Iis_opt_s+=tmp[part][2]
                count+=len(pointSplit[part])
            try:
                nibabel.save(nibabel.Nifti1Image(Iis_opt_i, header),output['IS_opt']+'_I_'+str(alpha)+'.nii.gz')
                nibabel.save(nibabel.Nifti1Image(Iis_opt_s, header),output['IS_opt']+'_S_'+str(alpha)+'.nii.gz')
            except:
                print('Error saving IS_opt result')
                sys.exit()


    #--Compute IMAPA--#
    if 'IMAPA' in method:

        '''##Global Parameters##'''
        hatLabel = np.zeros(input.shape)
        patch_size=(2*hps[0]+1)*(2*hps[1]+1)*(2*hps[2]+1)
        num_examples=(2*hss[0]+1)*(2*hss[1]+1)*(2*hss[2]+1)*(num_atlas)

        for alpha in alphas:
            print('Iteration with alpha = '+str(alpha))
            alpha1_sqrt=(1-alpha)**0.5
            alpha_sqrt=(alpha)**0.5
            Iimapa_i=None
            Iimapa_s=None
            pool=multiprocessing.Pool(threads)
            tmp=pool.map(IMAPA_threads,pointSplit)
            pool.close()
            pool.join()
            Iimapa_i=np.zeros([input.shape[0],input.shape[1],input.shape[2]])
            Iimapa_s=np.zeros([input.shape[0],input.shape[1],input.shape[2]])
            for part in range(num_partitions):
                Iimapa_i+=tmp[part][0]
                Iimapa_s+=tmp[part][1]
            '''##Delete padding from result##'''
            Iimapa_i = Iimapa_i[padding[0][0]:-padding[0][1],padding[1][0]:-padding[1][1],padding[2][0]:-padding[2][1]]
            Iimapa_s = Iimapa_s[padding[0][0]:-padding[0][1],padding[1][0]:-padding[1][1],padding[2][0]:-padding[2][1]]
            try:
                nibabel.save(nibabel.Nifti1Image(Iimapa_i, header),output['IMAPA']+'_I_'+str(alpha)+'.nii.gz')
                nibabel.save(nibabel.Nifti1Image(Iimapa_s, header),output['IMAPA']+'_S_'+str(alpha)+'.nii.gz')
            except:
                print('Error saving IMAPA result')
                sys.exit()



    print('Script ran for '+str(np.round(time()-t0,2))+' s.')
