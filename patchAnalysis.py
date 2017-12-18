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
  Python script for analyse different kinds of patch weighting computation
  for segmentation estimation. The methods used are the classical Non-Local
  Means (NLM) algorithm, an optimization regarding the real intensity
  input image and an optimization regarding the real segmentation image.

  The examples used for extracting the neighbourhood patches of atlases
  (optional) or the same intensity image (if atlases are not providied).
  An additional method is used for atlas mode. It iteratively computes the
  weights using both intensity and segmentation images. Segmentation images
  is estimates at each iteration.

  It returns 5 plots:
    - Comparison of the real patch and the estimation obtained
    - Comparison between different kind of weighting
    - Comparison between patch correlation and weighting error
     in intensity image
    - Comparison between patch correlation and weighting error
     in segmentation image
    - Rate between patch centers distance and patchs without centers
     distance


  Example:
  python patchAnalysis.py -i ~/Data/ALBERT_makropoulos/T2/ALBERT_01.nii.gz
        -s ~/Data/ALBERT_makropoulos/gm-posteriors-v3/ALBERT_01.nii.gz
        -x 10 -y 76 -z 65
  #####################################################################

"""

import argparse,sys
import nibabel
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy import ndimage


def computeOptWeights(P,P_j,initialisation,reg=1e-03):
    Pw=P.copy()
    Pw_j=P_j.copy()
    w=initialisation.copy()
    num_examples=Pw_j.shape[0]
    weights=np.zeros(num_examples)
    Z=np.zeros(Pw_j.shape)
    b=np.ones(num_examples)
    for j in range(num_examples):
        Z[j,:]=Pw_j[j,:]-Pw
    C=np.dot(Z,np.transpose(Z))
    trace = np.trace(C)
    R = reg * trace if trace>0 else reg
    C.flat[::num_examples + 1] += R
    fmin = lambda x: np.linalg.norm(np.dot(C,x)-b)
    sol = optimize.minimize(fmin, w, method='L-BFGS-B', bounds=[(0.,1.) for x in xrange(num_examples)])
    w = sol['x']
    weights=w/np.sum(w)
    return weights


def computeNLMWeights(P,P_j,h):
    Pw=P.copy()
    Pw_j=P_j.copy()
    num_examples=Pw_j.shape[0]
    w=np.zeros([num_examples])
    dist_j=np.array([np.sum((Pw-Pw_j[j,:])**2) for j in range(num_examples)]) #distance between Pw and Pw_j
    w=np.exp(-(dist_j/h))
    weights=w/np.sum(w)
    return weights


def computeSigmaNoise(img):
    hx=np.array([[[0.,0.,0.],[0.,-(1./6.),0.],[0.,0.,0.]],
                [[0.,-(1./6.),0.],[-(1./6.),1.,-(1./6.)],[0.,-(1./6.),0.]],
                [[0.,0.,0.],[0.,-(1./6.),0.],[0.,0.,0.]]])
    sigma2=(6.0/7.0) * np.sum(np.square(ndimage.convolve(img,hx))) / np.float(img.shape[0]*img.shape[1]*img.shape[2])
    return sigma2


def extractPatches4D(data4D,i):
    P_j=np.zeros([num_atlasexamples,patch_size])
    count=0
    for ind in xrange(num_atlas):
        data=data4D[ind,:,:,:]
        xmin = i[0]-hss[0]
        if xmin < hps[0]:
            xmin = hps[0]
        ymin = i[1]-hss[1]
        if ymin < hps[1]:
            ymin = hps[1]
        zmin = i[2]-hss[2]
        if zmin < hps[2]:
            zmin = hps[2]
        xmax = i[0]+hss[0]
        if xmax> data.shape[0]-hps[0]:
            xmax = data.shape[0]-hps[0]
        ymax = i[1]+hss[1]
        if ymax> data.shape[1]-hps[1]:
            ymax = data.shape[1]-hps[1]
        zmax = i[2]+hss[2]
        if zmax> data.shape[2]-hps[2]:
            zmax = data.shape[2]-hps[2]

        for ii in xrange(xmin,xmax+1):
            for jj in xrange(ymin,ymax+1):
                for kk in xrange(zmin,zmax+1):
                    P_j[count,:]=data[ii-hps[0]:ii+hps[0]+1,jj-hps[1]:jj+hps[1]+1,kk-hps[2]:kk+hps[2]+1].reshape([patch_size])
                    count+=1
    return P_j


def extractPatches(data,i):
    xmin = i[0]-hss[0]
    if xmin < hps[0]:
        xmin = hps[0]
    ymin = i[1]-hss[1]
    if ymin < hps[1]:
        ymin = hps[1]
    zmin = i[2]-hss[2]
    if zmin < hps[2]:
        zmin = hps[2]
    xmax = i[0]+hss[0]
    if xmax> data.shape[0]-hps[0]:
        xmax = data.shape[0]-hps[0]
    ymax = i[1]+hss[1]
    if ymax> data.shape[1]-hps[1]:
        ymax = data.shape[1]-hps[1]
    zmax = i[2]+hss[2]
    if zmax> data.shape[2]-hps[2]:
        zmax = data.shape[2]-hps[2]

    count=0
    size_patch=(2*hps[0]+1)*(2*hps[1]+1)*(2*hps[2]+1)
    P_j=np.zeros([(2*hss[0]+1)*(2*hss[1]+1)*(2*hss[2]+1)-1,size_patch])
    for ii in range(xmin,xmax+1):
        for jj in range(ymin,ymax+1):
            for kk in range(zmin,zmax+1):
                if (ii==i[0]) & (jj==i[1]) & (kk==i[2]):
                    P=data[ii-hps[0]:ii+hps[0]+1,jj-hps[1]:jj+hps[1]+1,kk-hps[2]:kk+hps[2]+1].reshape([size_patch])
                else:
                    P_j[count,:]=data[ii-hps[0]:ii+hps[0]+1,jj-hps[1]:jj+hps[1]+1,kk-hps[2]:kk+hps[2]+1].reshape([size_patch])
                    count+=1
    return P,P_j


def computeOptWeightsAtlas(Pw,Pw_j,initialisation,reg=1e-03):
    w=initialisation
    weights=np.zeros(num_atlasexamples)
    Z=np.zeros(Pw_j.shape)
    b=np.ones(num_atlasexamples)
    for j in xrange(num_atlasexamples):
        Z[j,:]=Pw_j[j,:]-Pw
    C=np.dot(Z,np.transpose(Z))
    trace = np.trace(C)
    R = reg * trace if trace>0 else reg
    C.flat[::num_atlasexamples + 1] += R
    fmin = lambda x: np.linalg.norm(np.dot(C,x)-b)
    sol = optimize.minimize(fmin, w, method='L-BFGS-B', bounds=[(0.,1.) for x in xrange(num_atlasexamples)])
    w = sol['x']
    weights=w/np.sum(w)
    return weights






if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='patchAnalysis')

    parser.add_argument('-i', '--input', help='Input anatomical image (required)', type=str, required = True)
    parser.add_argument('-s', '--seg', help='Input segmentation image (probability map, required)', type=str, required = True)
    parser.add_argument('-a', '--atlas', help='Anatomical atlas images', type=str, nargs='*', required = False)
    parser.add_argument('-l', '--label', help='Label atlas images', type=str, nargs='*', required = False)
    parser.add_argument('-iter', '--iterations', help='Number of iterations for optimal weighting computation', type=int, default=1, required = False)
    parser.add_argument('-x', '--x', help='x location (required)', type=int, required = True)
    parser.add_argument('-y', '--y', help='y location (required)', type=int, required = True)
    parser.add_argument('-z', '--z', help='z location (required)', type=int, required = True)
    parser.add_argument('-hss', '--hss', help='Half search area size (default)', type=int, default=4, required = False)
    parser.add_argument('-hps', '--hps', help='Half path size', type=int, default=1, required = False)


    args = parser.parse_args()
    np.seterr(divide='ignore', invalid='ignore')

    if args.atlas!=None:
        if len(args.atlas)!=len(args.label):
            print 'Number of atlas and label images have to be equal.'
            sys.exit()


    ###--Print Input Information--###

    print 'Input anatomical image: ',args.input
    print 'Input segmentation image (probability map): ',args.seg
    if args.atlas!=None:
        print 'Anatomical exemple images: ',args.atlas
        print 'Label exemple images: ',args.label
        print 'Number of iterations: ',str(args.iterations)
    print 'Location: (',str(args.x),',',str(args.y),',',str(args.z),')'
    print 'Half search area size: ',str(args.hss)
    print 'Half patch size: ',str(args.hps)


    ###--Load input data--###

    input=np.float32(nibabel.load(args.input).get_data())
    seg=np.float32(nibabel.load(args.seg).get_data())
    vx=[args.x,args.y,args.z]
    hss=[args.hss,args.hss,args.hss]
    hps=[args.hps,args.hps,args.hps]
    beta=1.0
    h=np.float(2.0 * beta * (computeSigmaNoise(input)) * np.float((2*hps[0]+1)*(2*hps[1]+1)*(2*hps[2]+1)) )
    patch_size=(2*hps[0]+1)*(2*hps[1]+1)*(2*hps[2]+1)
    if args.atlas!=None:
        num_atlas=len(args.atlas)
        num_atlasexamples=(2*hss[0]+1)*(2*hss[1]+1)*(2*hss[2]+1)*(num_atlas)
        atlas=np.array([nibabel.load(a).get_data() for a in args.atlas])
        label=np.array([nibabel.load(a).get_data() for a in args.label])
        iterations=args.iterations

    ###--Compute analysis--###

    #--Extract patches--#
    Pi,Pi_j=extractPatches(input,vx)
    Ps,Ps_j=extractPatches(seg,vx)
    if args.atlas!=None:
        Pa_j=extractPatches4D(atlas,vx)
        Pl_j=extractPatches4D(label,vx)


    #--Compute weights--#
    Wi_nlm=computeNLMWeights(Pi,Pi_j,h)
    Ws_opt=computeOptWeights(Ps,Ps_j,Wi_nlm)
    Wi_opt=computeOptWeights(Pi,Pi_j,Wi_nlm)
    if args.atlas!=None:
        Wa_nlm=computeNLMWeights(Pi,Pa_j,h)
        Wis_opt=Wa_nlm
        Wis_iter=Wa_nlm
        hatPs=np.sum([Wis_iter[j]*Pl_j[j,:] for j in xrange(num_atlasexamples)], axis=0)
        Pis_iter=np.concatenate((Pi,hatPs))
        Pis_j=np.concatenate((Pa_j,Pl_j),axis=1)
        Pis_opt=np.concatenate((Pi,Ps))
        Wis_opt=computeOptWeightsAtlas(Pis_opt,Pis_j,Wis_opt)
        Wis_iter=computeOptWeightsAtlas(Pis_iter,Pis_j,Wis_iter)
        for iter in xrange(iterations):
            hatPs=np.sum([Wis_iter[j]*Pl_j[j,:] for j in xrange(num_atlasexamples)], axis=0)
            Pis_iter=np.concatenate((Pi,hatPs))
            Wis_opt=computeOptWeightsAtlas(Pis_opt,Pis_j,Wis_opt)
            Wis_iter=computeOptWeightsAtlas(Pis_iter,Pis_j,Wis_iter)


    #--Compute loss--#
    num_examples=Pi_j.shape[0]
    Ps_i=np.sum([Ws_opt[j]*Pi_j[j,:] for j in xrange(num_examples)] ,axis=0)
    Ps_s=np.sum([Ws_opt[j]*Ps_j[j,:] for j in xrange(num_examples)] ,axis=0)
    Pi_i=np.sum([Wi_opt[j]*Pi_j[j,:] for j in xrange(num_examples)] ,axis=0)
    Pi_s=np.sum([Wi_opt[j]*Ps_j[j,:] for j in xrange(num_examples)] ,axis=0)
    Pnlm_i=np.sum([Wi_nlm[j]*Pi_j[j,:] for j in xrange(num_examples)] ,axis=0)
    Pnlm_s=np.sum([Wi_nlm[j]*Ps_j[j,:] for j in xrange(num_examples)] ,axis=0)
    if args.atlas!=None:
        Pa_nlm_i=np.sum([Wa_nlm[j]*Pa_j[j,:] for j in xrange(num_examples)] ,axis=0)
        Pa_nlm_s=np.sum([Wa_nlm[j]*Pl_j[j,:] for j in xrange(num_examples)] ,axis=0)
        Pis_opt_i=np.sum([Wis_opt[j]*Pa_j[j,:] for j in xrange(num_atlasexamples)], axis=0)
        Pis_opt_s=np.sum([Wis_opt[j]*Pl_j[j,:] for j in xrange(num_atlasexamples)], axis=0)
        Pis_iter_i=np.sum([Wis_iter[j]*Pa_j[j,:] for j in xrange(num_atlasexamples)], axis=0)
        Pis_iter_s=np.sum([Wis_iter[j]*Pl_j[j,:] for j in xrange(num_atlasexamples)], axis=0)

    loss_Ps_i=np.linalg.norm(Ps_i-Pi)/np.float(num_examples)
    loss_Ps_s=np.linalg.norm(Ps_s-Ps)/np.float(num_examples)
    loss_Pi_i=np.linalg.norm(Pi_i-Pi)/np.float(num_examples)
    loss_Pi_s=np.linalg.norm(Pi_s-Ps)/np.float(num_examples)
    loss_Pnlm_i=np.linalg.norm(Pnlm_i-Pi)/np.float(num_examples)
    loss_Pnlm_s=np.linalg.norm(Pnlm_s-Ps)/np.float(num_examples)
    if args.atlas!=None:
        loss_Pa_nlm_i=np.linalg.norm(Pa_nlm_i-Pi)/np.float(num_atlasexamples)
        loss_Pa_nlm_s=np.linalg.norm(Pa_nlm_s-Ps)/np.float(num_atlasexamples)
        loss_Pis_opt_i=np.linalg.norm(Pis_opt_i-Pi)/np.float(num_atlasexamples)
        loss_Pis_opt_s=np.linalg.norm(Pis_opt_s-Ps)/np.float(num_atlasexamples)
        loss_Pis_iter_i=np.linalg.norm(Pis_iter_i-Pi)/np.float(num_atlasexamples)
        loss_Pis_iter_s=np.linalg.norm(Pis_iter_s-Ps)/np.float(num_atlasexamples)


    #--Compute correlations--#

    corr_ws_wi = np.abs(Ws_opt - Wi_opt)
    corr_ws_wnlm = np.abs(Ws_opt - Wi_nlm)
    corr_wi_wnlm = np.abs(Ws_opt - Wi_nlm)

    corr_Ps=np.zeros(num_examples)
    corr_Pi=np.zeros(num_examples)
    diff_PnCs=np.zeros(num_examples)
    diff_PnCi=np.zeros(num_examples)
    diff_PCs=np.zeros(num_examples)
    diff_PCi=np.zeros(num_examples)
    center = hps[0] + (2*hps[1]+1) * ( hps[1] + (2*hps[2]+1) * hps[2] )
    for j in range(num_examples):
        corr_Ps[j]=np.corrcoef(Pi_j[j,:],Pi)[1,0]
        corr_Pi[j]=np.corrcoef(Ps_j[j,:],Ps)[1,0]
        diff_PnCs[j]=np.linalg.norm([ps-Ps_j[j,index] for index,ps in enumerate(Ps) if index!=center])
        diff_PnCi[j]=np.linalg.norm([pi-Pi_j[j,index] for index,pi in enumerate(Pi) if index!=center])
        diff_PCs[j]=np.linalg.norm(Ps[center]-Ps_j[j,center])
        diff_PCi[j]=np.linalg.norm(Pi[center]-Pi_j[j,center])

    max_corr_y=np.ceil( np.max([corr_ws_wi,corr_ws_wnlm,corr_wi_wnlm]) *10.)/10.

    Pi=Pi.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
    Ps=Ps.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
    Ps_i=Ps_i.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
    Ps_s=Ps_s.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
    Pi_i=Pi_i.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
    Pi_s=Pi_s.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
    Pnlm_i=Pnlm_i.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
    Pnlm_s=Pnlm_s.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
    if args.atlas!=None:
        Pa_nlm_i=Pa_nlm_i.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
        Pa_nlm_s=Pa_nlm_s.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
        Pis_opt_i=Pis_opt_i.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
        Pis_opt_s=Pis_opt_s.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
        Pis_iter_i=Pis_iter_i.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
        Pis_iter_s=Pis_iter_s.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])

    ###--Show output plots--###

    if args.atlas!=None:
        sl=int(round(Ps_i.shape[2]/2))
        font={'family': 'serif','color':  'darkred','weight': 'normal','size': 16}

        f, axarr = plt.subplots(4, 4)
        axarr[0, 0].set_title('Int', fontdict=font)
        axarr[0, 0].imshow(np.rot90(Pi[:,:,sl]), cmap="Greys_r", interpolation='nearest')
        axarr[0, 0].axis('off')
        axarr[0, 1].set_title('Seg', fontdict=font)
        axarr[0, 1].imshow(np.rot90(Ps[:,:,sl]), cmap="Greys_r", interpolation='nearest')
        axarr[0, 1].axis('off')
        axarr[0, 2].set_title('Loss Int', fontdict=font)
        axarr[0, 2].axis('off')
        axarr[0, 3].set_title('Loss Seg', fontdict=font)
        axarr[0, 3].axis('off')

        axarr[1, 0].text(sl*(-3),sl*1.25,'Wis_opt', fontdict=font)
        axarr[1, 0].imshow(np.rot90(Pis_opt_i[:,:,sl]), cmap="Greys_r", interpolation='nearest')
        axarr[1, 0].axis('off')
        axarr[1, 1].imshow(np.rot90(Pis_opt_s[:,:,sl]), cmap="Greys_r", interpolation='nearest')
        axarr[1, 1].axis('off')
        axarr[1, 2].text(sl*0.15,sl*0.2,str(round(loss_Pis_opt_i,2)))
        axarr[1, 2].axis('off')
        axarr[1, 3].text(sl*0.15,sl*0.2,str(round(loss_Pis_opt_s,2)))
        axarr[1, 3].axis('off')

        axarr[2, 0].text(sl*(-3),sl*1.25,'Wis_iter', fontdict=font)
        axarr[2, 0].imshow(np.rot90(Pis_iter_i[:,:,sl]), cmap="Greys_r", interpolation='nearest')
        axarr[2, 0].axis('off')
        axarr[2, 1].imshow(np.rot90(Pis_iter_s[:,:,sl]), cmap="Greys_r", interpolation='nearest')
        axarr[2, 1].axis('off')
        axarr[2, 2].text(sl*0.15,sl*0.2,str(round(loss_Pis_iter_i,2)))
        axarr[2, 2].axis('off')
        axarr[2, 3].text(sl*0.15,sl*0.2,str(round(loss_Pis_iter_s,2)))
        axarr[2, 3].axis('off')

        axarr[3, 0].text(sl*(-3),sl*1.25,'Wnlm', fontdict=font)
        axarr[3, 0].imshow(np.rot90(Pa_nlm_i[:,:,sl]), cmap="Greys_r", interpolation='nearest')
        axarr[3, 0].axis('off')
        axarr[3, 1].imshow(np.rot90(Pa_nlm_s[:,:,sl]), cmap="Greys_r", interpolation='nearest')
        axarr[3, 1].axis('off')
        axarr[3, 2].text(sl*0.15,sl*0.2,str(round(loss_Pa_nlm_i,2)))
        axarr[3, 2].axis('off')
        axarr[3, 3].text(sl*0.15,sl*0.2,str(round(loss_Pa_nlm_s,2)))
        axarr[3, 3].axis('off')


        f2, axarr2 = plt.subplots(3,figsize=(3,9))
        axarr2[0].plot(Wis_opt,Wis_iter,'go')
        axarr2[0].plot([0,1],[0,1],'r')
        axarr2[0].set_xlabel('Wis opt')
        axarr2[0].set_ylabel('Wis iter')

        axarr2[1].plot(Wis_opt,Wa_nlm,'go')
        axarr2[1].plot([0,1],[0,1],'r')
        axarr2[1].set_xlabel('Wis opt')
        axarr2[1].set_ylabel('Wnlm')

        axarr2[2].plot(Wis_iter,Wa_nlm,'go')
        axarr2[2].plot([0,1],[0,1],'r')
        axarr2[2].set_xlabel('Wis iter')
        axarr2[2].set_ylabel('Wnlm')

        for ax in axarr2:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set(adjustable='box-forced', aspect='equal')

    else:
        sl=int(round(Ps_i.shape[2]/2))
        font={'family': 'serif','color':  'darkred','weight': 'normal','size': 16}

        f, axarr = plt.subplots(4, 4)
        axarr[0, 0].set_title('Int', fontdict=font)
        axarr[0, 0].imshow(np.rot90(Pi[:,:,sl]), cmap="Greys_r", interpolation='nearest')
        axarr[0, 0].axis('off')
        axarr[0, 1].set_title('Seg', fontdict=font)
        axarr[0, 1].imshow(np.rot90(Ps[:,:,sl]), cmap="Greys_r", interpolation='nearest')
        axarr[0, 1].axis('off')
        axarr[0, 2].set_title('Loss Int', fontdict=font)
        axarr[0, 2].axis('off')
        axarr[0, 3].set_title('Loss Seg', fontdict=font)
        axarr[0, 3].axis('off')

        axarr[1, 0].text(sl*(-3),sl*1.25,'Ws', fontdict=font)
        axarr[1, 0].imshow(np.rot90(Ps_i[:,:,sl]), cmap="Greys_r", interpolation='nearest')
        axarr[1, 0].axis('off')
        axarr[1, 1].imshow(np.rot90(Ps_s[:,:,sl]), cmap="Greys_r", interpolation='nearest')
        axarr[1, 1].axis('off')
        axarr[1, 2].text(sl*0.15,sl*0.2,str(round(loss_Ps_i,2)))
        axarr[1, 2].axis('off')
        axarr[1, 3].text(sl*0.15,sl*0.2,str(round(loss_Ps_s,2)))
        axarr[1, 3].axis('off')

        axarr[2, 0].text(sl*(-3),sl*1.25,'Wi', fontdict=font)
        axarr[2, 0].imshow(np.rot90(Pi_i[:,:,sl]), cmap="Greys_r", interpolation='nearest')
        axarr[2, 0].axis('off')
        axarr[2, 1].imshow(np.rot90(Pi_s[:,:,sl]), cmap="Greys_r", interpolation='nearest')
        axarr[2, 1].axis('off')
        axarr[2, 2].text(sl*0.15,sl*0.2,str(round(loss_Pi_i,2)))
        axarr[2, 2].axis('off')
        axarr[2, 3].text(sl*0.15,sl*0.2,str(round(loss_Pi_s,2)))
        axarr[2, 3].axis('off')

        axarr[3, 0].text(sl*(-3),sl*1.25,'Wnlm', fontdict=font)
        axarr[3, 0].imshow(np.rot90(Pnlm_i[:,:,sl]), cmap="Greys_r", interpolation='nearest')
        axarr[3, 0].axis('off')
        axarr[3, 1].imshow(np.rot90(Pnlm_s[:,:,sl]), cmap="Greys_r", interpolation='nearest')
        axarr[3, 1].axis('off')
        axarr[3, 2].text(sl*0.15,sl*0.2,str(round(loss_Pnlm_i,2)))
        axarr[3, 2].axis('off')
        axarr[3, 3].text(sl*0.15,sl*0.2,str(round(loss_Pnlm_s,2)))
        axarr[3, 3].axis('off')


        f2, axarr2 = plt.subplots(3,figsize=(3,9))
        axarr2[0].plot(Ws_opt,Wi_opt,'go')
        axarr2[0].plot([0,1],[0,1],'r')
        axarr2[0].set_xlabel('Ws opt')
        axarr2[0].set_ylabel('Wi opt')

        axarr2[1].plot(Ws_opt,Wi_nlm,'go')
        axarr2[1].plot([0,1],[0,1],'r')
        axarr2[1].set_xlabel('Ws opt')
        axarr2[1].set_ylabel('Wnlm')

        axarr2[2].plot(Wi_opt,Wi_nlm,'go')
        axarr2[2].plot([0,1],[0,1],'r')
        axarr2[2].set_xlabel('Wi opt')
        axarr2[2].set_ylabel('Wnlm')

        for ax in axarr2:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set(adjustable='box-forced', aspect='equal')


        f3, axarr3 = plt.subplots(3,figsize=(3,9))
        axarr3[0].set_title('Seg', fontdict=font)
        axarr3[0].plot(corr_Ps,corr_ws_wi,'go')
        axarr3[0].set_xlabel('Cross-correlation between patches')
        axarr3[0].set_ylabel('Ws-Wi error')

        axarr3[1].plot(corr_Ps,corr_ws_wnlm,'go')
        axarr3[1].set_xlabel('Cross-correlation between patches')
        axarr3[1].set_ylabel('Ws-Wnlm error')

        axarr3[2].plot(corr_Ps,corr_wi_wnlm,'go')
        axarr3[2].set_xlabel('Cross-correlation between patches')
        axarr3[2].set_ylabel('Wi-Wnlm error')

        for ax in axarr3:
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, max_corr_y)


        f4, axarr4 = plt.subplots(3,figsize=(3,9))
        axarr4[0].set_title('Int', fontdict=font)
        axarr4[0].plot(corr_Pi,corr_ws_wi,'go')
        axarr4[0].set_xlabel('Cross-correlation between patches')
        axarr4[0].set_ylabel('Ws-Wi error')

        axarr4[1].plot(corr_Pi,corr_ws_wnlm,'go')
        axarr4[1].set_xlabel('Cross-correlation between patches')
        axarr4[1].set_ylabel('Ws-Wnlm error')

        axarr4[2].plot(corr_Pi,corr_wi_wnlm,'go')
        axarr4[2].set_xlabel('Cross-correlation between patches')
        axarr4[2].set_ylabel('Wi-Wnlm error')

        for ax in axarr4:
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, max_corr_y)



        f5, axarr5 = plt.subplots(2,figsize=(5,8))
        axarr5[0].set_title('Seg', fontdict=font)
        axarr5[0].plot(diff_PnCs,diff_PCs,'go')
        axarr5[0].set_xlabel('patch distance without central voxel')
        axarr5[0].set_ylabel('distance of central voxel')

        axarr5[1].set_title('Int', fontdict=font)
        axarr5[1].plot(diff_PnCi,diff_PCi,'go')
        axarr5[1].set_xlabel('patch distance without central voxel')
        axarr5[1].set_ylabel('distance of central voxel')

        for ax in axarr5:
            ax.set_xlim(0, 1000)
            ax.set_ylim(0, 200)


    plt.show()
