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

import argparse
import nibabel
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy import ndimage


def computeOptWeights(P,P_j,initialisation,reg=1e-03):
    Pw=P.copy()
    Pw_j=P_j.copy()
    w=initialisation.copy()
    num_example=Pw_j.shape[0]
    weights=np.zeros(num_example)
    Z=np.zeros(Pw_j.shape)
    b=np.ones(num_example)
    for j in range(num_example):
        Z[j,:]=Pw_j[j,:]-Pw
    C=np.dot(Z,np.transpose(Z))
    trace = np.trace(C)
    R = reg * trace if trace>0 else reg
    C.flat[::num_example + 1] += R
    fmin = lambda x: np.linalg.norm(np.dot(C,x)-b)
    sol = optimize.minimize(fmin, w, method='L-BFGS-B', bounds=[(0.,1.) for x in xrange(num_example)])
    w = sol['x']
    weights=w/np.sum(w)
    return weights


def computeNLMWeights(P,P_j,h):
    Pw=P.copy()
    Pw_j=P_j.copy()
    num_example=Pw_j.shape[0]
    w=np.zeros([num_example])
    dist_j=np.array([np.sum((Pw-Pw_j[j,:])**2) for j in range(num_example)]) #distance between Pw and Pw_j
    w=np.exp(-(dist_j/h))
    weights=w/np.sum(w)
    return weights


def computeSigmaNoise(img):
    hx=np.array([[[0.,0.,0.],[0.,-(1./6.),0.],[0.,0.,0.]],
                [[0.,-(1./6.),0.],[-(1./6.),1.,-(1./6.)],[0.,-(1./6.),0.]],
                [[0.,0.,0.],[0.,-(1./6.),0.],[0.,0.,0.]]])
    sigma2=(6.0/7.0) * np.sum(np.square(ndimage.convolve(img,hx))) / np.float(img.shape[0]*img.shape[1]*img.shape[2])
    return sigma2


def extractPatches(data,i,hss,hps):
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='patchAnalysis')

    parser.add_argument('-i', '--input', help='Input anatomical image (required)', type=str, required = True)
    parser.add_argument('-s', '--seg', help='Input segmentation image (probability map, required)', type=str, required = True)
    parser.add_argument('-x', '--x', help='x location (required)', type=int, required = True)
    parser.add_argument('-y', '--y', help='y location (required)', type=int, required = True)
    parser.add_argument('-z', '--z', help='z location (required)', type=int, required = True)
    parser.add_argument('-hss', '--hss', help='Half search area size (default)', type=int, default=1, required = False)
    parser.add_argument('-hps', '--hps', help='Half path size', type=int, default=1, required = False)


    args = parser.parse_args()

    ###--Print Input Information--###

    print 'Input anatomical image: ',args.input
    print 'Input segmentation image (probability map): ',args.seg
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

    ###--Compute analysis--###

    #--Extract patches--#
    Pi,Pi_j=extractPatches(input,vx,hss,hps)
    Ps,Ps_j=extractPatches(seg,vx,hss,hps)

    #--Compute weights--#
    W_nlm=computeNLMWeights(Pi,Pi_j,h)
    Ws_opt=computeOptWeights(Ps,Ps_j,W_nlm)
    Wi_opt=computeOptWeights(Pi,Pi_j,W_nlm)

    #--Compute loss--#
    num_example=Pi_j.shape[0]
    Ps_i=np.zeros(Pi.shape)
    Ps_s=np.zeros(Pi.shape)
    Pi_i=np.zeros(Pi.shape)
    Pi_s=np.zeros(Pi.shape)
    Pnlm_i=np.zeros(Pi.shape)
    Pnlm_s=np.zeros(Pi.shape)
    for j in range(num_example):
        Ps_i+=Ws_opt[j]*Pi_j[j,:]
        Ps_s+=Ws_opt[j]*Ps_j[j,:]
        Pi_i+=Wi_opt[j]*Pi_j[j,:]
        Pi_s+=Wi_opt[j]*Ps_j[j,:]
        Pnlm_i+=W_nlm[j]*Pi_j[j,:]
        Pnlm_s+=W_nlm[j]*Ps_j[j,:]

    loss_Ps_i=np.linalg.norm(Ps_i-Pi)/np.float(num_example)
    loss_Ps_s=np.linalg.norm(Ps_s-Ps)/np.float(num_example)
    loss_Pi_i=np.linalg.norm(Pi_i-Pi)/np.float(num_example)
    loss_Pi_s=np.linalg.norm(Pi_s-Ps)/np.float(num_example)
    loss_Pnlm_i=np.linalg.norm(Pnlm_i-Pi)/np.float(num_example)
    loss_Pnlm_s=np.linalg.norm(Pnlm_s-Ps)/np.float(num_example)

    corr_ws_wi = np.abs(Ws_opt - Wi_opt)
    corr_ws_wnlm = np.abs(Ws_opt - W_nlm)
    corr_wi_wnlm = np.abs(Ws_opt - W_nlm)

    corr_Ps=np.zeros(num_example)
    corr_Pi=np.zeros(num_example)
    for j in range(num_example):
        corr_Ps[j]=np.corrcoef(Pi_j[j,:],Pi)[1,0]
        corr_Pi[j]=np.corrcoef(Ps_j[j,:],Ps)[1,0]

    max_corr_y=np.ceil( np.max([corr_ws_wi,corr_ws_wnlm,corr_wi_wnlm]) *10.)/10.

    Pi=Pi.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
    Ps=Ps.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
    Ps_i=Ps_i.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
    Ps_s=Ps_s.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
    Pi_i=Pi_i.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
    Pi_s=Pi_s.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
    Pnlm_i=Pnlm_i.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])
    Pnlm_s=Pnlm_s.reshape([(2*hps[0]+1),(2*hps[1]+1),(2*hps[2]+1)])

    ###--Show output plots--###

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

    axarr2[1].plot(Ws_opt,W_nlm,'go')
    axarr2[1].plot([0,1],[0,1],'r')
    axarr2[1].set_xlabel('Ws opt')
    axarr2[1].set_ylabel('Wnlm')

    axarr2[2].plot(Wi_opt,W_nlm,'go')
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
    axarr3[0].set_xlabel('corr_Ps')
    axarr3[0].set_ylabel('corr_ws_wi')

    axarr3[1].plot(corr_Ps,corr_ws_wnlm,'go')
    axarr3[1].set_xlabel('corr_Ps')
    axarr3[1].set_ylabel('corr_ws_wnlm')

    axarr3[2].plot(corr_Ps,corr_wi_wnlm,'go')
    axarr3[2].set_xlabel('corr_Ps')
    axarr3[2].set_ylabel('corr_wi_wnlm')

    for ax in axarr3:
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, max_corr_y)


    f4, axarr4 = plt.subplots(3,figsize=(3,9))
    axarr4[0].set_title('Int', fontdict=font)
    axarr4[0].plot(corr_Pi,corr_ws_wi,'go')
    axarr4[0].set_xlabel('corr_Pi')
    axarr4[0].set_ylabel('corr_ws_wi')

    axarr4[1].plot(corr_Pi,corr_ws_wnlm,'go')
    axarr4[1].set_xlabel('corr_Pi')
    axarr4[1].set_ylabel('corr_ws_wnlm')

    axarr4[2].plot(corr_Pi,corr_wi_wnlm,'go')
    axarr4[2].set_xlabel('corr_Pi')
    axarr4[2].set_ylabel('corr_wi_wnlm')

    for ax in axarr4:
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, max_corr_y)


    plt.show()
