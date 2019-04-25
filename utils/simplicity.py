# coding=utf-8

import numpy as np
from skimage import measure



def isBinarySimplePoint6_26(im):
    X = im.copy()
    ### 6-connected test ###
    ## X without corners ##
    X[np.array([[[True,False,True],[False,False,False],[True,False,True]],[[False,False,False],[False,False,False],[False,False,False]],[[True,False,True],[False,False,False],[True,False,True]]])]=0
    ## Labels 6-connected ##
    label,numLabelCenter=measure.label(X,connectivity=1,return_num=True)
    ## Labels 6-connected switching the center ##
    X[1,1,1] = 1-X[1,1,1]
    label,numLabelNoCenter=measure.label(X,connectivity=1,return_num=True)
    ## Numbers of labels have to be equals ##
    if numLabelCenter-numLabelNoCenter==0:
        ### 26-connected background test ###
        ## Complementary of X without center ##
        X = 1-im
        X[1,1,1] = 0
        ## Labels 26-connected ##
        label,numLabel=measure.label(X,connectivity=3,return_num=True)
        ## Result ###
        if numLabel==1:
            return 1
    return 0


def isBinarySimplePoint26_6(im):
    X = im.copy()
    ### 26-connected test ###
    X[1,1,1] = 0
    ## Labels 26-connected ##
    label,numLabel=measure.label(X,connectivity=3,return_num=True)
    if numLabel==1:
        ### 6-connected test background ###
        ## Complementary of X ##
        X = 1-im
        ## X without corners ##
        X[np.array([[[True,False,True],[False,False,False],[True,False,True]],[[False,False,False],[False,False,False],[False,False,False]],[[True,False,True],[False,False,False],[True,False,True]]])]=0
        ## Labels 6-connected ##
        label,numLabelCenter=measure.label(X,connectivity=1,return_num=True)
        ## Labels 6-connected switching the center ##
        X[1,1,1] = 1-X[1,1,1]
        label,numLabelNoCenter=measure.label(X,connectivity=1,return_num=True)
        if (numLabelCenter-numLabelNoCenter==0):
            return 1
    return 0



def isBinarySimplePair6_26(im,center):
    X = im.copy()
    ### 6-connected test ####
    ## X without corners ##
    X[0,0,0]=0
    X[0,0,-1]=0
    X[0,-1,0]=0
    X[0,-1,-1]=0
    X[-1,0,0]=0
    X[-1,0,-1]=0
    X[-1,-1,0]=0
    X[-1,-1,-1]=0
    ## Labels 6-connected ##
    label,numLabelCenter=measure.label(X,connectivity=1,return_num=True)
    ## Labels 6-connected switching the center ##
    X[center] = 1-X[1,1,1]
    label,numLabelNoCenter=measure.label(X,connectivity=1,return_num=True)
    ## Numbers of labels have to be equals ##
    if (numLabelCenter-numLabelNoCenter==0):
        ### 26-connected background test ###
        ## Complementary of X ##
        X = 1-im
        X[center] = 1-X[1,1,1]
        ## Labels 26-connected ##
        label,numLabel=measure.label(X,connectivity=3,return_num=True)
        if numLabel==1:
            return 1
    return 0



def isBinarySimplePair26_6(im,center):
    X = im.copy()
    ### 26-connected test ###
    X[center] = 1-X[1,1,1]
    ## Labels 26-connected ##
    label,numLabel=measure.label(X,connectivity=3,return_num=True)
    if numLabel==1:
        ### 6-connected background test ####
        ## Complementary of X ##
        X = 1-im
        ## X without corners ##
        X[0,0,0]=0
        X[0,0,-1]=0
        X[0,-1,0]=0
        X[0,-1,-1]=0
        X[-1,0,0]=0
        X[-1,0,-1]=0
        X[-1,-1,0]=0
        X[-1,-1,-1]=0
        ## Labels 6-connected keeping the center ##
        label,numLabelCenter=measure.label(X,connectivity=1,return_num=True)
        ## Labels 6-connected removing the center ##
        ## Switch the center of X ##
        X[center] = 1-X[1,1,1]
        label,numLabelNoCenter=measure.label(X,connectivity=1,return_num=True)
        ## Numbers of labels have to be equals ##
        if (numLabelCenter-numLabelNoCenter==0):
            return 1
    return 0




def isSimple6_26(im,topologyList,totalLabel):
    simplicity=np.zeros([totalLabel],dtype=int)
    im_c=im.copy()
    im_c[im_c==im[1,1,1]]=-1
    candidatesLabel=np.unique(im_c)[1:]
    for cand in candidatesLabel:
        simplicityLabel=1
        ## Discard when current label and evalued label are in a topology group or there are not at all ##
        topoResList=[topo for topo in topologyList if ((cand in topo) + (im[1,1,1] in topo))==1]
        ## Evaluate label simplicity in each topology ##
        for topo in topoResList:
            im_c=np.zeros(im.shape)
            ## Evaluation of all labels ##
            for t in topo:
                im_c+=1*(im==t)
            if (len(np.unique(im_c))>1):
                simplicityLabel*=isBinarySimplePoint6_26(im_c)
        simplicity[cand]=simplicityLabel
    return simplicity

def isSimple26_6(im,topologyList,totalLabel):
    simplicity=np.zeros([totalLabel],dtype=int)
    im_c=im.copy()
    im_c[im_c==im[1,1,1]]=-1
    candidatesLabel=np.unique(im_c)[1:]
    for cand in candidatesLabel:
        simplicityLabel=1
        ## Discard when current label and evalued label are in a topology group or there are not at all ##
        topoResList=[topo for topo in topologyList if ((cand in topo) + (im[1,1,1] in topo))==1]
        ## Evaluate label simplicity in each topology ##
        for topo in topoResList:
            im_c=np.zeros(im.shape)
            ## Evaluation of all labels ##
            for t in topo:
                im_c+=1*(im==t)
            if (len(np.unique(im_c))>1):
                simplicityLabel*=isBinarySimplePoint26_6(im_c)
        simplicity[cand]=simplicityLabel
    return simplicity



# OLDdef isSimple26_6(im,topologyList,totalLabel):
#     simplicity=np.zeros([totalLabel],dtype=int)
#     im_c=im.copy()
#     im_c[im_c==im[1,1,1]]=-1
#     candidatesLabel=np.unique(im_c)[1:]
#     numLabel=len(candidatesLabel)
#     ## Loop of labels in the neighbourhood of (x,y,z) ##
#     while(numLabel>0):
#         #evalLabel=candidatesLabel[numLabel-1]
#         simplicityLabel=1
#         numTopology=len(topologyList)
#         ## Evaluate label simplicity in each topology ##
#         while((simplicityLabel==1) & (numTopology>0)):
#             ## Discard when current label and evalued label are in a topology group or there are not at all ##
#             im_c=im.copy()
#             ## Evaluation of all labels ##
#             for l in xrange(totalLabel):
#                 im_c[im==l]=1 if l in topologyList[numTopology-1] else 0
#             if len(np.unique(im_c))>1:
#                 simplicityLabel=isBinarySimplePoint26_6(im_c)
#             numTopology-=1
#         simplicity[candidatesLabel[numLabel-1]]=simplicityLabel
#         numLabel-=1
#     return simplicity



def isSimplePair6_26(im,center,topologyList,totalLabel):
    simplicity=np.zeros([totalLabel],dtype=int)
    im_c=im.copy()
    im_c[im_c==im[center]]=-1
    candidatesLabel=np.unique(im_c)[1:]
    for cand in candidatesLabel:
        simplicityLabel=1
        ## Discard when current label and evalued label are in a topology group or there are not at all ##
        topoResList=[topo for topo in topologyList if ((cand in topo) + (im[1,1,1] in topo))==1]
        ## Evaluate label simplicity in each topology ##
        for topo in topoResList:
            im_c=np.zeros(im.shape)
            ## Evaluation of all labels ##
            for t in topo:
                im_c+=1*(im==t)
            if (len(np.unique(im_c))>1):
                simplicityLabel*=isBinarySimplePair6_26(im_c,center)
        simplicity[cand]=simplicityLabel
    return simplicity


def isSimplePair6_26Label(im,center,label,topologyList,totalLabel):
    simplicityLabel=0
    im_c=im.copy()
    im_c[im_c==im[center]]=-1
    candidatesLabel=np.unique(im_c)[1:]
    if label in candidatesLabel:
        simplicityLabel=1
        ## Discard when current label and evalued label are in a topology group or there are not at all ##
        topoResList=[topo for topo in topologyList if ((label in topo) + (im[1,1,1] in topo))==1]
        ## Evaluate label simplicity in each topology ##
        for topo in topoResList:
            im_c=np.zeros(im.shape)
            ## Evaluation of all labels ##
            for t in topo:
                im_c+=1*(im==t)
            if (len(np.unique(im_c))>1):
                simplicityLabel*=isBinarySimplePair6_26(im_c,center)
    return simplicityLabel


def isSimplePair26_6(im,center,topologyList,totalLabel):
    simplicity=np.zeros([totalLabel],dtype=int)
    im_c=im.copy()
    im_c[im_c==im[center]]=-1
    candidatesLabel=np.unique(im_c)[1:]
    for cand in candidatesLabel:
        simplicityLabel=1
        ## Discard when current label and evalued label are in a topology group or there are not at all ##
        topoResList=[topo for topo in topologyList if ((cand in topo) + (im[1,1,1] in topo))==1]
        ## Evaluate label simplicity in each topology ##
        for topo in topoResList:
            im_c=np.zeros(im.shape)
            ## Evaluation of all labels ##
            for t in topo:
                im_c+=1*(im==t)
            if (len(np.unique(im_c))>1):
                simplicityLabel*=isBinarySimplePair26_6(im_c,center)
        simplicity[cand]=simplicityLabel
    return simplicity

def isSimplePair26_6Label(im,center,label,topologyList,totalLabel):
    simplicityLabel=0
    im_c=im.copy()
    im_c[im_c==im[center]]=-1
    candidatesLabel=np.unique(im_c)[1:]
    if label in candidatesLabel:
        simplicityLabel=1
        ## Discard when current label and evalued label are in a topology group or there are not at all ##
        topoResList=[topo for topo in topologyList if ((label in topo) + (im[1,1,1] in topo))==1]
        ## Evaluate label simplicity in each topology ##
        for topo in topoResList:
            im_c=np.zeros(im.shape)
            ## Evaluation of all labels ##
            for t in topo:
                im_c+=1*(im==t)
            if (len(np.unique(im_c))>1):
                simplicityLabel*=isBinarySimplePair26_6(im_c,center)
    return simplicityLabel
