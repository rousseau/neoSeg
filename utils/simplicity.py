# coding=utf-8

import numpy as np
from skimage import measure




############## Functions for a binary image ##############

def isBinarySimplePoint6_26(im):
    '''
    Check the simplicity of a point in a binary image considering the object and
    the background in 6-adjancy and 26-adjancy respectly.
    -------------------------------
    Input:  im      --> A binary image of size 3x3x3
    Output: simplicity  --> A digit that provides the simplicity for the label
                            (0: non-simple pair, 1: simple pair)
    '''
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
    '''
    Check the simplicity of a point in a binary image considering the object and
    the background in 26-adjancy and 6-adjancy respectly.
    -------------------------------
    Input:  im      --> A binary image of size 3x3x3
    Output: simplicity  --> A digit that provides the simplicity for the label
                            (0: non-simple pair, 1: simple pair)
    '''
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



    ## Simple pairs ##

def isBinarySimplePair6_26(im,center):
    '''
    Check the simplicity of a pair in a binary image considering the object and
    the background in 6-adjancy and 26-adjancy respectly.
    -------------------------------
    Input:  im      --> A binary image of size 4x3x3 or 3x4x3 or 3x3x4
            center  --> Coordinates of the pair that corresponds to the
                        center of the multilabel image
    Output: simplicity  --> A digit that provides the simplicity for the label
                            (0: non-simple pair, 1: simple pair)
    '''
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
        # X[center] = 1-X[1,1,1]
        X[center] = 0
        ## Labels 26-connected ##
        label,numLabel=measure.label(X,connectivity=3,return_num=True)
        if numLabel==1:
            return 1
    return 0



def isBinarySimplePair26_6(im,center):
    '''
    Check the simplicity of a pair in a binary image considering the object and
    the background in 26-adjancy and 6-adjancy respectly.
    -------------------------------
    Input:  im      --> A binary image of size 4x3x3 or 3x4x3 or 3x3x4
            center  --> Coordinates of the pair that corresponds to the
                        center of the multilabel image
    Output: simplicity  --> A digit that provides the simplicity for the label
                            (0: non-simple pair, 1: simple pair)
    '''
    X = im.copy()
    ### 26-connected test ###
    # X[center] = 1-X[1,1,1]
    X[center] = 0
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







############## Functions for a multilabel image ##############


def isSimple6_26(im,topologyList,totalLabel):
    '''
    Check the simplicity of a point in multiple labels considering the object and
    the background in 6-adjancy and 26-adjancy respectly.
    -------------------------------
    Input:  im              --> A multilabel image of size 3x3x3
            topologyList    --> Configuration of label topologies to be preserved
            totalLabel      --> Number of labels
    Output: simplicity  --> An array given the simplicity in each label
                            (0: non-simple point, 1: simple point)
    '''
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
    '''
    Check the simplicity of a point in multiple labels considering the object and
    the background in 26-adjancy and 6-adjancy respectly.
    -------------------------------
    Input:  im              --> A multilabel image of size 3x3x3
            topologyList    --> Configuration of label topologies to be preserved
            totalLabel      --> Number of labels
    Output: simplicity  --> An array given the simplicity in each label
                            (0: non-simple point, 1: simple point)
    '''
    simplicity=[0]*totalLabel #np.zeros([totalLabel],dtype=int)
    im_c=im.copy()
    im_c[im_c==im[1,1,1]]=-1
    candidatesLabel=np.unique(im_c)[1:]
    for cand in candidatesLabel:
        simplicityLabel=1
        ## Discard when current label and evalued label are in a topology group or there are not at all ##
        topoResList=[topo for topo in topologyList if ((cand in topo) + (im[1,1,1] in topo))==1]
        for topo in topoResList:
            im_c=np.zeros(im.shape)
            ## Evaluation of all labels ##
            for t in topo:
                im_c+=1*(im==t)
            if (len(np.unique(im_c))>1):
                simplicityLabel*=isBinarySimplePoint26_6(im_c)
        simplicity[cand]=simplicityLabel
    return simplicity



def isSimpleConn(im,topologyList,connectivityList,totalLabel):
    '''
    Check the simplicity of a point in multiple labels given the connectivity
    configuration (only available in 6/26 or 26/6).
    -------------------------------
    Input:  im                  --> A multilabel image of size 3x3x3
            topologyList        --> Configuration of label topologies
            connectivityList    --> Configuration of label connectivity
            totalLabel          --> Number of labels
    Output: simplicity  --> An array given the simplicity in each label
                            (0: non-simple point, 1: simple point)
    '''
    simplicity=[0]*totalLabel #np.zeros([totalLabel],dtype=int)
    im_c=im.copy()
    im_c[im_c==im[1,1,1]]=-1
    candidatesLabel=np.unique(im_c)[1:]
    for cand in candidatesLabel:
        simplicityLabel=1
        ## Discard when current label and evalued label are in a topology group or there are not at all ##
        topoResList=[topo for topo in topologyList if ((cand in topo) + (im[1,1,1] in topo))==1]
        for topo in topoResList:
            im_c=np.zeros(im.shape)
            ## Evaluation of all labels ##
            # conn=connectivityList[topologyList.index(topo)]
            for t in topo:
                im_c+=1*(im==t)
            if (len(np.unique(im_c))>1):
                if connectivityList[topologyList.index(topo)]==26:
                    simplicityLabel*=isBinarySimplePoint26_6(im_c)
                elif connectivityList[topologyList.index(topo)]==6:
                    simplicityLabel*=isBinarySimplePoint6_26(im_c)
                else:
                    print('Error in connectivity')
                    sys.exit()
        simplicity[cand]=simplicityLabel

    return simplicity



    ## Simple pairs ##

def isSimplePair6_26(im,center,topologyList,totalLabel):
    '''
    Check the simplicity of a pair in multiple labels considering the object and
    the background in 6-adjancy and 26-adjancy respectly.
    -------------------------------
    Input:  im              --> A multilabel image of size 4x3x3 or 3x4x3 or 3x3x4
            center          --> Coordinates of the pair that corresponds to the
                                center of the multilabel image
            topologyList    --> Configuration of label topologies to be preserved
            totalLabel      --> Number of labels
    Output: simplicity  --> An array given the simplicity in each label
                            (0: non-simple pair, 1: simple pair)
    '''
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
                simplicityLabel*=isBinarySimplePair6_26(im_c,center)
        simplicity[cand]=simplicityLabel
    return simplicity


def isSimplePair6_26Label(im,center,label,topologyList,totalLabel):
    '''
    Check the simplicity of a pair in multiple labels for a specific label
    considering the object and the background in 6-adjancy and 26-adjancy respectly.
    -------------------------------
    Input:  im              --> A multilabel image of size 4x3x3 or 3x4x3 or 3x3x4
            center          --> Coordinates of the pair that corresponds to the
                                center of the multilabel image
            label           --> Label that is evaluated
            topologyList    --> Configuration of label topologies to be preserved
            totalLabel      --> Number of labels
    Output: simplicity  --> A digit that provides the simplicity for the label
                            (0: non-simple pair, 1: simple pair)
    '''
    simplicityLabel=0
    im_c=im.copy()
    im_c[im_c==im[1,1,1]]=-1
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
    '''
    Check the simplicity of a pair in multiple labels considering the object and
    the background in 26-adjancy and 6-adjancy respectly.
    -------------------------------
    Input:  im              --> A multilabel image of size 4x3x3 or 3x4x3 or 3x3x4
            center          --> Coordinates of the pair that corresponds to the
                                center of the multilabel image
            topologyList    --> Configuration of label topologies to be preserved
            totalLabel      --> Number of labels
    Output: simplicity  --> An array given the simplicity in each label
                            (0: non-simple pair, 1: simple pair)
    '''
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
                simplicityLabel*=isBinarySimplePair26_6(im_c,center)
        simplicity[cand]=simplicityLabel
    return simplicity

def isSimplePair26_6Label(im,center,label,topologyList,totalLabel):
    '''
    Check the simplicity of a pair in multiple labels for a specific label
    considering the object and the background in 26-adjancy and 6-adjancy respectly.
    -------------------------------
    Input:  im              --> A multilabel image of size 4x3x3 or 3x4x3 or 3x3x4
            center          --> Coordinates of the pair that corresponds to the
                                center of the multilabel image
            label           --> Label that is evaluated
            topologyList    --> Configuration of label topologies to be preserved
            totalLabel      --> Number of labels
    Output: simplicity  --> A binary number given the simplicity for the label
                            (0: non-simple pair, 1: simple pair)
    '''
    simplicityLabel=0
    im_c=im.copy()
    im_c[im_c==im[1,1,1]]=-1
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


def isSimplePairConnLabel(im,center,label,topologyList,connectivityList,totalLabel):
    '''
    Check the simplicity of a pair in multiple labels for a specific label
    given the connectivity configuration (only available in 6/26 or 26/6).
    -------------------------------
    Input:  im                  --> A multilabel image of size 4x3x3 or 3x4x3 or 3x3x4
            center              --> Coordinates of the pair that corresponds to the
                                    center of the multilabel image
            label               --> Label that is evaluated
            topologyList        --> Configuration of label topologies to be preserved
            connectivityList    --> Configuration of label connectivity
            totalLabel          --> Number of labels
    Output: simplicity  --> A binary number given the simplicity for the label
                            (0: non-simple pair, 1: simple pair)
    '''
    simplicityLabel=0
    im_c=im.copy()
    im_c[im_c==im[1,1,1]]=-1
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
                if connectivityList[topologyList.index(topo)]==26:
                    simplicityLabel*=isBinarySimplePair26_6(im_c,center)
                elif connectivityList[topologyList.index(topo)]==6:
                    simplicityLabel*=isBinarySimplePair6_26(im_c,center)
                else:
                    print('Error in connectivity')
                    sys.exit()

    return simplicityLabel
