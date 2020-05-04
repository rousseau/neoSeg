# coding=utf-8
import os,sys
import numpy as np
from skimage import measure
from scipy import ndimage
from .simplicity import isSimpleConn, isSimple26_6, isSimple6_26, isSimplePairConnLabel,isBinarySimplePoint_26, \
						isSimplePair6_26, isSimplePair26_6, isSimplePair26_6Label, isSimplePair6_26Label, \
						isBinarySimplePoint26_6, isBinarySimplePoint6_26, isBinarySimplePoint_6
from .distances import 	computeBenefitPosDistance, computeBenefitPairPosDistance, computeBenefitPosSimilarity, \
						computeBenefitPairPosSimilarity, computeBenefitNegDistance, computeBenefitNegSimilarity




'''
#############################
####### Simple Points #######
#############################
'''

def initPosConn(Ilabel,Sref,distanceMap,topologyList,connectivityList,totalLabel,metric,padding):
	'''
    Compute the initialization for omega>=0 (i.e. Ilabel's size is lower
	than Sref's). This function allows to select the connectivity between
	object and background.
    -------------------------------
    Input:  Ilabel 				--> Image of labels
            Sref 				--> Segmentation of reference
			distanceMap			--> Distance transform map
			topologyList		--> Configuration of label topologies to be preserved
			connectivityList	-->	Configuration of label connectivity (6 or 26)
			totalLabel			--> Number of labels
			metric				--> Metric for the cost function
			padding				--> Padding for avoiding border effect
    Output: simpleMap    --> Simple point map
			benefitMap   --> Benefit map
    '''
	simpleMap = np.zeros(Sref.shape,dtype=int)
	benefitMap = np.zeros(Sref.shape)

	## Simple Points ##
	for x in range(padding,Ilabel.shape[0]-padding):
		for y in range(padding,Ilabel.shape[1]-padding):
			for z in range(padding,Ilabel.shape[2]-padding):
				simpleMap[x,y,z,:] = isSimpleConn(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2],topologyList,connectivityList,totalLabel)

	## Benefit Map ##
	pts= np.where(simpleMap==1)
	if metric == 'similarity':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],Sref[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric == 'distanceMap':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]]]-distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]]

	return simpleMap,benefitMap

def initAtlasPosConn(Ilabel,Sref,atlasPrior,distanceMap,topologyList,topologyAtlasList,connectivityList,connectivityAtlasList,totalLabel,metric,padding):
	'''
    Compute the initialization for omega>=0 (i.e. Ilabel's size is lower
	than Sref's). This function allows to select the connectivity between
	object and background. An atlas prior is used to lead the topological
	relaxation locally.
    -------------------------------
    Input:  Ilabel 					--> Image of labels
            Sref 					--> Segmentation of reference
			atlasPrior				--> Binary atlas defining areas where the
										topological relaxation will be applied
			distanceMap				--> Distance transform map
			topologyList			--> Configuration of label topologies to be
										preserved
			topologyAtlasList		--> Configuration of label topologies to be
										preserved in areas with topological relaxation
			connectivityList		-->	Configuration of label connectivity (6 or 26)
			connectivityAtlasList	-->	Configuration of label connectivity (6 or 26)
										for the topological relaxation
			totalLabel				--> Number of labels
			metric					--> Metric for the cost function
			padding					--> Padding for avoiding border effect
    Output: simpleMap    --> Simple point map
			benefitMap   --> Benefit map
    '''
	simpleMap = np.zeros(Sref.shape,dtype=int)
	benefitMap = np.zeros(Sref.shape)

	## Simple Points ##
	for x in range(padding,Ilabel.shape[0]-padding):
		for y in range(padding,Ilabel.shape[1]-padding):
			for z in range(padding,Ilabel.shape[2]-padding):
				if atlasPrior[x,y,z] == 0:
					simpleMap[x,y,z,:] = isSimpleConn(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2],topologyList,connectivityList,totalLabel) ###NEW###
				else:
					simpleMap[x,y,z,:] = isSimpleConn(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2],topologyAtlasList,connectivityAtlasList,totalLabel)

	## Benefit Map ##
	pts=np.where(simpleMap==1)
	if metric == 'similarity':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],Sref[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric == 'distanceMap':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]]]-distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]]

	return simpleMap,benefitMap



def initBenefitMapPosConn(Ilabel,Sref,distanceMap,simpleMap,topologyList,connectivityList,totalLabel,metric,padding):
	'''
    Compute the initialization providing just the benefit map for omega>=0 (i.e.
	Ilabel's size is lower than Sref's). This function allows to select the
	connectivity between object and background.
    -------------------------------
    Input:  Ilabel 				--> Image of labels
            Sref 				--> Segmentation of reference
			distanceMap			--> Distance transform map
			topologyList		--> Configuration of label topologies to be preserved
			connectivityList	-->	Configuration of label connectivity (6 or 26)
			totalLabel			--> Number of labels
			metric				--> Metric for the cost function
			padding				--> Padding for avoiding border effect
    Output: benefitMap   --> Benefit map
    '''
	benefitMap = np.zeros(Sref.shape)
	## Benefit Map ##
	pts = np.where(simpleMap==1)
	if metric == 'similarity':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],Sref[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric == 'distanceMap':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]]]-distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]]

	return benefitMap

def initBenefitMapAtlasPosConn(Ilabel,Sref,atlasPrior,distanceMap,simpleMap,topologyList,topologyAtlasList,connectivityList,connectivityAtlasList,totalLabel,metric,padding):
	'''
    Compute the initialization providing just the benefit map for omega>=0 (i.e.
	Ilabel's size is lower than Sref's). This function allows to select the
	connectivity between object and background. An atlas prior is used to lead
	the topological relaxation locally.
    -------------------------------
    Input:  Ilabel 					--> Image of labels
            Sref 					--> Segmentation of reference
			atlasPrior				--> Binary atlas defining areas where the
										topological relaxation will be applied
			distanceMap				--> Distance transform map
			topologyList			--> Configuration of label topologies to be
										preserved
			topologyAtlasList		--> Configuration of label topologies to be
										preserved in areas with topological relaxation
			connectivityList		-->	Configuration of label connectivity (6 or 26)
			connectivityAtlasList	-->	Configuration of label connectivity (6 or 26)
										for the topological relaxation
			totalLabel				--> Number of labels
			metric					--> Metric for the cost function
			padding					--> Padding for avoiding border effect
    Output: benefitMap   --> Benefit map
    '''
	benefitMap = np.zeros(Sref.shape)

	## Benefit Map ##
	pts = np.where(simpleMap==1)
	if metric == 'similarity':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],Sref[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric == 'distanceMap':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]]]-distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]]

	return benefitMap


def initPos6_26(Ilabel,Sref,distanceMap,topologyList,totalLabel,metric,padding):
	'''
    Compute the initialization for omega>=0 (i.e. Ilabel's size is lower
	than Sref's) considering the object and the background in 6-adjancy and
	26-adjancy respectly.
    -------------------------------
    Input:  Ilabel 			--> Image of labels
            Sref 			--> Segmentation of reference
			distanceMap		--> Distance transform map
			topologyList	--> Configuration of label topologies to be preserved
			totalLabel		--> Number of labels
			metric			--> Metric for the cost function
			padding			--> Padding for avoiding border effect
    Output: simpleMap    --> Simple point map
			benefitMap   --> Benefit map
    '''
	simpleMap = np.zeros(Sref.shape,dtype=int)
	benefitMap = np.zeros(Sref.shape)

	## Simple Points ##
	for x in range(padding,Ilabel.shape[0]-padding):
		for y in range(padding,Ilabel.shape[1]-padding):
			for z in range(padding,Ilabel.shape[2]-padding):
				simpleMap[x,y,z,:] = isSimple6_26(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2],topologyList,totalLabel)

	## Benefit Map ##
	pts = np.where(simpleMap==1)
	if metric == 'similarity':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],Sref[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric == 'distanceMap':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]]]-distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]]

	return simpleMap,benefitMap


def initAtlasPos6_26(Ilabel,Sref,atlasPrior,distanceMap,topologyList,topologyAtlasList,totalLabel,metric,padding):
	'''
    Compute the initialization for omega>=0 (i.e. Ilabel's size is lower
	than Sref's) considering the object and the background in 6-adjancy and
	26-adjancy respectly and providing an atlas prior that relax locally
	the topological constraints.
    -------------------------------
    Input:  Ilabel 				--> Image of labels
            Sref 				--> Segmentation of reference
			atlasPrior			--> Binary atlas defining areas where the
									topological relaxation will be applied
			distanceMap			--> Distance transform map
			topologyList		--> Configuration of label topologies to be preserved
			topologyAtlasList	--> Configuration of label topologies to be preserved
									in areas with topological relaxation
			totalLabel			--> Number of labels
			metric				--> Metric for the cost function
			padding				--> Padding for avoiding border effect
    Output: simpleMap    --> Simple point map
			benefitMap   --> Benefit map
    '''
	simpleMap=np.zeros(Sref.shape,dtype=int)
	benefitMap=np.zeros(Sref.shape)

	## Simple Points ##
	for x in range(padding,Ilabel.shape[0]-padding):
		for y in range(padding,Ilabel.shape[1]-padding):
			for z in range(padding,Ilabel.shape[2]-padding):
				if atlasPrior[x,y,z] == 0:
					simpleMap[x,y,z,:] = isSimple6_26(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2],topologyList,totalLabel)
				else:
					simpleMap[x,y,z,:] = isSimple6_26(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2],topologyAtlasList,totalLabel)

	## Benefit Map ##
	pts = np.where(simpleMap==1)
	if metric == 'similarity':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],Sref[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric == 'distanceMap':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]]]-distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]]

	return simpleMap,benefitMap


def initPos26_6(Ilabel,Sref,distanceMap,topologyList,totalLabel,metric,padding):
	'''
    Compute the initialization for omega>=0 (i.e. Ilabel's size is lower
	than Sref's) considering the object and the background in 26-adjancy and
	6-adjancy respectly.
    -------------------------------
    Input:  Ilabel 			--> Image of labels
            Sref 			--> Segmentation of reference
			distanceMap		--> Distance transform map
			topologyList	--> Configuration of label topologies to be preserved
			totalLabel		--> Number of labels
			metric			--> Metric for the cost function
			padding			--> Padding for avoiding border effect
    Output: simpleMap    --> Simple point map
			benefitMap   --> Benefit map
    '''
	simpleMap = np.zeros(Sref.shape,dtype=int)
	benefitMap =np.zeros(Sref.shape)

	## Simple Points ##
	for x in range(padding,Ilabel.shape[0]-padding):
		for y in range(padding,Ilabel.shape[1]-padding):
			for z in range(padding,Ilabel.shape[2]-padding):
				simpleMap[x,y,z,:] = isSimple26_6(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2],topologyList,totalLabel)

	## Benefit Map ##
	pts = np.where(simpleMap==1)
	if metric == 'similarity':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],Sref[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric == 'distanceMap':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]]]-distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]]

	return simpleMap,benefitMap

def initAtlasPos26_6(Ilabel,Sref,atlasPrior,distanceMap,topologyList,topologyAtlasList,totalLabel,metric,padding):
	'''
    Compute the initialization for omega>=0 (i.e. Ilabel's size is lower
	than Sref's) considering the object and the background in 26-adjancy and
	6-adjancy respectly and providing an atlas prior that relax locally
	the topological constraints.
    -------------------------------
    Input:  Ilabel 				--> Image of labels
            Sref 				--> Segmentation of reference
			atlasPrior			--> Binary atlas defining areas where the
									topological relaxation will be applied
			distanceMap			--> Distance transform map
			topologyList		--> Configuration of label topologies to be preserved
			topologyAtlasList	--> Configuration of label topologies to be preserved
									in areas with topological relaxation
			totalLabel			--> Number of labels
			metric				--> Metric for the cost function
			padding				--> Padding for avoiding border effect
    Output: simpleMap    --> Simple point map
			benefitMap   --> Benefit map
    '''
	simpleMap=np.zeros(Sref.shape,dtype=int)
	benefitMap=np.zeros(Sref.shape)

	## Simple Points ##
	for x in range(padding,Ilabel.shape[0]-padding):
		for y in range(padding,Ilabel.shape[1]-padding):
			for z in range(padding,Ilabel.shape[2]-padding):
				if atlasPrior[x,y,z] == 0:
					simpleMap[x,y,z,:] = isSimple26_6(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2],topologyList,totalLabel) ###NEW###
				else:
					simpleMap[x,y,z,:] = isSimple26_6(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2],topologyAtlasList,totalLabel)

	## Benefit Map ##
	pts = np.where(simpleMap==1)
	if metric == 'similarity':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],Sref[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric == 'distanceMap':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]]]-distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]]

	return simpleMap,benefitMap



def initRegPos(Ilabel,Sref,distanceMap,topologyList,totalLabel,metric,padding):
	'''
    Compute the initialization for omega>=0 (i.e. Ilabel's size is lower
	than Sref's) considering the object and the background in 26-adjancy and
	6-adjancy respectly, and a regularization term wrt the local labelling
	for the banefit calculation.
    -------------------------------
    Input:  Ilabel 			--> Image of labels
            Sref 			--> Segmentation of reference
			distanceMap		--> Distance transform map
			topologyList	--> Configuration of label topologies to be preserved
			totalLabel		--> Number of labels
			metric			--> Metric for the cost function
			padding			--> Padding for avoiding border effect
    Output: simpleMap    --> Simple point map
			benefitMap   --> Benefit map
    '''
	simpleMap = np.zeros(Sref.shape,dtype=int)
	benefitMap = np.zeros(Sref.shape)

	## Simple Points ##
	for x in range(padding,Ilabel.shape[0]-padding):
		for y in range(padding,Ilabel.shape[1]-padding):
			for z in range(padding,Ilabel.shape[2]-padding):
				simpleMap[x,y,z,:] = isSimple26_6(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2],topologyList,totalLabel) ###NEW###

	## Benefit Map ##
	pts = np.where(simpleMap==1)
	if metric == 'similarity':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],Sref[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric == 'distanceMap':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]]]-distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]]
			#regularisation#
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] *= (np.count_nonzero(Ilabel[pts[0][ipt]-1:pts[0][ipt]+2,pts[1][ipt]-1:pts[1][ipt]+2,pts[2][ipt]-1:pts[2][ipt]+2]==pts[3][ipt])+1.)/27.

	return simpleMap,benefitMap

def initAtlasRegPos(Ilabel,Sref,atlasPrior,distanceMap,topologyList,topologyAtlasList,totalLabel,metric,padding):
	'''
    Compute the initialization for omega>=0 (i.e. Ilabel's size is lower
	than Sref's) considering the object and the background in 26-adjancy and
	6-adjancy respectly and providing an atlas prior that relax locally
	the topological constraints. In addition, a regularization term is provided
	wrt the local labelling	for the banefit calculation.
    -------------------------------
    Input:  Ilabel 				--> Image of labels
            Sref 				--> Segmentation of reference
			atlasPrior			--> Binary atlas defining areas where the
									topological relaxation will be applied
			distanceMap			--> Distance transform map
			topologyList		--> Configuration of label topologies to be preserved
			topologyAtlasList	--> Configuration of label topologies to be preserved
									in areas with topological relaxation
			totalLabel			--> Number of labels
			metric				--> Metric for the cost function
			padding				--> Padding for avoiding border effect
    Output: simpleMap    --> Simple point map
			benefitMap   --> Benefit map
    '''
	simpleMap = np.zeros(Sref.shape,dtype=int)
	benefitMap = np.zeros(Sref.shape)

	## Simple Points ##
	for x in range(padding,Ilabel.shape[0]-padding):
		for y in range(padding,Ilabel.shape[1]-padding):
			for z in range(padding,Ilabel.shape[2]-padding):
				if atlasPrior[x,y,z] == 0:
					simpleMap[x,y,z,:] = isSimple26_6(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2],topologyList,totalLabel) ###NEW###
				else:
					simpleMap[x,y,z,:] = isSimple26_6(Ilabel[x-1:x+2,y-1:y+2,z-1:z+2],topologyAtlasList,totalLabel)

	## Benefit Map ##
	pts = np.where(simpleMap==1)
	if metric == 'similarity':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],Sref[pts[0][ipt],pts[1][ipt],pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric == 'distanceMap':
		for ipt in range(len(pts[0])):
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]]]-distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]]
			#regularisation#
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] *= (np.count_nonzero(Ilabel[pts[0][ipt]-1:pts[0][ipt]+2,pts[1][ipt]-1:pts[1][ipt]+2,pts[2][ipt]-1:pts[2][ipt]+2]==pts[3][ipt])+1.)/27.

	return simpleMap,benefitMap




'''
############################
####### Simple Pairs #######
############################
'''

def initSimplePairPosConn(Ilabel,Sref,simpleMap,distanceMap,topologyList,connectivityList,totalLabel,metric,padding=1):
	'''
    Compute the initialization considering simple pairs for omega>=0 (i.e.
	Ilabel's size is lower than Sref's). This function allows to select the
	connectivity between object and background.
    -------------------------------
    Input:  Ilabel 				--> Image of labels
            Sref 				--> Segmentation of reference
			simpleMap			--> Simple point map
			distanceMap			--> Distance transform map
			topologyList		--> Configuration of label topologies to be preserved
			connectivityList	-->	Configuration of label connectivity (6 or 26)
			totalLabel			--> Number of labels
			metric				--> Metric for the cost function
			padding				--> Padding for avoiding border effect
    Output: simplePairMap    --> Simple pair map
			benefitPairMap   --> Benefit map of the simple pairs
    '''
	simplePairMap = np.zeros((Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],3,totalLabel))
	benefitPairMap = np.zeros((Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],3,totalLabel))

	## Simple Pairs ##
	##Binary mask avoiding homogeneous ares, simple points and padding##
	homogeneous = computeHomogeneousArea(Ilabel,totalLabel)
	paddingImage = np.ones_like(Ilabel)
	paddingImage[padding:Ilabel.shape[0]-padding,padding:Ilabel.shape[1]-padding,padding:Ilabel.shape[2]-padding]=0
	for l in range(totalLabel):
		maskIni = 1-1*((simpleMap[:,:,:,l] + homogeneous + paddingImage)!=0)
		for ll in range(totalLabel):
			if ll!=l:
				mask = maskIni*(Ilabel==ll)

				##Simple pair in z-axis##
				maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0,0,0]],[[1,-1,0]],[[0,0,0]]]))==0)
				ct = np.where(maskFiltered==1)
				for ict in range(len(ct[0])):
					simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],2,l] = isSimplePairConnLabel( \
												Ilabel[ct[0][ict]-1:ct[0][ict]+2,ct[1][ict]-1:ct[1][ict]+2,ct[2][ict]-1:ct[2][ict]+3],\
												([1,1],[1,1],[1,2]),l,topologyList,connectivityList,totalLabel )
				##Simple pair in y-axis##
				maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0],[0],[0]],[[1],[-1],[0]],[[0],[0],[0]]]))==0)
				ct = np.where(maskFiltered==1)
				for ict in range(len(ct[0])):
					simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],1,l] = isSimplePairConnLabel( \
												Ilabel[ct[0][ict]-1:ct[0][ict]+2,ct[1][ict]-1:ct[1][ict]+3,ct[2][ict]-1:ct[2][ict]+2],\
												([1,1],[1,2],[1,1]),l,topologyList,connectivityList,totalLabel )
				##Simple pair in x-axis##
				maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0],[1],[0]],[[0],[-1],[0]],[[0],[0],[0]]]))==0)
				ct = np.where(maskFiltered==1)
				for ict in range(len(ct[0])):
					simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],0,l] = isSimplePairConnLabel( \
												Ilabel[ct[0][ict]-1:ct[0][ict]+3,ct[1][ict]-1:ct[1][ict]+2,ct[2][ict]-1:ct[2][ict]+2],\
												([1,2],[1,1],[1,1]),l,topologyList,connectivityList,totalLabel )

	## Benefit Map ##
	spm = np.where(simplePairMap[:,:,:,2,:] == 1)
	if metric == 'similarity':
		for ispm in range(len(spm[0])):
			benefitPairMap[spm[0][ispm],spm[1][ispm],spm[2][ispm],2,spm[3][ispm]] = computeBenefitPairPosSimilarity(Ilabel[spm[0][ispm],spm[1][ispm],spm[2][ispm]],Sref[spm[0][ispm],spm[1][ispm],\
																												spm[2][ispm],:],spm[3][ispm],totalLabel)
	elif metric == 'distanceMap':
		for ispm in range(len(spm[0])):
			benefitPairMap[spm[0][ispm],spm[1][ispm],spm[2][ispm],2,spm[3][ispm]] = computeBenefitPairPosDistance(Ilabel[spm[0][ispm],spm[1][ispm],spm[2][ispm]],\
																						distanceMap[spm[0][ispm],spm[1][ispm],spm[2][ispm]:spm[2][ispm]+2,:],spm[3][ispm])
	spm = np.where(simplePairMap[:,:,:,1,:] == 1)
	if metric == 'similarity':
		for ispm in range(len(spm[0])):
			benefitPairMap[spm[0][ispm],spm[1][ispm],spm[2][ispm],1,spm[3][ispm]] = computeBenefitPairPosSimilarity(Ilabel[spm[0][ispm],spm[1][ispm],spm[2][ispm]],Sref[spm[0][ispm],spm[1][ispm],\
																												spm[2][ispm],:],spm[3][ispm],totalLabel)
	elif metric == 'distanceMap':
		for ispm in range(len(spm[0])):
			benefitPairMap[spm[0][ispm],spm[1][ispm],spm[2][ispm],1,spm[3][ispm]] = computeBenefitPairPosDistance(Ilabel[spm[0][ispm],spm[1][ispm],spm[2][ispm]],\
																						distanceMap[spm[0][ispm],spm[1][ispm]:spm[1][ispm]+2,spm[2][ispm],:],spm[3][ispm])
	spm = np.where(simplePairMap[:,:,:,0,:] == 1)
	if metric == 'similarity':
		for ispm in range(len(spm[0])):
			benefitPairMap[spm[0][ispm],spm[1][ispm],spm[2][ispm],0,spm[3][ispm]] = computeBenefitPairPosSimilarity(Ilabel[spm[0][ispm],spm[1][ispm],spm[2][ispm]],Sref[spm[0][ispm],spm[1][ispm],\
																												spm[2][ispm],:],spm[3][ispm],totalLabel)
	elif metric == 'distanceMap':
		for ispm in range(len(spm[0])):
			benefitPairMap[spm[0][ispm],spm[1][ispm],spm[2][ispm],0,spm[3][ispm]] = computeBenefitPairPosDistance(Ilabel[spm[0][ispm],spm[1][ispm],spm[2][ispm]],\
																						distanceMap[spm[0][ispm]:spm[0][ispm]+2,spm[1][ispm],spm[2][ispm],:],spm[3][ispm])
	return simplePairMap,benefitPairMap


def initSimplePairAtlasPosConn(Ilabel,Sref,simpleMap,atlasPrior,distanceMap,topologyList,topologyAtlasList,totalLabel,metric,padding=1):
	'''
    Compute the initialization considering simple pairs for omega>=0 (i.e.
	Ilabel's size is lower than Sref's). This function allows to select the
	connectivity between object and background. An atlas prior is used to lead
	the topological	relaxation locally.
    -------------------------------
    Input:  Ilabel 					--> Image of labels
            Sref 					--> Segmentation of reference
			simpleMap				--> Simple point map
			atlasPrior				--> Binary atlas defining areas where the
										topological relaxation will be applied
			distanceMap				--> Distance transform map
			topologyList			--> Configuration of label topologies to be
										preserved
			topologyAtlasList		--> Configuration of label topologies to be
										preserved in areas with topological relaxation
			connectivityList		-->	Configuration of label connectivity (6 or 26)
			connectivityAtlasList	-->	Configuration of label connectivity (6 or 26)
										for the topological relaxation
			totalLabel				--> Number of labels
			metric					--> Metric for the cost function
			padding					--> Padding for avoiding border effect
    Output: simplePairMap    --> Simple pair map
			benefitPairMap   --> Benefit map of the simple pairs
    '''
	simplePairMap = np.zeros((Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],3,totalLabel))
	benefitPairMap = np.zeros((Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],3,totalLabel))

	## Simple Pairs ##
	##Binary mask avoiding homogeneous ares, simple points and padding##
	homogeneous = computeHomogeneousArea(Ilabel,totalLabel)
	paddingImage = np.ones_like(Ilabel)
	paddingImage[padding:Ilabel.shape[0]-padding,padding:Ilabel.shape[1]-padding,padding:Ilabel.shape[2]-padding]=0
	for l in range(totalLabel):
		##Topological atlas##
		mask = 1*((simpleMap[:,:,:,l] + homogeneous + paddingImage)!=0)
		mask = (1*(atlasPrior!=0))*(1-mask)
		##Simple pair in z-axis##
		maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0,0,0]],[[1,-1,0]],[[0,0,0]]]))==0)
		ct = np.where(maskFiltered==1)
		for ict in range(len(ct[0])):
			simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],2,l] = isSimplePairConnLabel( Ilabel[ct[0][ict]-1:ct[0][ict]+2,ct[1][ict]-1:ct[1][ict]+2,ct[2][ict]-1:ct[2][ict]+3],\
																							([1,1],[1,1],[1,2]),l,topologyAtlasList,connectivityList,totalLabel )
		##Simple pair in y-axis##
		maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0],[0],[0]],[[1],[-1],[0]],[[0],[0],[0]]]))==0)
		ct = np.where(maskFiltered==1)
		for ict in range(len(ct[0])):
			simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],1,l] = isSimplePairConnLabel( Ilabel[ct[0][ict]-1:ct[0][ict]+2,ct[1][ict]-1:ct[1][ict]+3,ct[2][ict]-1:ct[2][ict]+2],\
																							([1,1],[1,2],[1,1]),l,topologyAtlasList,connectivityList,totalLabel )
		##Simple pair in x-axis##
		maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0],[1],[0]],[[0],[-1],[0]],[[0],[0],[0]]]))==0)
		ct = np.where(maskFiltered==1)
		for ict in range(len(ct[0])):
			simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],0,l] = isSimplePairConnLabel( Ilabel[ct[0][ict]-1:ct[0][ict]+3,ct[1][ict]-1:ct[1][ict]+2,ct[2][ict]-1:ct[2][ict]+2],\
																							([1,2],[1,1],[1,1]),l,topologyAtlasList,connectivityList,totalLabel )
		print('Label '+str(l)+', atlas: \tmask '+str(np.count_nonzero(mask))+' and filtered '+str(np.count_nonzero(maskFiltered)))
		print('Simple pairs in 0 '+str(np.count_nonzero(simplePairMap[:,:,:,:,0])))
		print('Simple pairs in 1 '+str(np.count_nonzero(simplePairMap[:,:,:,:,1])))
		print('Simple pairs in 2 '+str(np.count_nonzero(simplePairMap[:,:,:,:,2])))
		##Rest of Ilabel##
		mask = 1*((simpleMap[:,:,:,l] + homogeneous + paddingImage)!=0)
		mask = (1*(atlasPrior==0))*(1-mask)
		##Simple pair in x-axis##
		maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0,0,0]],[[1,-1,0]],[[0,0,0]]]))==0)
		ct = np.where(maskFiltered==1)
		for ict in range(len(ct[0])):
			simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],2,l] = isSimplePair6_26Label( Ilabel[ct[0][ict]-1:ct[0][ict]+2,ct[1][ict]-1:ct[1][ict]+2,ct[2][ict]-1:ct[2][ict]+3],\
																							([1,1],[1,1],[1,2]),l,topologyList,totalLabel )
		##Simple pair in y-axis##
		maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0],[0],[0]],[[1],[-1],[0]],[[0],[0],[0]]]))==0)
		ct = np.where(maskFiltered==1)
		for ict in range(len(ct[0])):
			simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],1,l] = isSimplePair6_26Label( Ilabel[ct[0][ict]-1:ct[0][ict]+2,ct[1][ict]-1:ct[1][ict]+3,ct[2][ict]-1:ct[2][ict]+2],\
																							([1,1],[1,2],[1,1]),l,topologyList,totalLabel )
		##Simple pair in z-axis##
		maskFiltered = mask*(ndimage.convolve(mask,np.array([[[0],[1],[0]],[[0],[-1],[0]],[[0],[0],[0]]]))==0)
		ct = np.where(maskFiltered==1)
		for ict in range(len(ct[0])):
			simplePairMap[ct[0][ict],ct[1][ict],ct[2][ict],0,l] = isSimplePair6_26Label( Ilabel[ct[0][ict]-1:ct[0][ict]+3,ct[1][ict]-1:ct[1][ict]+2,ct[2][ict]-1:ct[2][ict]+2],\
																							([1,2],[1,1],[1,1]),l,topologyList,totalLabel )
		print('Label '+str(l)+', noAtlas: \tmask '+str(np.count_nonzero(mask))+' and filtered '+str(np.count_nonzero(maskFiltered)))
		print('Simple pairs in 0 '+str(np.count_nonzero(simplePairMap[:,:,:,:,0])))
		print('Simple pairs in 1 '+str(np.count_nonzero(simplePairMap[:,:,:,:,1])))
		print('Simple pairs in 2 '+str(np.count_nonzero(simplePairMap[:,:,:,:,2])))

	print('Simple pairs '+str(np.count_nonzero(simplePairMap)))

	## Benefit Map ##
	pts = np.where(simplePairMap[:,:,:,2,:] == 1)
	if metric == 'similarity':
		for ipt in range(len(pts[0])):
			benefitPairMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],2,pts[3][ipt]] = computeBenefitPairPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],\
																					Sref[pts[0][ipt],pts[1][ipt],pts[2][ipt]:pts[2][ipt]+2,:],pts[3][ipt],totalLabel)
	elif metric == 'distanceMap':
		for ipt in range(len(pts[0])):
			benefitPairMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],2,pts[3][ipt]] = computeBenefitPairPosDistance(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],\
																					distanceMap[pts[0][ipt],pts[1][ipt],pts[2][ipt]:pts[2][ipt]+2,:],pts[3][ipt])

	pts = np.where(simplePairMap[:,:,:,1,:] == 1)
	if metric == 'similarity':
		for ipt in range(len(pts[0])):
			benefitPairMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],1,pts[3][ipt]] = computeBenefitPairPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],\
																					Sref[pts[0][ipt],pts[1][ipt]:pts[1][ipt]+2,pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric == 'distanceMap':
		for ipt in range(len(pts[0])):
			benefitPairMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],1,pts[3][ipt]] = computeBenefitPairPosDistance(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],\
																					distanceMap[pts[0][ipt],pts[1][ipt]:pts[1][ipt]+2,pts[2][ipt],:],pts[3][ipt])

	pts = np.where(simplePairMap[:,:,:,0,:] == 1)
	if metric == 'similarity':
		for ipt in range(len(pts[0])):
			benefitPairMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],0,pts[3][ipt]] = computeBenefitPairPosSimilarity(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],\
																					Sref[pts[0][ipt]:pts[0][ipt]+2,pts[1][ipt],pts[2][ipt],:],pts[3][ipt],totalLabel)
	elif metric == 'distanceMap':
		for ipt in range(len(pts[0])):
			benefitPairMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],0,pts[3][ipt]] = computeBenefitPairPosDistance(Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]],\
																					distanceMap[pts[0][ipt]:pts[0][ipt]+2,pts[1][ipt],pts[2][ipt],:],pts[3][ipt])

	return simplePairMap,benefitPairMap



def computeCostMap(Ilabel,Sref,distanceMap,totalLabel,metrics,padding=1):
	'''
    Compute the cost map for omega>=0 (i.e. Ilabel's size is lower than Sref's).
	This function can provide several cost at the same time.
    -------------------------------
    Input:  Ilabel 				--> Image of labels
            Sref 				--> Segmentation of reference
			distanceMap			--> Distance transform map
			totalLabel			--> Number of labels
			metrics				--> Array with metrics for the cost function
			padding				--> Padding for avoiding border effect
    Output: costMap   --> Cost map (Ilabel_size x nb_metrics)
    '''
	costMap = np.zeros(np.concatenate((Sref.shape,[len(metrics)])))
	index = 0
	if 'similarity' in metrics:
		for l in range(totalLabel):
			tmp = np.zeros(Sref.shape)
			tmp[:,:,:,l] = 1*(Ilabel==l)
			costMap[:,:,:,l,index] = np.linalg.norm(tmp-Sref,axis=3)
		index += 1
	if 'distanceMap' in metrics:
		for l in range(totalLabel):
			costMap[:,:,:,l,index] = distanceMap[:,:,:,l]*(Ilabel==l)
		index += 1
	if 'geometry' in metrics:
		costMap[:,:,:,:,index] = 0
		index += 1

	return  costMap



'''##########################################'''
'''####				NEGATIVE			 ####'''
'''##########################################'''


def initNegConn(Ilabel, Sref, distanceMap, topologyList, connectivityList, totalLabel, metric, padding, omega):
	'''
    Compute the initialization for omega<0 (i.e. Ilabel's size is bigger
	than Sref's). This function allows to select the connectivity between
	object and background.
    -------------------------------
    Input:  Ilabel 				--> Image of labels
            Sref 				--> Segmentation of reference
			distanceMap			--> Distance transform map
			topologyList		--> Configuration of label topologies to be preserved
			connectivityList	-->	Configuration of label connectivity (6 or 26)
			totalLabel			--> Number of labels
			metric				--> Metric for the cost function
			padding				--> Padding for avoiding border effect
			omega				--> Factor of upsampling, 2^{omega} (omega is
									considered negative)
    Output: simpleMap    		--> Simple point map
			proportionLabelMap	-->	Label proportion map of Ilabel (Sref's size)
			benefitMap   		--> Benefit map
    '''
	simpleMap = np.zeros([Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],totalLabel],dtype=int)
	proportionLabelMap = np.zeros(Sref.shape)
	benefitMap = np.zeros([Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],totalLabel])

	resolutionGap = int((2**abs(omega))**3)

	## Simple Points ##
	##Binary mask avoiding homogeneous ares, simple points and padding##
	heterogeneous = np.zeros([Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],totalLabel])
	paddingImage = np.zeros_like(Ilabel)
	paddingImage[padding*(2**abs(omega)):Ilabel.shape[0]-padding*(2**abs(omega)),padding*(2**abs(omega)):Ilabel.shape[1]-padding*(2**abs(omega)),padding*(2**abs(omega)):Ilabel.shape[2]-padding*(2**abs(omega))] = 1
	for l in range(totalLabel):
		binIm = 1*(Ilabel==l)
		heterogeneous[:,:,:,l] = paddingImage*(1-1*(ndimage.convolve(binIm,np.array([[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,-26,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]))==0))

	pts = np.where(heterogeneous==1)
	x = chunkIt(pts[0],resolutionGap)
	y = chunkIt(pts[1],resolutionGap)
	z = chunkIt(pts[2],resolutionGap)

	for part in range(resolutionGap):
		for ipt in range(len(x[part])):
			simpleMap[x[part][ipt],y[part][ipt],z[part][ipt],:] = isSimpleConn(\
								Ilabel[	x[part][ipt]-1:x[part][ipt]+2,
										y[part][ipt]-1:y[part][ipt]+2,
										z[part][ipt]-1:z[part][ipt]+2],
								topologyList,connectivityList,totalLabel)

	for xref in range(padding,Sref.shape[0]-padding):
		for yref in range(padding,Sref.shape[1]-padding):
			for zref in range(padding,Sref.shape[2]-padding):
					proportionLabelMap[xref,yref,zref,:]=computeProportionLabel(Ilabel,xref,yref,zref,omega,totalLabel)

	## Benefit Map ##
	pts = np.where(simpleMap==1)
	print(str(len(pts[0]))+' simple points detected')
	if metric=='similarity':
		for ipt in range(len(pts[0])):
			xref,yref,zref = int(pts[0][ipt]*(2**omega)),int(pts[1][ipt]*(2**omega)),int(pts[2][ipt]*(2**omega))
			currentProportionIlabelPoint = proportionLabelMap[xref,yref,zref,:]
			newProportionIlabelPoint = currentProportionIlabelPoint.copy()
			newProportionIlabelPoint[ Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]] ] -= 1./resolutionGap #delete new element
			newProportionIlabelPoint[ pts[3][ipt] ] += 1./resolutionGap #add new element
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitNegSimilarity(currentProportionIlabelPoint,newProportionIlabelPoint,Sref[xref,yref,zref,:])
			#regularisation#
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] *= (np.count_nonzero(Ilabel[pts[0][ipt]-1:pts[0][ipt]+2,pts[1][ipt]-1:pts[1][ipt]+2,pts[2][ipt]-1:pts[2][ipt]+2]==pts[3][ipt])+1.)/27.
	elif metric=='distanceMap':
		for ipt in range(len(pts[0])):
			xref,yref,zref = int(pts[0][ipt]*(2**omega)),int(pts[1][ipt]*(2**omega)),int(pts[2][ipt]*(2**omega))
			currentProportionIlabelPoint = proportionLabelMap[xref,yref,zref,:]
			newProportionIlabelPoint = currentProportionIlabelPoint.copy()
			newProportionIlabelPoint[ Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]] ] -= 1./resolutionGap #delete new element
			newProportionIlabelPoint[ pts[3][ipt] ] += 1./resolutionGap #add new element
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitNegDistance(currentProportionIlabelPoint,newProportionIlabelPoint,distanceMap[xref,yref,zref,:])
			#regularisation#
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] *= (np.count_nonzero(Ilabel[pts[0][ipt]-1:pts[0][ipt]+2,pts[1][ipt]-1:pts[1][ipt]+2,pts[2][ipt]-1:pts[2][ipt]+2]==pts[3][ipt])+1.)/27.

	return simpleMap,proportionLabelMap,benefitMap


def initAtlasNeg(Ilabel,Sref,atlasPrior,distanceMap,topologyList,topologyAtlasList,connectivityList,connectivityAtlasList,totalLabel,metric,padding,omega):
	'''
    Compute the initialization for omega<0 (i.e. Ilabel's size is bigger
	than Sref's). This function allows to select the connectivity between
	object and background. An atlas prior is used to lead the topological
	relaxation locally.
    -------------------------------
    Input:  Ilabel 					--> Image of labels
            Sref 					--> Segmentation of reference
			atlasPrior				--> Binary atlas defining areas where the
										topological relaxation will be applied
			distanceMap				--> Distance transform map
			topologyList			--> Configuration of label topologies to be
										preserved
			topologyAtlasList		--> Configuration of label topologies to be
										preserved in areas with topological relaxation
			connectivityList		-->	Configuration of label connectivity (6 or 26)
			connectivityAtlasList	-->	Configuration of label connectivity (6 or 26)
										for the topological relaxation
			totalLabel				--> Number of labels
			metric					--> Metric for the cost function
			padding					--> Padding for avoiding border effect
			omega					--> Factor of upsampling, 2^{omega} (omega is
										considered negative)
    Output: simpleMap    		--> Simple point map
			proportionLabelMap	-->	Label proportion map of Ilabel (Sref's size)
			benefitMap  		 --> Benefit map
    '''
	simpleMap = np.zeros([Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],totalLabel],dtype=int)
	proportionLabelMap = np.zeros(Sref.shape)
	benefitMap = np.zeros([Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],totalLabel])

	resolutionGap = int((2**abs(omega))**3)

	## Simple Points ##
	##Binary mask avoiding homogeneous ares, simple points and padding##
	heterogeneous = np.zeros([Ilabel.shape[0],Ilabel.shape[1],Ilabel.shape[2],totalLabel])
	paddingImage = np.zeros_like(Ilabel)
	paddingImage[padding*(2**abs(omega)):Ilabel.shape[0]-padding*(2**abs(omega)),padding*(2**abs(omega)):Ilabel.shape[1]-padding*(2**abs(omega)),padding*(2**abs(omega)):Ilabel.shape[2]-padding*(2**abs(omega))] = 1
	for l in range(totalLabel):
		binIm = 1*(Ilabel==l)
		heterogeneous[:,:,:,l] = paddingImage*(1-1*(ndimage.convolve(binIm,np.array([[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,-26,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]))==0))

	pts = np.where(heterogeneous==1)
	print(str(len(pts[0]))+' of '+str(heterogeneous.size)+' --> '+str(100*len(pts[0])/heterogeneous.size)+' %')

	for ipt in range(len(pts[0])):
		if atlasPrior[pts[0][ipt],pts[1][ipt],pts[2][ipt]]==0:
			simpleMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],:]=isSimpleConn(Ilabel[pts[0][ipt]-1:pts[0][ipt]+2,pts[1][ipt]-1:pts[1][ipt]+2,pts[2][ipt]-1:pts[2][ipt]+2],topologyList,connectivityList,totalLabel)
		else:
			simpleMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],:]=isSimpleConn(Ilabel[pts[0][ipt]-1:pts[0][ipt]+2,pts[1][ipt]-1:pts[1][ipt]+2,pts[2][ipt]-1:pts[2][ipt]+2],topologyAtlasList,connectivityList,totalLabel)

	simpleMap = simpleMap[5:-5,5:-5,5:-5]

	for xref in range(padding,Sref.shape[0]-padding):
		for yref in range(padding,Sref.shape[1]-padding):
			for zref in range(padding,Sref.shape[2]-padding):
					proportionLabelMap[xref,yref,zref,:]=computeProportionLabel(Ilabel,xref,yref,zref,omega,totalLabel)

	## Benefit Map ##
	pts=np.where(simpleMap==1)
	if metric=='similarity':
		for ipt in range(len(pts[0])):
			xref,yref,zref = int(pts[0][ipt]*(2**omega)),int(pts[1][ipt]*(2**omega)),int(pts[2][ipt]*(2**omega))
			currentProportionIlabelPoint = proportionLabelMap[xref,yref,zref,:]
			newProportionIlabelPoint = currentProportionIlabelPoint.copy()
			newProportionIlabelPoint[ Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]] ] -= 1./resolutionGap #delete new element
			newProportionIlabelPoint[ pts[3][ipt] ] += 1./resolutionGap #add new element
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitNegSimilarity(currentProportionIlabelPoint,newProportionIlabelPoint,Sref[xref,yref,zref,:])
			#regularisation#
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] *= (np.count_nonzero(Ilabel[pts[0][ipt]-1:pts[0][ipt]+2,pts[1][ipt]-1:pts[1][ipt]+2,pts[2][ipt]-1:pts[2][ipt]+2]==pts[3][ipt])+1.)/27.
	elif metric=='distanceMap':
		for ipt in range(len(pts[0])):
			xref,yref,zref = int(pts[0][ipt]*(2**omega)),int(pts[1][ipt]*(2**omega)),int(pts[2][ipt]*(2**omega))
			currentProportionIlabelPoint = proportionLabelMap[xref,yref,zref,:]
			newProportionIlabelPoint = currentProportionIlabelPoint.copy()
			newProportionIlabelPoint[ Ilabel[pts[0][ipt],pts[1][ipt],pts[2][ipt]] ] -= 1./resolutionGap #delete new element
			newProportionIlabelPoint[ pts[3][ipt] ] += 1./resolutionGap #add new element
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] = computeBenefitNegDistance(currentProportionIlabelPoint,newProportionIlabelPoint,distanceMap[xref,yref,zref,:])
			#regularisation#
			benefitMap[pts[0][ipt],pts[1][ipt],pts[2][ipt],pts[3][ipt]] *= (np.count_nonzero(Ilabel[pts[0][ipt]-1:pts[0][ipt]+2,pts[1][ipt]-1:pts[1][ipt]+2,pts[2][ipt]-1:pts[2][ipt]+2]==pts[3][ipt])+1.)/27.

	return simpleMap,proportionLabelMap,benefitMap





def updateMapsNeg(Ilabel,Sref,simpleMap,proportionLabelMap,benefitMap,x,y,z,newLabel,topologyList,totalLabel,omega):
	neighbours = getNeighbourCoordonates(x,y,z)
	for nb in neighbours:
		## update simpleMap ##
		simpleMap[nb[0],nb[1],nb[2],:] = isSimple(Ilabel,topologyList,totalLabel,nb[0],nb[1],nb[2])
		## update proportionLabelMap ##
		xref,yref,zref = int(nb[0]*(2**omega)),int(nb[1]*(2**omega)),int(nb[2]*(2**omega))
		proportionLabelMap[xref,yref,zref,:] = computeProportionLabel(Ilabel,xref,yref,zref,omega,totalLabel)
		## update benefitMap ##
		for l in range(totalLabel):
			if simpleMap[nb[0],nb[1],nb[2],l] == 1:
				benefitMap[nb[0],nb[1],nb[2],l] = computeBenefitNeg(Ilabel,Sref,proportionLabelMap,x,y,z,l,totalLabel)
			else:
				benefitMap[nb[0],nb[1],nb[2],l] = 0

	return simpleMap,proportionLabelMap,benefitMap

def computeProportionLabel(Ilabel,xref,yref,zref,omega,totalLabel):
	'''
    Compute the label porportion of Ilabel given a particular Sref point, i.e.,
	assuming that Ilabel size is bigger than Sref (omage<0), it provides the
	percentage of each label in the equivalent Ilabel cube of the particular
	Sref point.
    -------------------------------
    Input:  Ilabel 				--> Image of labels
            xref,yref,zref 		--> Coordinates of a Sref point
			omega				--> Factor of upsampling, 2^{omega} (omega is
									considered negative)
			totalLabel			--> Number of labels
    Output: proprotionLabelMap	--> Label propotion map (2^-{omega} x 2^-{omega}
	 							  	x 2^-{omega} x totalLabel)
	'''
	proprotionLabelMap = np.zeros(totalLabel)
	im = Ilabel[int(xref/(2**omega)):int((xref+1)/(2**omega)),int(yref/(2**omega)):int((yref+1)/(2**omega)),int(zref/(2**omega)):int((zref+1)/(2**omega))]

	for l in range(totalLabel):
		proprotionLabelMap[l] = len(im[im==l])
	proprotionLabelMap /= im.size
	return proprotionLabelMap

def computeBenefitNeg(Ilabel,Sref,proportionLabelMap,x,y,z,newLabel,totalLabel):
	'''
    Compute the benefit of a point given a new label (different from current)
	for omega<0 (Ilabel's size bigger than Sref's).
    -------------------------------
    Input:  Ilabel 				--> Image of labels
			Sref				-->	Segmentation of reference
			proportionLabelMap	--> Label proportion map of Ilabel (Sref's size)
            x,y,z 				--> Coordinates of a Ilabel point
			newLabel			--> Label for computing benefit
			totalLabel			--> Number of labels
    Output: benefitNewLabel		--> Benefit of a point for the new label
	'''
	xref,yref,zref = int(x*(2**omega)),int(y*(2**omega)),int(z*(2**omega))
	newIlabel = Ilabel.copy()
	newIlabel[x,y,z] = newLabel
	nextProportionalLabelValue = computeProportionLabel(newIlabel,xref,yref,zref,omega,totalLabel)
	benefitNewLabel = np.linalg.norm(proportionLabelMap[xref,yref,zref,:]-Sref[xref,yref,zref,:])-np.linalg.norm(nextProportionalLabelValue-Sref[xref,yref,zref,:])
	return benefitNewLabel


def chunkIt(seq, num):
	'''
	Chunk a list in N equal lists (last element will adapt its length to
	compensate in case the total list length is not divisible by N)
    -------------------------------
	Input:	seq	--> Python list or 1D-array
			num	--> Number of list in the output (called N in the description)
	Output:	out	--> List of N lists containing the seq list
	'''
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def computeHomogeneousArea(im,totalLabel,kernelSize=5):
	'''
	Detect the homogeneous areas in an image of labels
    -------------------------------
	Input:	im			--> 3D image
			totalLabel	--> Number of labels
			kernelSize	-->	Kernel size for defining a homogeneous area
	Output:	homogeneousMap	--> 3D homogeneous map (1:homogeneous, 0:heterogeneous)
	'''
	homogeneousMap = np.zeros_like(im)
	for l in range(totalLabel):
		binIm = 1*(im==l)
		##Binary mask avoiding homogeneous ares, simple points and padding##
		homogeneousMap += 1*(ndimage.convolve(binIm,np.ones((kernelSize,kernelSize,kernelSize)))==kernelSize**3)
	homogeneousMap = 1*(homogeneousMap!=0)
	return homogeneousMap
