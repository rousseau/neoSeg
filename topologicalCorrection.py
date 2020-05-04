# coding=utf-8
import os,sys
import numpy as np
import nibabel
import random
import argparse
from scipy import ndimage
from time import time
from skimage import morphology, feature, measure
import skfmm

from utils.simplicity import isSimpleConn, isSimple26_6
from utils.sampling import changeVoxelSizeInAffineMatrix, upsampling3D, downsamplingMean3D, \
						downsamplingMean4D, downsamplingGauss3D, downsamplingGauss4D
from utils.distances import computeBenefitPosDistance, computeBenefitPairPosDistance, \
			computeBenefitPosGradientDistance, computeBenefitPosSimilarity, computeBenefitPairPosSimilarity, \
			computeGaussianDistanceMap, computeFastMarchingGaussianDistanceMap, computeDilationFastMarchingGaussianDistanceMap,\
			computeThicknessDistance, computeThicknessStructure, computeThicknessMap, computeFastMarchingGaussianErosionGMDistanceMap
from utils.concentricLabels import constructIlabelByCentroidSref, constructSref
from utils.initialisation import initNeg, initAtlasNeg, initPosConn, initAtlasPosConn, computeHomogeneousArea
from evaluation.topologicalTools import checkConnectedComponents, computeBettiNumbers








def updateMapsPos(IlabelPatch,SrefPatch,simpleMapPatch,benefitMapPatch,distanceMapPatch,topologyList,connectivityList,totalLabel,metric):
	'''
    Update the simple point and beneficit maps
    -------------------------------
    Input:  IlabelPatch 		--> Image of labels patch of size 4x4x4 vx
            SrefPatch 			--> Segmentation of reference patch of size 3x3x3x{totalLabel} vx
			simpleMapPatch		--> Simple point map patch of size 3x3x3x{totalLabel} vx
			benefitMapPatch		--> Benefit map patch of size 3x3x3 vx
			distanceMapPatch	--> Distance transform map patch of size 3x3x3 vx
			topologyList		--> Configuration of label topologies to be preserved
			connectivityList	--> Configuration of label connectivities (6 or 26)
			totalLabel			--> Number of labels
			metric				--> Metric for the cost function
    Output: simpleMapPatch	--> Simple point map patch updated
			benefitMapPatch --> Benefit map patch updated
    '''
	simpleMapPatch[:] = 0
	benefitMapPatch[:] = 0
	if metric == 'similarity':
		for ii in range(SrefPatch.shape[0]):
			for jj in range(SrefPatch.shape[1]):
				for kk in range(SrefPatch.shape[2]):
					## update simpleMap ##
					simpleMapPatch[ii,jj,kk,:] = isSimpleConn(IlabelPatch[ii:ii+3,jj:jj+3,kk:kk+3],topologyList,connectivityList,totalLabel) ###NEW###
					### update benefitMap ##
					for ll in range(totalLabel):
						if simpleMapPatch[ii,jj,kk,ll] == 1:
							benefitMapPatch[ii,jj,kk,ll] = computeBenefitPosSimilarity(IlabelPatch[ii+1,jj+1,kk+1],SrefPatch[ii,jj,kk,:],ll,totalLabel)
	elif metric == 'distanceMap':
		for ii in range(SrefPatch.shape[0]):
			for jj in range(SrefPatch.shape[1]):
				for kk in range(SrefPatch.shape[2]):
					## update simpleMap ##
					simpleMapPatch[ii,jj,kk,:] = isSimpleConn(IlabelPatch[ii:ii+3,jj:jj+3,kk:kk+3],topologyList,connectivityList,totalLabel) ###NEW###
					### update benefitMap ##
					for ll in range(totalLabel):
						if simpleMapPatch[ii,jj,kk,ll] == 1:
							benefitMapPatch[ii,jj,kk,ll] = distanceMapPatch[ii,jj,kk,IlabelPatch[ii+1,jj+1,kk+1]]-distanceMapPatch[ii,jj,kk,ll]

	return simpleMapPatch,benefitMapPatch







if __name__ == '__main__':
	'''
	This python script provide a topological correction given a set of segmentation
	maps (the segmentation of reference). The correction is based on a deformation
	by preserving the topology. It needs an image with a desired topology (image
	of labels) which is	going to deformated in order to be similar to the
	segmentation of reference. The method uses two strategies: the multilabel and
	the multiscale. As optional, there is a local relaxation of the topology
	provided by a prior map that allow to adapt the initial topology to the desired
	object topology.

	Example of utilisation in your terminal:

		python neoSeg/topologicalCorrection.py  -sref  neoSeg/examples/31_wm.nii.gz
			neoSeg/examples/31_ribbon_cortex.nii.gz -opath neoSeg/results
			-spath neoSeg/topology/intermediate -rp neoSeg/31_brainstem_drawem.nii.gz

		Note: A set of files for a demonstration is available in the repository
			'github.com/rousseau/neoSeg/examples'. In order to check obtained results, they are available in 'github.com/rousseau/neoSeg/results'
	'''


	t0 = time()
	np.seterr(divide='ignore', invalid='ignore')
	parser = argparse.ArgumentParser(prog='topologicalCorrection')

	parser.add_argument('-sref', '--Sreference', help='Segmentation of reference to be corrected (it can be given by a multilabel image or mutiple monolabel images) \
						(required)', type=str, nargs='*', required=True)
	parser.add_argument('-b', '--background', help='Add to the segmentation of reference a background as a complementary label (0: no, 1: yes)', type=int, required=False, default=1)
	parser.add_argument('-ini', '--initialization', help='Initialization of the deformable image with a correct topology (some default options for brain images are considered: \
						circleSmall, circleMedium, circleLarge, ellipseSmall, ellipseMedium, ellipseLarge, maskSref)', type=str, required=False, default='maskSref')
	parser.add_argument('-std', '--stdGaussian', help='Standard deviation of the Gaussian filter used for the downsampling', type=int, default=1, required=False)
	parser.add_argument('-m', '--metric', help='Metric for the cost function (\'distanceMap\' or \'similarity\')', default='distanceMap')
	parser.add_argument('-mdm', '--metricDistanceMap', help='[Option for \'distanceMap\' metric] Threshold for each label in order to build the distance map \
						(0: majority voting, >=0.5: one label per voxel, <0.5: may be multiple labels for a voxels)', type=float, default=0.5, required=False)
	parser.add_argument('-tl', '--topologyList', help='Configuration of label topologies to be preserved (required)', type=str, nargs='*', default=[[0],[1],[2]], required=False)
	parser.add_argument('-cl', '--connectivityList', help='Configuration of label connectivities (6 or 26) (required)', type=str, nargs='*', default=[6,6,26], required=False)
	parser.add_argument('-oi', '--omegaInitial', help='Initial value of omega', type=int, required=False, default=3)
	parser.add_argument('-of', '--omegaFinal', help='Final value of omega', type=int, required=False, default=0)
	parser.add_argument('-opath', '--outputPath', help='Output path', type=str, default='', required=False)
	parser.add_argument('-rp', '--relaxPrior', help='Binary image providing the spatial prior where the topological relaxation will be applied (\'no\' for avoid the local relaxation)', type=str, default='no', required=False)
	parser.add_argument('-ro', '--relaxOmega', help='Value of omega, which determines the scale, for the topological relaxation', type=int, default=1, required=False)
	parser.add_argument('-rtl', '--relaxTopologyList', help='Configuration of label topologies to be preserved for the topological relaxation', type=str, default=[[0],[1],[0,2],[1,2]], nargs='*', required=False)
	parser.add_argument('-rcl', '--relaxConnectivityList', help='Configuration of label connectivity (6 or 26) for the topological relaxation', type=str, default=[6,6,26,26], nargs='*', required=False)
	parser.add_argument('-ssv', '--stepSave', help='Save intermediate steps (0: no, 1: yes)', type=int, default=1, required=False)
	parser.add_argument('-sdef', '--stepDefine', help='Define the step (in number of iterations)', type=int, default=1000, required=False)
	parser.add_argument('-spath', '--stepPath', help='Step path', type=str, default='steps', required=False)
	parser.add_argument('-ba', '--base', help='Base parameter which increase the scale in each step', type=int, default=2, required=False)
	parser.add_argument('-p', '--padding', help='Padding parameter for avoiding issues in the simple point search', type=int, default=1, required=False)

	args = parser.parse_args()



	###--Load input data--###

	if len(args.topologyList)!=len(args.connectivityList):
		print('ERROR: Number of label topologies and connectivities have to be equal.')
		sys.exit()
	else:
		topologyList = args.topologyList
		connectivityList = args.connectivityList

	if args.omegaInitial<args.omegaFinal:
		print('ERROR: Initial omega must be greater than or equal to final omega.')
		sys.exit()
	else:
		omegaInitial = args.omegaInitial
		omegaFinal = args.omegaFinal


	if args.metric not in ['distanceMap', 'similarity']:
		print('ERROR: Metric selected is unknown (possible are \'distanceMap\' or \'similarity\').')
		sys.exit()
	else:
		metric = args.metric

	if metric == 'distanceMap':
		if (args.metricDistanceMap<0) | (args.metricDistanceMap>=1):
			print('ERROR: Threshold for building the distance maps must be positive and lower than 1.')
			sys.exit()
		else:
			metricDistanceMap = args.metricDistanceMap
			if metricDistanceMap==0:
				modeDistanceMap = 'majorityVoting'
			elif metricDistanceMap>=0.5:
				modeDistanceMap = 'threshold'
			else:
				modeDistanceMap = 'fuzzy'


	if args.base<=1:
		print('ERROR: Base parameter have to be greater than 1.')
		sys.exit()
	base = args.base

	if args.stdGaussian<1:
		print('ERROR: Standard deviation for Gaussian filter must be greater than 0.')
		sys.exit()
	else:
		stdGaussian = args.stdGaussian

	if args.padding < 1:
		print('ERROR: Padding parameter must be greater than 0.')
		sys.exit()
	else:
		padding = args.padding

	padding3D = [(padding,padding),(padding,padding),(padding,padding)]
	padding4D = [(padding,padding),(padding,padding),(padding,padding),(0,0)]




	if args.background not in [0,1]:
		print('ERROR: Background parameter must be 0 or 1.')
		sys.exit()
	else:
		background = args.background

	if args.stepSave not in [0,1]:
		print('ERROR: Stepsave parameter must be 0 or 1.')
		sys.exit()
	else:
		stepSave = args.stepSave



	if args.stepSave==1:
		try:
			os.system('mkdir -p '+args.stepPath)
		except:
			print('ERROR: Intermediate path not found')
			sys.exit()
		if args.stepDefine<=0:
			print('ERROR: A step must be bigger than 0 interations.')
			sys.exit()
		stepPath = args.stepPath
		stepDefine = args.stepDefine

	try:
		os.system('mkdir -p '+args.outputPath)
	except:
		print('ERROR: Output path not found')
		sys.exit()
	outputPath = args.outputPath

	try:
		Sreference = list()
		for strSreference in args.Sreference:
			Sreference.append( nibabel.load(strSreference).get_data()/np.max(nibabel.load(strSreference).get_data()) )
		header = nibabel.load(strSreference).affine
	except:
		print('ERROR: Segmentation of reference file not found.')
		sys.exit()


	## build Sref and Ilabel ##
	Sref = constructSref(Sreference,omegaInitial,addBackground=args.background,base=args.base)
	totalLabel = Sref.shape[-1]

	if args.initialization not in ['circleSmall', 'circleMedium', 'circleLarge', 'ellipseSmall', 'ellipseMedium', 'ellipseLarge', 'maskSref']:
		try:
			Ilabel = nibabel.load(args.initialization).get_data()
		except:
			print('ERROR: Initialization file not found.')
			sys.exit()
		initialization = 'definedInitialization'
	else:
		Ilabel = constructIlabelByCentroidSref(Sref,omegaInitial,base=base,mode=args.initialization)
		header = changeVoxelSizeInAffineMatrix(header,2**omegaInitial)
		initialization = args.initialization

	if args.stepSave == 1:
		nibabel.save(nibabel.Nifti1Image(Ilabel.astype(float),header),outputPath+'/Ilabel_'+str(totalLabel)+'labels_'+initialization+'_omega'+str(omegaInitial)+'.nii.gz')
		nibabel.save(nibabel.Nifti1Image(Sref,header),outputPath+'/Sref_'+str(totalLabel)+'labels_omega'+str(omegaInitial)+'.nii.gz')


	Ilabel = Ilabel.astype(int)




	if args.relaxPrior != 'no':
		try:
			relaxPriorMap = nibabel.load(args.relaxPrior).get_data()
		except:
			print('ERROR: Prior image for enable the relaxation not found.')
			sys.exit()
		if args.relaxOmega>args.omegaInitial:
			print('ERROR: Omega for topological relaxation must be lower than or equal to initial omega.')
			sys.exit()
		if args.relaxOmega<args.omegaFinal:
			print('ERROR: Omega for topological relaxation must be greater than or equal to final omega.')
			sys.exit()
		if len(args.relaxTopologyList)!=len(args.relaxConnectivityList):
			print('ERROR: Number of label topologies and connectivities for topological relaxation have to be equal.')
			sys.exit()
		relaxPrior = args.relaxPrior
		relaxOmega = args.relaxOmega
		relaxTopologyList = args.relaxTopologyList
		relaxConnectivityList = args.relaxConnectivityList

		tmp = relaxPriorMap.copy()
		relaxPriorMap = np.zeros((Sref.shape[0],Sref.shape[1],Sref.shape[2]))
		relaxPriorMap[:tmp.shape[0],:tmp.shape[1],:tmp.shape[2]] = tmp
		if relaxOmega!=args.omegaFinal:
			relaxPriorMap = downsamplingGauss3D(relaxPriorMap,relaxOmega,stdGaussian,base=base)
		relaxPriorMap = np.pad(relaxPriorMap,padding3D,mode='constant', constant_values=0)






	###--Print Input Information--###
	print('Segmentation of reference: ',str(args.Sreference))
	if background==1:
		print('\tadding background')
	print('Labels to be treated: '+str(totalLabel))
	print('Initialization: ',initialization)
	print('Metric for the cost function: ',metric)
	if metric=='distanceMap':
		print('Mode of threshold selected: ',modeDistanceMap)
	print('Standard deviation for Gaussian filter: ',str(stdGaussian))
	print('Omegas from ',str(omegaInitial),' to ',str(omegaFinal))
	print('Topology list: ',str(topologyList))
	print('Connectivity list: ',str(connectivityList))
	print('Output will be saved in ',outputPath)
	if relaxPrior!='no':
		print('Relaxation prior: ',relaxPrior,' \n\twill be applied in omega = ',str(relaxOmega),' with topology list: ',str(relaxTopologyList),' and connectivity list: ',str(relaxConnectivityList))
	if stepSave==1:
		print('Intermediate results will be saved every ',str(stepDefine),' iterations in ',stepPath)
	print('Base parameter: ',base)



	## Start the steps ##
	for omega in range(omegaFinal,omegaInitial+1)[::-1]:


		if omega>0:
			std = stdGaussian
		else:
			std = 0.0


		print('\n\nOMEGA = '+str(omega))
		header = changeVoxelSizeInAffineMatrix(header,2**omega)
		Ilabel = np.pad(Ilabel,padding3D,mode='constant', constant_values=0)

		## Downsampling Sref ##
		newSref = downsamplingMean4D(Sref,omega,base=base)

		newSref = np.pad(newSref,padding4D,mode='constant', constant_values=0)


		if Ilabel.size!=newSref[:,:,:,0].size:
			print('WARNING: Ilabel and Sref don\'t have the same size!!')
			print(Ilabel.size)
			print(Ilabel.shape)
			print(newSref[:,:,:,0].size)
			print(newSref[:,:,:,0].shape)

		if Ilabel.dtype!=np.dtype('int'):
			print('ERROR: Ilabel is not a integer numpy array!!')
			sys.exit()

		## Distance Maps ##
		distanceMap = computeFastMarchingGaussianDistanceMap(Sref,std,padding,omega,base,totalLabel,mode=modeDistanceMap,threshold=metricDistanceMap)

		loops = 0


		print('Topology list: '+str(topologyList))
		simpleMap,benefitMap = initPosConn(Ilabel,newSref,distanceMap,topologyList,connectivityList,totalLabel,metric,padding)


		print('\nConnected components of Ilabel = '+str(checkConnectedComponents(Ilabel,totalLabel,connectivityList)))


		t=time()
		loops = 0
		while(np.count_nonzero(benefitMap>0)>0):

			if (loops%stepDefine==0):
				print(str(loops)+' loops done \t\t '+str(np.count_nonzero(benefitMap>0))+' positive benefit points')
				tmp = Ilabel[padding:-padding,padding:-padding,padding:-padding]
				nibabel.save(nibabel.Nifti1Image(tmp.astype(float),header),stepPath+'/'+str(int(loops/stepDefine))+'iter_Ilabel_'+str(totalLabel)+'labels_'+initialization+'_omega'+str(omega)+'.nii.gz')

			pt = np.where(benefitMap == np.max(benefitMap))
			sel = random.randrange(len(pt[0]))
			## Update values ##
			Ilabel[pt[0][sel],pt[1][sel],pt[2][sel]]=pt[3][sel]
			simpleMap[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:], \
				benefitMap[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:] \
				= updateMapsPos(Ilabel[pt[0][sel]-2:pt[0][sel]+3,pt[1][sel]-2:pt[1][sel]+3,pt[2][sel]-2:pt[2][sel]+3], \
				newSref[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:], \
				simpleMap[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:], \
				benefitMap[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:], \
				distanceMap[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:], \
				topologyList,connectivityList,totalLabel,metric)

			loops += 1

		print('While time '+str(time()-t))

		Ilabel = Ilabel[padding:-padding,padding:-padding,padding:-padding]
		nibabel.save(nibabel.Nifti1Image(Ilabel.astype(np.int16),header),outputPath+'/Ilabel_'+str(totalLabel)+'labels_'+initialization+'_omega'+str(omega)+'_omegaIni'+str(omegaInitial)+'.nii.gz')

		print('\nConnected components of Ilabel = '+str(checkConnectedComponents(Ilabel,totalLabel,connectivityList)))
		print(computeBettiNumbers(Ilabel,totalLabel,connectivityList))
		print(str(loops)+' loops \n'+'\n')


		if (relaxPrior!='no') & (omega==relaxOmega):
			print('Local topological relaxation step')
			loops = 0
			Ilabel = np.pad(Ilabel,padding3D,mode='constant', constant_values=0)
			print('Topology list: '+str(relaxTopologyList))
			simpleMap,benefitMap = initAtlasPosConn(Ilabel,newSref,relaxPriorMap,distanceMap,topologyList,relaxTopologyList,connectivityList,relaxConnectivityList,totalLabel,metric,padding)
			while(np.count_nonzero(benefitMap>0)>0):

				if (loops%stepDefine==0):
					print(str(loops)+' loops done \t\t '+str(np.count_nonzero(benefitMap>0))+' positive benefit points')
					tmp = Ilabel[padding:-padding,padding:-padding,padding:-padding]
					nibabel.save(nibabel.Nifti1Image(tmp.astype(float),header),stepPath+'/'+str(int(loops/stepDefine))+'iter_Ilabel_'+str(totalLabel)+'labels_'+initialization+'_omega'+str(omega)+'_relaxMode.nii.gz')

				pt = np.where(benefitMap==np.max(benefitMap))
				sel = random.randrange(len(pt[0]))
				## Update values ##
				prevIlabel = Ilabel[pt[0][sel],pt[1][sel],pt[2][sel]]
				Ilabel[pt[0][sel],pt[1][sel],pt[2][sel]] = pt[3][sel]

				if relaxPriorMap[pt[0][sel],pt[1][sel],pt[2][sel]]==0:
					simpleMap[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:], \
						benefitMap[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:] \
						= updateMapsPos(Ilabel[pt[0][sel]-2:pt[0][sel]+3,pt[1][sel]-2:pt[1][sel]+3,pt[2][sel]-2:pt[2][sel]+3], \
						newSref[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:], \
						simpleMap[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:], \
						benefitMap[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:], \
						distanceMap[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:], \
						topologyList,connectivityList,totalLabel,metric)
				else:
					simpleMap[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:], \
						benefitMap[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:] \
						= updateMapsPos(Ilabel[pt[0][sel]-2:pt[0][sel]+3,pt[1][sel]-2:pt[1][sel]+3,pt[2][sel]-2:pt[2][sel]+3], \
						newSref[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:], \
						simpleMap[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:], \
						benefitMap[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:], \
						distanceMap[pt[0][sel]-1:pt[0][sel]+2,pt[1][sel]-1:pt[1][sel]+2,pt[2][sel]-1:pt[2][sel]+2,:], \
						relaxTopologyList,relaxConnectivityList,totalLabel,metric)

				loops += 1

			print('While time '+str(time()-t))

			Ilabel = Ilabel[padding:-padding,padding:-padding,padding:-padding]
			nibabel.save(nibabel.Nifti1Image(Ilabel.astype(np.int16),header),outputPath+'/Ilabel_'+str(totalLabel)+'labels_'+initialization+'_omega'+str(omega)+'_relaxMode_omegaIni'+str(omegaInitial)+'.nii.gz')

			print('\nConnected components of Ilabel = '+str(checkConnectedComponents(Ilabel,totalLabel,connectivityList)))
			print(computeBettiNumbers(Ilabel,totalLabel,connectivityList))
			print(str(loops)+' loops \n'+'\n')


		Ilabel = upsampling3D(Ilabel,base=base).astype(int)
	print('The correction finished in '+str(time()-t0))
