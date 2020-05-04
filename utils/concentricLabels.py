# coding: utf-8
import nibabel
import numpy as np
from scipy import ndimage
from skimage import measure
from sampling import downsamplingMean3D,downsamplingMean4D


def addBackgroundToSref(Sref):
    '''
	Add background class to a stack of segmentations (Sref). Its probability
    is computed wrt other segmentation probabilities.
	-------------------------------
    Input:  Sref       --> Segmentation of reference (stack of segmentations)
    Output: SrefWithBg --> Segmentation of reference with background as class
	'''
    SrefWithBg = np.zeros([Sref.shape[0],Sref.shape[1],Sref.shape[2],Sref.shape[3]+1])
    SrefWithBg[:,:,:,1:] = Sref
    SrefWithBg[:,:,:,0] = 1-np.sum(Sref,axis=-1)
    return SrefWithBg

def constructSref(segList,delta,addBackground=1):
    '''
	Construct the segmentation of reference given a list of segmentations
    (binary or fuzzy).
	-------------------------------
    Input:  SegList         --> Segmentation of reference (stack of segmentations)
            delta           --> Padding for making the size of the image disivible
                                by downsampling times preset by the strategy.
            addBackground   --> Add background class (1: yes, 0: no)
    Output: Sref    --> Segmentation of reference
	'''
    Sref = np.zeros([segList[0].shape[0]+delta[0],segList[0].shape[1]+delta[1],segList[0].shape[2]+delta[2],len(segList)])
    for l,S in enumerate(segList):
        Sref[:segList[0].shape[0],:segList[0].shape[1],:segList[0].shape[2],l]=S
    ##Normalise##
    Sref /= np.max(np.sum(Sref,axis=-1))
    Sref /= np.max(Sref)
    if addBackground==1:
        Sref = addBackgroundToSref(Sref)
    return Sref


def tracePerimeter(im):
    '''
    Chart the perimeter of a binary image.
	-------------------------------
    Input:  im           --> Binary image
    Output: perimeter    --> Perimeter of the binary image
    '''
    perimeter = ndimage.distance_transform_cdt(im,metric='taxicab')
    perimeter[perimeter>3] = 0
    perimeter[perimeter<1] = 0
    return perimeter

def getNeighbourCoordinates(x,y,z,sizeIm,connectivity):
    '''
    Get neighbor coordinates of a specific point.
	-------------------------------
    Input:  x,y,z           --> Coordinates of the target point
            sizeIm          --> Size of the 3D image where target point belongs
            connectivity    --> Connectivity of the target point wrt its
                                neighborhood (6 or 26)
    Output: neighCoord   --> List of coordinates of every neighbor
    '''
	minx = x-1 if x!=0 else 0
	miny = y-1 if y!=0 else 0
	minz = z-1 if z!=0 else 0
	maxx = x+2 if x!=sizeIm[0] else sizeIm[0]+1
	maxy = y+2 if y!=sizeIm[1] else sizeIm[1]+1
	maxz = z+2 if z!=sizeIm[2] else sizeIm[2]+1

	coordinates=list()
	for ii in range(minx,maxx):
		for jj in range(miny,maxy):
			for kk in range(minz,maxz):
				if connectivity==26:
					coordinates.append((ii,jj,kk))
				elif 1*(ii==x) + 1*(jj==y) + 1*(kk==z) >=2:
					coordinates.append((ii,jj,kk))
    neighCoord = list(set(coordinates))
	return neighCoord


def constructSyntheticIlabel(sizeRef,omega0,base=2):
    '''
    Construct the initialisation of the image of labels (Ilabel).
	-------------------------------
    Input:  sizeRef     --> Size of the segementation fo reference (Sref)
            omega0      --> Initial omega (parameter that determines the
                            relation between sizes of Sref and Ilabel)
            base        --> Base parameter of increase the scale in each step
    Output: Ilabel      --> Image of labels
            newSizeSref --> New size of Sref for being compatible with future
                            downsamplings (given by omega)
    '''
    newSizeSref=list(sizeRef)
    while (newSizeSref[0]%(base**omega0) != 0):
        newSizeSref[0]+=1
    while (newSizeSref[1]%(base**omega0) != 0):
        newSizeSref[1]+=1
    while (newSizeSref[2]%(base**omega0) != 0):
        newSizeSref[2]+=1
    sizeIlabel=[newSizeSref[0]/(base**omega0),newSizeSref[1]/(base**omega0),newSizeSref[2]/(base**omega0)]
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




def constructIlabelBySref(Sref,omega0,base=2):
    '''
    Conctruct an initialisation of image of labels from the shape of multiple
    segmentation maps (Sref).
	-------------------------------
    Input:  Sref    --> Segmentation of reference (a stack of segmentation maps)
            omega0  --> Initial omega (parameter that determines the relation
                        between sizes of Sref and Ilabel)
            base    --> Base parameter of increase the scale in each step
    Output: Ilabel  --> Image of labels
    '''
    newSizeSref=Sref.shape
    sizeIlabel=[newSizeSref[0]/(base**omega0),newSizeSref[1]/(base**omega0),newSizeSref[2]/(base**omega0)]
    Ilabel=np.zeros(sizeIlabel)

    newSref=downsamplingMean4D(Sref,omega)
    Ilabel[newSref[:,:,:,0]<0.5]=1

    ## Fill holes ##
    Ilabel = ndimage.morphology.binary_fill_holes(Ilabel)
    Ilabel = ndimage.morphology.binary_closing(Ilabel,iterations=5)
    Ilabel = Ilabel.astype(np.int)

    if omega0 <= 2:
        ## Perimeters ##
        csf = tracePerimeter(Ilabel)
        cortexWm = Ilabel.copy()
        cortexWm[csf==1] = 0
        cortex = tracePerimeter(cortexWm)
        wm = cortexWm.copy()
        wm[cortex==1] = 0

        Ilabel[csf==1] = 3
        Ilabel[cortex==1] = 2
        Ilabel[wm==1] = 1

    return Ilabel




def constructIlabelByCentroidSref(Sref,omega0,threshold=0.5,base=2,mode='circleMedium'):
    '''
    Conctruct an initialisation of image of labels with a particular shape which
    its radius is computed by the centroid of multiple segmentation maps (Sref).
	-------------------------------
    Input:  Sref        --> Segmentation of reference (a stack of segmentations)
            omega0      --> Initial omega (parameter that determines the relation
                            between sizes of Sref and Ilabel)
            threshold   --> Threshold for binarizing each segmentation of Sref
            base        --> Base parameter of increase the scale in each step
            mode        --> Initial shape of the image of labels. Possible values:
                            circleSmall, circleMedium, circleLarge, ellipseSmall,
                            ellipseMedium, ellipseLarge and maskSref
    Output: Ilabel  --> Image of labels
    '''
    newSizeSref = Sref.shape
    sizeIlabel = [int(newSizeSref[0]/(base**omega0)),int(newSizeSref[1]/(base**omega0)),int(newSizeSref[2]/(base**omega0))]
	Ilabel = np.zeros(sizeIlabel,dtype=int)

    ## Threshold the downsampled Sref##
    newSref = downsamplingMean4D(Sref,omega0,base)
    Ilabel[newSref[:,:,:,0]<threshold] = 1

    ## Fill holes ##
    Ilabel = ndimage.morphology.binary_fill_holes(Ilabel)

    ## Keep the biggest part ##
    labels,numLabels = measure.label(Ilabel,connectivity=1,return_num=True)
    max = 0
    for l in range(1,numLabels+1):
        if (len(np.where(labels==l)[0]) > max):
            max = len(np.where(labels==l)[0])
            label = l

    Ilabel[labels!=label] = 0
    Ilabel[labels==label] = 1
    Ilabel = Ilabel.astype(np.int16)

    centroid = ndimage.measurements.center_of_mass(Ilabel)
    centroid = [int(c) for c in centroid]
    x,y,z = np.ogrid[-centroid[0]:sizeIlabel[0]-centroid[0], -centroid[1]:sizeIlabel[1]-centroid[1], -centroid[2]:sizeIlabel[2]-centroid[2]]
	points=np.where(Ilabel==1)
	if mode == 'circleMedium':
		type = 'circle'
		rx,ry,rz =  int(np.min([centroid[0]-np.min(points[0]),np.max(points[0])-centroid[0]])),\
					int(np.min([centroid[1]-np.min(points[1]),np.max(points[1])-centroid[1]])),\
					int(np.min([centroid[2]-np.min(points[2]),np.max(points[2])-centroid[2]]))
	elif mode == 'circleSmall':
		type = 'circle'
		rx,ry,rz =  int(np.min([(centroid[0]-np.min(points[0]))/2,(np.max(points[0])-centroid[0])/2])),\
					int(np.min([(centroid[1]-np.min(points[1]))/2,(np.max(points[1])-centroid[1])/2])),\
					int(np.min([(centroid[2]-np.min(points[2]))/2,(np.max(points[2])-centroid[2])/2]))
	elif mode == 'circleLarge':
		type = 'circle'
		rx,ry,rz =  int(np.min([centroid[0]-2,sizeIlabel[0]-2-centroid[0]])),\
					int(np.min([centroid[1]-2,sizeIlabel[1]-2-centroid[1]])),\
					int(np.min([centroid[2]-2,sizeIlabel[2]-2-centroid[2]]))
	elif mode == 'ellipseMedium':
		type = 'ellipse'
		rx,ry,rz =  int(np.min([centroid[0]-np.min(points[0]),np.max(points[0])-centroid[0]])),\
					int(np.min([centroid[1]-np.min(points[1]),np.max(points[1])-centroid[1]])),\
					int(np.min([centroid[2]-np.min(points[2]),np.max(points[2])-centroid[2]]))
	elif mode == 'ellipseSmall':
		type = 'ellipse'
		rx,ry,rz =  int(np.min([(centroid[0]-np.min(points[0]))/2,(np.max(points[0])-centroid[0])/2])),\
					int(np.min([(centroid[1]-np.min(points[1]))/2,(np.max(points[1])-centroid[1])/2])),\
					int(np.min([(centroid[2]-np.min(points[2]))/2,(np.max(points[2])-centroid[2])/2]))
	elif mode == 'ellipseLarge':
		type = 'ellipse'
		rx,ry,rz =  int(np.min([centroid[0]-2,sizeIlabel[0]-2-centroid[0]])),\
					int(np.min([centroid[1]-2,sizeIlabel[1]-2-centroid[1]])),\
					int(np.min([centroid[2]-2,sizeIlabel[2]-2-centroid[2]]))
	else: #by mask Sref
		type = 'mask'
		newSref = downsamplingMean4D(Sref,omega0,base)
		for l in range(1,newSizeSref[3]):
			## Threshold the downsampled Sref##
			Ilabel[newSref[:,:,:,l]!=0] = 1
		## Fill holes ##
		Ilabel = ndimage.morphology.binary_fill_holes(Ilabel)
		iter = 0
		while ( measure.label(Ilabel,connectivity=1,return_num=True)[1] != 1):
			iter += 1
			Ilabel = ndimage.morphology.binary_closing(Ilabel,iterations=iter)
			Ilabel = ndimage.morphology.binary_fill_holes(Ilabel)
		Ilabel = Ilabel.astype(int)
		tmp = np.ones_like(Ilabel)
		tmp[1:-1,1:-1,1:-1] = 0
		Ilabel[tmp==1] = 0
		points=np.where(Ilabel==1)
		for ipt in range(len(points[0])):
			x,y,z = points[0][ipt],points[1][ipt],points[2][ipt]
			neighbours26 = getNeighbourCoordonates(x,y,z,Ilabel.shape,connectivity=26)
			if (len([nb for nb in neighbours26 if Ilabel[nb[0],nb[1],nb[2]]==0]) > 0):
				Ilabel[x,y,z] = 2

	if type == 'circle':
		r = int(np.min([rx,ry,rz]))
		r = 1 if r<=0 else r
		brain = x**2 + y**2 + z**2 <= r**2
		Ilabel = np.zeros(sizeIlabel).astype(int)
		Ilabel[brain] = 1
		tmp = np.ones_like(Ilabel)
		tmp[1:-1,1:-1,1:-1] = 0
		Ilabel[tmp==1] = 0
		points = np.where(Ilabel==1)
		for ipt in range(len(points[0])):
			x,y,z = points[0][ipt],points[1][ipt],points[2][ipt]
			neighbours26 = getNeighbourCoordonates(x,y,z,Ilabel.shape,connectivity=26)
			if (len([nb for nb in neighbours26 if Ilabel[nb[0],nb[1],nb[2]]==0]) > 0):
				Ilabel[x,y,z] = 2
	elif type == 'ellipse':
		Ilabel = np.zeros(sizeIlabel).astype(int)
		x,y,z = np.linspace(0,sizeIlabel[0]-1,sizeIlabel[0])[:,None,None], np.linspace(0,sizeIlabel[1]-1,sizeIlabel[1])[:,None], np.linspace(0,sizeIlabel[2]-1,sizeIlabel[2])
		Rx,Ry,Rz = np.min(rx), np.min(ry), np.min(rz)
		Rx = 1 if Rx<=0 else Rx
		Ry = 1 if Ry<=0 else Ry
		Rz = 1 if Rz<=0 else Rz
		mask = ((x-centroid[0])/Rx)**2 + ((y-centroid[1])/Ry)**2 + ((z-centroid[2])/Rz)**2 <= 1
		Ilabel[mask] = 1
		tmp = np.ones_like(Ilabel)
		tmp[1:-1,1:-1,1:-1] = 0
		Ilabel[tmp==1] = 0
		points=np.where(Ilabel==1)
		for ipt in range(len(points[0])):
			x,y,z = points[0][ipt],points[1][ipt],points[2][ipt]
			neighbours26 = getNeighbourCoordonates(x,y,z,Ilabel.shape,connectivity=26)
			if (len([nb for nb in neighbours26 if Ilabel[nb[0],nb[1],nb[2]]==0]) > 0):
				Ilabel[x,y,z] = 2

	return Ilabel



def generateConcentricInputs(labelsPath,opath,omega,threshold=0.5,base=2):
    '''
    Generate concentric labels as an initialisation of image of labels (Ilabel)
    and a stack of segmentations as segmentation of reference (Sref).
	-------------------------------
    Input:  labelsPath  --> Path where segmenetation maps of reference are located
            opath       --> Path where outputs will be saved
            omega       --> Factor of downsampling, 2^{omega} (here omega is
						    considered positive)
            threshold   --> Threshold for binarizing each segmentation of Sref
            base        --> Base parameter of increase the scale in each step
    Output: Ilabel  --> Image of labels
            Sref    --> Segmentation of reference
    '''
    header = nibabel.load(labelsPath[0]).affine
    S = list()
    for labPath in labelsPath:
        S.append(nibabel.load(labPath).get_data())
    sizeRef = S[0].shape

    newSizeSref = list(sizeRef)
    while (newSizeSref[0]%(base**omega) != 0):
		newSizeSref[0]+=1
    while (newSizeSref[1]%(base**omega) != 0):
    	newSizeSref[1]+=1
    while (newSizeSref[2]%(base**omega) != 0):
    	newSizeSref[2]+=1
    sizeIlabel = [newSizeSref[0]/(base**omega),newSizeSref[1]/(base**omega),newSizeSref[2]/(base**omega)]

    delta = [newSizeSref[0]-sizeRef[0],newSizeSref[1]-sizeRef[1],newSizeSref[2]-sizeRef[2]]

    Sref = constructSref(S,delta,1)

    Ilabel = constructIlabelByCentroidSref(Sref,omega,threshold,base=base)

    return Ilabel,Sref



def buildTestFarSpheres2D(sizeIm,numLabels,header,opath,omega):
    '''
    Build a test model with 2D concentrical spheres (i.e. concentrical disks).
	-------------------------------
    Input:  sizeIm  --> Image size
            numLabels   --> Number of labels
            header      --> Affine matrix for saving results as a Nifti image
            opath       --> Path where outputs will be saved
            omega       --> Factor of downsampling, 2^{omega} (here omega is
						    considered positive)
    Output: Ilabel  --> 2D image of labels
    '''
    Sref = np.zeros([sizeIm[0],sizeIm[1],numLabels])
    Ilabel = np.zeros([sizeIm[0],sizeIm[1]])
    centroid = (int(sizeIm[0]/2),int(sizeIm[1]/2))
    rWm,rCortex,rCsf = int(np.min(centroid)*0.7),int(np.min(centroid)*0.8),int(np.min(centroid)*0.9)
    x,y = np.ogrid[-centroid[0]:sizeIm[0]-centroid[0], -centroid[1]:sizeIm[1]-centroid[1]]
    wm = x**2 + y**2 <= rWm**2
    cortex = x**2 + y**2 <= rCortex**2
    Ilabel[cortex] = 2
    Ilabel[wm] = 1
    for i in xrange(numLabels):
        Sref[:,:,i] = 1*Ilabel==i
    nibabel.save(nibabel.Nifti1Image(Sref,header),opath+'/testFarSpheres2D_Sref.nii.gz')

    Ilabel = np.zeros([sizeIm[0],sizeIm[1]])
    rWm,rCortex,rCsf = int(np.min(centroid)*0.3),int(np.min(centroid)*0.4),int(np.min(centroid)*0.5)
    centroid = (int(sizeIm[0]/2),int(sizeIm[1]/2))
    x,y = np.ogrid[-centroid[0]:sizeIm[0]-centroid[0], -centroid[1]:sizeIm[1]-centroid[1]]
    wm = x**2 + y**2 <= rWm**2
    cortex = x**2 + y**2 <= rCortex**2
    Ilabel[cortex] = 2
    Ilabel[wm] = 1

    return Ilabel


def buildTestFarSpheres(sizeIm,numLabels,header,opath,omega):
    '''
    Build a test model with 3D concentrical spheres.
	-------------------------------
    Input:  sizeIm  --> Image size
            numLabels   --> Number of labels
            header      --> Affine matrix for saving results as a Nifti image
            opath       --> Path where outputs will be saved
            omega       --> Factor of downsampling, 2^{omega} (here omega is
						    considered positive)
    Output: Ilabel  --> Image of labels
    '''
    Sref=np.zeros([sizeIm[0],sizeIm[1],sizeIm[2],numLabels])
    Ilabel=np.zeros([sizeIm[0],sizeIm[1],sizeIm[2]])
    centroid=(int(sizeIm[0]/2),int(sizeIm[1]/2),int(sizeIm[2]/2))
    rWm,rCortex,rCsf=int(np.min(centroid)*0.7),int(np.min(centroid)*0.8),int(np.min(centroid)*0.9)
    x,y,z = np.ogrid[-centroid[0]:sizeIm[0]-centroid[0], -centroid[1]:sizeIm[1]-centroid[1], -centroid[2]:sizeIm[2]-centroid[2]]
    wm = x**2 + y**2 + z**2 <= rWm**2
    cortex = x**2 + y**2 + z**2 <= rCortex**2
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
    Ilabel[cortex]=2
    Ilabel[wm]=1

    return Ilabel


def buildTestCrossEllipsoids(sizeIm,numLabels,header,opath,omega):
    '''
    Build a test model with concentrical ellipsoids.
	-------------------------------
    Input:  sizeIm  --> Image size
            numLabels   --> Number of labels
            header      --> Affine matrix for saving results as a Nifti image
            opath       --> Path where outputs will be saved
            omega       --> Factor of downsampling, 2^{omega} (here omega is
						    considered positive)
    Output: Ilabel  --> Image of labels
    '''
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

    return Ilabel
