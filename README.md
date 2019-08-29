# neoSeg

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Run](#run)
- [Publications](#publications)

## Introduction

This repository contains the work during my PhD, titled "Automatic segmentation of the cortical surface in neonatal brain MRI". It includes two main contributions:

- **patchBasedSegmentation.py**, where several methods of label fusion (the final step of the multi-atlas segmentation approaches) are performed, including IMAPA [1].
- **topologicalCorrection.py**, where a topological correction for segmentation is presented, which consists in a multi-scale, multi-label homotopic deformation [2].

## Requirements

All scripts were coded in `python 2.7`, but we are working to be compatible to `python 3.7`.

<details>
<summary><b>Python packages</b></summary>

- argparse
- nibabel
- numpy
- scipy
- time
- itertools
- multiprocessing
- numba
- math
- random
- matplotlib
- skimage
- skfmm
</details>

## Run

### patchBasedSegmentation.py

Example of IMAPA application using a atlas set of two pairs of images using two iterations (`alpha` = 0 and 0.25) using 4 threads in parallel:

```
python neoSeg/patchBasedSegmentation.py  -i  brain.nii.gz -a  atlas1_registered_HM.nii.gz  atlas2_registered_HM.nii.gz -l  label1_propagated.nii.gz  label2_propagated.nii.gz  -mask mask.nii.gz  -m IMAPA  -hss 3  -hps 1  -k 15  -alphas 0 0.25  -t 4
```

**i**: input anatomical image
**a**: anatomical atlas images in the input space
**l**: label atlas images in the input space
**mask**: binary image for input
**m**: segmentation method chosen (LP, S_opt, I_opt, IS_opt or IMAPA)
**hss**: half search window size
**hps**: half patch size
**k**: k-Nearest Neighbors (kNN)
**alphas**: alphas parameter for IS_opt and IMAPA methods
**t**: Number of threads (0 for the maximum number of cores available)

> Note: We recommend to previously register the intensity image from the atlas set to the input image, apply a histogram matching algorithm and propagate the transformations to the label maps.

### topologicalCorrection.py

Coming soon...

## Publications

-[1] C. Tor-Díez, N. Passat, I. Bloch, S. Faisan, N. Bednarek and F. Rousseau, “An iterative multi-atlas patch-based approach for cortex segmentation from neonatal MRI,” Computerized Medical Imaging and Graphics, 70:73–82, 2018, hal-01761063.

-[2] C. Tor-Díez, S. Faisan, L. Mazo, N. Bednarek, Hélène Meunier, I. Bloch, N. Passat and F. Rousseau, “Multilabel, multiscale topological transformation for cerebral MRI segmentation post-processing,” In 14th International Symposium on Mathematical Morphology (ISMM 2019), pp. 471–482, 2019, hal-01982972.
