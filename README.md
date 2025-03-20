# unet-brain-segmentation
## Project Description
This repository contains code that defines model architecture, data processing, training schemes, and evaluation metrics for performing brain tumor segmentation with U-net and Dilated Inception U-net (DIU-net).

`data.py` primarily defines classes and functions for loading data and early pre-processing.

`unet_models/` contains `base.py` and `dilated_inception.py` which correspond to the model class definitions for U-net and DIU-net, respectively.

`metrics.py` contains definitions for the Dice coefficient and Jaccard index. These will be called during training to records the training and validation performance.

`train.py` loads data, initializes and trains the model, and outputs training/validation history incorporating the previously-defined performance metrics.

## Dataset
The models in this project are trained and validated on the BraTS2020 Dataset [4,5,6,7,8], which can be downloaded through the [Training + Validation Kaggle link](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation).

## References
[1] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation.” Available: https://arxiv.org/pdf/1505.04597 

[2] D. E. Cahall, G. Rasool, N. C. Bouaynaya, and H. M. Fathallah-Shaykh, “Dilated Inception U-Net (DIU-Net) for Brain Tumor Segmentation,” arXiv.org, Aug. 15, 2021. https://arxiv.org/abs/2108.06772

[3] Y. Thakre, Y. Gajera, S. Joshi, and J. George, “Dilated Inception U-Net for Nuclei Segmentation in Multi-Organ
Histology Images” irjet.net, Oct., 2022. https://www.irjet.net/archives/V9/i10/IRJET-V9I10135.pdf

[4] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[5] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

[6] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)

[7] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q

[58] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.GJQ7R0EF
