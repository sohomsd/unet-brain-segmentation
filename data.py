import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot

# Function to load in .nii image data as a torch tensor
def load_BraTS_image(data_path, slice=None):
    img = nib.load(data_path).get_fdata()
    if slice != None:
        img = img[:,:,slice]
    return torch.tensor(img)

# Function to change expand segmentation data to 3 channels
# Note that class label 4 corresponds to the 3rd channel
def expand_BraTS_segmentation(data_path, slice=None, labels=[0, 1, 2, 4]):
    seg_img = load_BraTS_image(data_path, slice)
    
    label_channels = []
    for label in labels:
        curr_ch = torch.zeros(seg_img.shape)
        curr_ch[seg_img == label] = 1
        label_channels.append(curr_ch)

    return torch.stack(label_channels)

# BraTS Dataset class
class BraTSDataset(Dataset):
    def __init__(self, root_dir, slice=None, contrast_transform=None, seg_transform=None):
        self.slice = slice
        self.contrast_transform = contrast_transform
        self.seg_transform = seg_transform
        self.image_path_prefixes = []

        img_dirs = os.listdir(root_dir)
        for dir in img_dirs:
            self.image_path_prefixes.append(root_dir + "/" + dir + f"/{dir}")


    def __len__(self):
        return len(self.image_path_prefixes)


    def __getitem__(self, idx):
        prefix = self.image_path_prefixes[idx]
        t1 = load_BraTS_image(prefix + "_t1.nii", self.slice)
        t1ce = load_BraTS_image(prefix + "_t1ce.nii", self.slice)
        t2 = load_BraTS_image(prefix + "_t2.nii", self.slice)
        flair = load_BraTS_image(prefix + "_flair.nii", self.slice)

        contrast_img = torch.stack([t1, t1ce, t2, flair])
        seg_img = expand_BraTS_segmentation(prefix + "_seg.nii", self.slice)

        if self.contrast_transform:
            contrast_img = self.contrast_transform(contrast_img)

        if self.seg_transform:
            seg_img = self.seg_transform(seg_img)

        return contrast_img, seg_img