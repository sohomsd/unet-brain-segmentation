import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

# Function to load in .nii image data as a torch tensor
def load_BraTS_image(data_path, slice=None, crop_indices=(37,197,28,220)):
    img = nib.load(data_path).get_fdata()
    if slice != None:
        img = img[:,:,slice]
    if crop_indices != None:
        u, d, l, r = crop_indices
        img = img[u:d, l:r]
    return torch.tensor(img, dtype=torch.float32)

# Function to change expand segmentation data to 3 channels
# Note that class label 4 corresponds to the 3rd channel
def expand_BraTS_segmentation(data_path, slice=None, labels=[0, 1, 2, 4], crop_indices=(37,197,28,220)):
    seg_img = load_BraTS_image(data_path, slice, crop_indices)
    
    label_channels = []
    for label in labels:
        curr_ch = torch.zeros(seg_img.shape, dtype=torch.float32)
        curr_ch[seg_img == label] = 1
        label_channels.append(curr_ch)

    return torch.stack(label_channels)

# BraTS Dataset class
class BraTSDataset(Dataset):
    def __init__(self, root_dir, slice=None, img_dirs=None, crop_indices=(37,197,28,220), img_transform=None):
        self.slice = slice
        self.crop_indices = crop_indices
        self.img_transform = img_transform
        self.image_path_prefixes = []

        if img_dirs == None:
            img_dirs = os.listdir(root_dir)
        
        for dir in img_dirs:
            self.image_path_prefixes.append(root_dir + "/" + dir + f"/{dir}")


    def __len__(self):
        return len(self.image_path_prefixes)


    def __getitem__(self, idx):
        prefix = self.image_path_prefixes[idx]
        t1 = load_BraTS_image(prefix + "_t1.nii", self.slice, self.crop_indices)
        t1ce = load_BraTS_image(prefix + "_t1ce.nii", self.slice, self.crop_indices)
        t2 = load_BraTS_image(prefix + "_t2.nii", self.slice, self.crop_indices)
        flair = load_BraTS_image(prefix + "_flair.nii", self.slice, self.crop_indices)

        contrast_img = torch.stack([t1, t1ce, t2, flair])
        seg_img = load_BraTS_image(prefix + "_seg.nii", self.slice, self.crop_indices)

        label_channels = []
        for label in [0, 1, 2, 4]:
            curr_ch = torch.zeros(seg_img.shape, dtype=torch.float32)
            curr_ch[seg_img == label] = 1
            label_channels.append(curr_ch)

        seg_img = torch.stack(label_channels)

        if self.img_transform:
            contrast_img = self.img_transform(contrast_img)

        return contrast_img, seg_img


# Function to get the mean and standard deviation per channel of a BraTS dataset
def get_distribution_stats(dataset: BraTSDataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=32)
    num_imgs = 0
    mean = 0
    var = 0
    for imgs, _ in loader:
        num_imgs += imgs.size(0)
        mean += torch.mean(imgs, dim=(2,3)).sum(0)
        var += torch.var(imgs, dim=(2,3)).sum(0)

    mean /= num_imgs
    std = torch.sqrt(var / num_imgs)

    return mean, std