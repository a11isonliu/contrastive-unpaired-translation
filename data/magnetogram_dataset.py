"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset
import pandas as pd
import os
import random
import numpy as np
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as transforms
import torch

# from PIL import Image
# from data.image_folder import make_dataset



class MagnetogramDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--file_savepath', type=str, default='placeholder', help='path to directory where files are saved')
        if is_train:
            parser.set_defaults(file_savepath='/media/faraday/magnetograms_fd', dataroot='/media/faraday/alli7928/mdi2hmi', load_size = 2048, crop_size = 360, batch_size = 8, preprocess = 'resize_and_crop', model = 'cut')  # specify dataset-specific default values
        else:
            parser.set_defaults(file_savepath='/media/faraday/magnetograms_fd', dataroot='/media/faraday/alli7928/mdi2hmi', preprocess = 'resize', load_size = 2048, batch_size = 4, model = 'cut')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        mdi_df = pd.read_csv(os.path.join(opt.dataroot, opt.phase + 'A.csv'), names=['filename'])
        hmi_df = pd.read_csv(os.path.join(opt.dataroot, opt.phase + 'B.csv'), names=['filename'])
        mdi_files = [os.path.join(opt.file_savepath, 'mdi_fd', file) for file in mdi_df['filename']]
        hmi_files = [os.path.join(opt.file_savepath, 'hmi_fd', file) for file in hmi_df['filename']]
        self.A_paths = sorted([file for file in mdi_files])
        self.B_paths = sorted([file for file in hmi_files])
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within the range
        if self.opt.serial_batches:   # make sure index is within the range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        #load npy array
        A_arr = np.load(A_path)
        B_arr = np.load(B_path)
        
        # remove nans
        # A_img = Image.fromarray(A_arr)
        # B_img = Image.fromarray(B_arr)
        
        # Normalize images to between 0-1 using standard min-max normalization

        A_img = np.nan_to_num(A_arr).astype('float32')
        A_img[np.where(A_img > 5000)] = 5000
        A_img[np.where(A_img < -5000)] = -5000
        A_img = (A_img + 5000) / 10000
        A_img = torch.from_numpy(A_img).unsqueeze(0) # convert from np.array to tensor
 
        B_img = np.nan_to_num(B_arr).astype('float32')
        B_img[np.where(B_img > 6000)] = 6000
        B_img[np.where(B_img < -6000)] = -6000
        B_img = (B_img + 6000) / 12000
        B_img = torch.from_numpy(B_img).unsqueeze(0) # convert from np.array to tensor
       

        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        transform = get_magnetogram_transform(self.opt, convert=False)
        A = transform(A_img)    # A_img must be a tensor or PIL Image
        B = transform(B_img)    # B_img must be a tensor or PIL Image
        
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images."""
        return max(self.A_size, self.B_size)

def remove_nans(arr, replace_val=0):
    no_nans = np.nan_to_num(arr, copy = False, nan = replace_val)
    return no_nans

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_magnetogram_transform(opt, params=None, method=InterpolationMode.BICUBIC, convert=False):
    transform_list = []
    if 'fixsize' in opt.preprocess:
        transform_list.append(transforms.Resize(params["size"], method))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, interpolation=method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))
    if 'crop' in opt.preprocess:
        if params is None or 'crop_pos' not in params:
            transform_list.append(transforms.RandomCrop(opt.crop_size))

    # if opt.preprocess == 'none':
    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None or 'flip' not in params:
            transform_list.append(transforms.RandomHorizontalFlip())

    if convert:
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5,), (0.5,))]

    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=InterpolationMode.BICUBIC):
    x, ow, oh = img.size()
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    return img.resize((1, w, h), interpolation=method)


def __scale_width(img, target_width, crop_width, method=InterpolationMode.BICUBIC):
    x, ow, oh = img.size()
    if ow == target_width and oh >= crop_width:
        return img
    w = target_width
    h = int(max(target_width * oh / ow, crop_width))
    return img.resize((1, w, h), interpolation=method)


def __crop(img, pos, size):
    x, ow, oh = img.size()
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
