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
from data.base_dataset import BaseDataset, get_transform
import pandas as pd
import os
import random
import numpy as np
from PIL import Image
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
            parser.set_defaults(file_savepath='/media/faraday/magnetograms_fd', dataroot='/media/faraday/alli7928/mdi2hmi', load_size = 4096, crop_size = 360, batch_size = 8, preprocess = 'resize_and_crop')  # specify dataset-specific default values
        else:
            parser.set_defaults(file_savepath='/media/faraday/magnetograms_fd', dataroot='/media/faraday/alli7928/mdi2hmi', preprocess = 'none')
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
        
        A_img = Image.fromarray(A_arr)
        B_img = Image.fromarray(B_arr)
        
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        transform = get_transform(self.opt, grayscale=True, convert=True, normalize=False)
        
        A = transform(A_img)    # A_img must be a tensor or PIL Image
        B = transform(B_img)    # B_img must be a tensor or PIL Image
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images."""
        return max(self.A_size, self.B_size)
