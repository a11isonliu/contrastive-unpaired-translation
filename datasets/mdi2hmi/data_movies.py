#!/usr/bin/env python
# coding: utf-8-sig

import sys
import os
from pathlib import Path
import sunpy 
import ffmpeg
import matplotlib.pyplot as plt
import sunpy.map
import sunpy.io
from sunpy.visualization.colormaps import color_tables as ct
import argparse
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import matplotlib
#Makes MPEG movies of AIA hdf5 files in a specified directory. 

dataset = 'trainA'
DATADIR = '/media/faraday/magnetograms_fd/mdi_fd'
FILELIST = './mdi2hmi_small/'+dataset+'.csv'

# Keyword arguments:
#   filerange  = list:, first, last, and delta filenumbers to use in compiling movie. Eg. RANGE=[1,1002,2] uses every other
#       file in the first 1000 files in the directory in the movie. [-1,1001,1] uses the last 1000 files in the directory.
#       Default = all files in directory in ascending sorted order.

def main():
    remove_png = True
    file_names = pd.read_csv(FILELIST, header=None)
    files = (DATADIR + os.sep + file_names[0].astype(str)).tolist()
    files.sort()
    nfiles = files.__len__()
    print('{} files in directory {}'.format(nfiles, DATADIR))
    
    colormap = plt.get_cmap('hmimag')
    
    #Loop over the range and use plt.save to make .png files in current directory:
    r0 = 0
    r1 = 9664
    delta = 1
    resolution = 300
    
    matplotlib.rcParams.update({'font.size': 6})

    fileroot = f'./mdi2hmi_small/movies/{dataset}' # SAVE PATH
    for i in range(r0,r1,delta):
        mag_data = np.load(files[i])
        file_time = datetime.strptime(str(files[i]).split('/')[-1].split('.')[2].strip('_TAI'),'%Y%m%d_%H%M%S')
        try:
            plt.subplot(1, 2, 1)
            # CHANGE PLOT TITLE
            plt.title(f'{dataset} - '+file_time.strftime('%Y/%m/%d %H:%M:%S'), fontsize = 9)
            plt.imshow(mag_data, cmap = colormap, vmin = -5000, vmax = 5000)
            plt.show()
            print('saving files in', fileroot+'.{0:05d}.png'.format(i))
            plt.savefig(fileroot + '.{0:05d}.png'.format(i), dpi=resolution)
            plt.close()
        except ValueError:
            continue

    #create MPEG movie from .png files using ffmpeg. Note command line for this: ffmpeg -f image2 -i sharp_7222.%06d.png movie.mpg
    (
        ffmpeg
        .input(fileroot+'.*.png', pattern_type='glob', framerate=20) 
        .output(fileroot + 'mdi2hmi_small_' + dataset + '.mov')
        .run()
    )

    #remove the .png files if removepng=True:
    if remove_png:
        for f in Path(fileroot).glob('*.png'):
            os.remove(f)

if __name__ == "__main__":
    
        main()

