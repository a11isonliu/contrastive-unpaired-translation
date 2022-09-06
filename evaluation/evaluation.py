import os
import numpy as np
import pandas as pd
import random
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio

def abs_error(im1, im2):
    return np.nansum(np.abs(im1-im2))

def eval_summary(results_path, n_images=50, verbose=True, save_csv=True):
    mdi_path = os.path.join(results_path + 'real_A')
    hmi_path = os.path.join(results_path + 'real_B')
    syn_path = os.path.join(results_path + 'fake_B')
    
    mdi_list = [os.path.join(mdi_path, f) for f in os.listdir(mdi_path) if str(f).endswith('.npy')]
    hmi_list = [os.path.join(hmi_path, f) for f in os.listdir(hmi_path) if f.endswith('.npy')]
    syn_list = [os.path.join(syn_path, f) for f in os.listdir(syn_path) if f.endswith('.npy')]

    mdi_list.sort()
    hmi_list.sort()
    syn_list.sort()
    
    abs_err0, abs_err = [], []
    mse0, mse = [], []
    rmse0, rmse = [], []
    psnr0, psnr = [], []
    ssim0, ssim = [], []
    files = []
    im_n = []
    
    n_mdi = len(mdi_list)
    if n_images > n_mdi:
        n_images = n_mdi

    image_index = random.sample(range(0, len(mdi_list)), n_images)
    for n, im in enumerate(mdi_list):
        if n in image_index:
            # LOAD FILES
            mdi = np.load(mdi_list[n])
            hmi = np.load(hmi_list[n])
            syn = np.load(syn_list[n])
            
            im_n.append(n)
            files.append(str(im).split('/')[-1])

            # HMI vs MDI
            abs_err0.append(abs_error(hmi, mdi))
            mse0.append(mean_squared_error(hmi, mdi))
            rmse0.append(np.sqrt(mean_squared_error(hmi, mdi)))
            psnr0.append(peak_signal_noise_ratio(hmi, mdi, data_range=10000))
            ssim0.append(structural_similarity(hmi, mdi))
            
            # HMI vs Synthetic
            abs_err.append(abs_error(hmi, syn))
            mse.append(mean_squared_error(hmi, syn))
            rmse.append(np.sqrt(mean_squared_error(hmi, syn)))
            psnr.append(peak_signal_noise_ratio(hmi, syn, data_range=10000))
            ssim.append(structural_similarity(hmi, syn))

    err_df = pd.DataFrame(list(zip(im_n, files, abs_err, mse, rmse, psnr, ssim)), columns=['Image Number', 'file', 'Abs Error', 'MSE', 'RMSE', 'PSNR', 'SSIM'])
    err0_df = pd.DataFrame(list(zip(im_n, files, abs_err0, mse0, rmse0, psnr0, ssim0)), columns=['Image Number', 'file', 'Abs Error', 'MSE', 'RMSE', 'PSNR', 'SSIM'])
    
    if save_csv == True:
        save_path = '/'.join(str(results_path).split('/')[:-2])
        err_df.to_csv(save_path + os.sep + 'hmi_syn_error.csv', columns = ['Image Number', 'file', 'Abs Error', 'MSE', 'RMSE', 'PSNR', 'SSIM'])
        err0_df.to_csv(save_path + os.sep + 'hmi_mdi_error.csv', columns = ['Image Number', 'file', 'Abs Error', 'MSE', 'RMSE', 'PSNR', 'SSIM'])

    if verbose==True:
        print('##################################')
        print('HMI vs Generated')
        print('##################################')
        print(f'MEAN:\n {err_df.mean()}')
        print('##################################')
        print(f'MEDIAN: \n {err_df.median()}')
        print('##################################')
        print('\n##################################')
        print('HMI vs MDI') 
        print('##################################')
        print(f'MEAN:\n {err0_df.mean()}')
        print('##################################')
        print(f'MEDIAN: \n {err0_df.median()}')
        print('##################################')

    return err_df, err0_df

