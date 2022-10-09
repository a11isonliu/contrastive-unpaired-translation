import pandas as pd
import os

def load_csv(csv_file):
    filelist = pd.read_csv(csv_file)
    return filelist

def rand_select(df_train, num_per_day = 1):
    '''Randomly select n samples (without replacement) per day from each dataset to train on.'''
    groupby_date_df = df_train.groupby(df_train['APPROX_DATETIME'].dt.date)
    df_subsampled = groupby_date_df.apply(lambda x: x.sample(n=num_per_day) if x.shape[0] >= num_per_day else x).reset_index(drop=True)
    return df_subsampled

def pair_mdi_hmi(phase, df_test_mdi, df_test_hmi):
    """
    phase = {'train', 'test'}
    """
    mdi_df = pd.DataFrame(df_test_mdi['npy_filename'], columns=['npy_filename'])
    hmi_df = pd.DataFrame(df_test_hmi['npy_filename'], columns=['npy_filename'])
    
    mdi_df ['APPROX_DT'] = mdi_df['npy_filename'].str.split('.').str[2].str.strip('_TAI')
    hmi_df ['APPROX_DT'] = hmi_df['npy_filename'].str.split('.').str[2].str.strip('_TAI')
    
    hmi_df['MATCH'] = hmi_df['APPROX_DT'].isin(mdi_df['APPROX_DT'])
    mdi_df['MATCH'] = mdi_df['APPROX_DT'].isin(hmi_df['APPROX_DT'])
    
    hmi_df.drop(hmi_df[hmi_df.MATCH == False].index, inplace=True)
    mdi_df.drop(mdi_df[mdi_df.MATCH == False].index, inplace=True)

    return pd.DataFrame(mdi_df['npy_filename'], columns=['npy_filename']), pd.DataFrame(hmi_df['npy_filename'], columns=['npy_filename'])

def filter_hmi_mdi_test(n, df_testA, df_testB):
    df_testA.reset_index(inplace=True, drop=True)
    df_testB.reset_index(inplace=True, drop=True)
    df_testA.drop(df_testA[df_testA.index % n != 0].index, inplace=True)
    df_testB.drop(df_testB[df_testB.index % n != 0].index, inplace=True)
    return pd.DataFrame(df_testA, columns=['npy_filename']), pd.DataFrame(df_testB, columns=['npy_filename'])


def main():
    # INPUT PARAMETERS
    save_path = './mdi2hmi_small'
    pair_test_set = True
    filter_test_set = 10 # keep 1/filter_test_set files in test set
    
    # SOURCE DATA PATHS
    hmi_csv_path = '/media/faraday/magnetograms_fd/hmi_fd/header_info/hmi_header_info.csv' # HMI
    mdi_csv_path = '/media/faraday/magnetograms_fd/mdi_fd/header_info/mdi_header_info.csv' # MDI

    csv_path_list = [mdi_csv_path, hmi_csv_path]
    for csv_path in csv_path_list:
        filelist = load_csv(csv_path)
        ds = filelist.fits_filename.str[0:3][0]
        assert ds == 'mdi' or ds == 'hmi', 'dataset not valid'
        filelist['APPROX_DATETIME'] = pd.to_datetime(filelist['approx_datetime'], format = "%Y%m%d_%H%M%S")
        
        # Drop rows with NaNs, drop non-datetime formatted column, and set index to datetime formatted approximate date
        filelist = filelist.dropna(axis='rows', subset=['fits_filename'])
        filelist = filelist.drop(['approx_datetime'], errors='ignore')
        print(filelist.__len__(), ds.upper(), "files")

        if ds == 'mdi':
            split_date = '2010-04-11 00:00:00'
            df_train_all = filelist.loc[filelist['APPROX_DATETIME'] < split_date]
            df_testA = filelist.loc[filelist['APPROX_DATETIME'] >= split_date]
            df_trainA = rand_select(df_train_all, num_per_day = 2)
            print("MDI train: ", df_trainA.__len__())
            print("MDI test: ", df_testA.__len__())
        elif ds == 'hmi':
            split_date = '2011-04-11 23:59:59'
            df_train_all = filelist.loc[filelist['APPROX_DATETIME'] > split_date]
            df_testB = filelist.loc[filelist['APPROX_DATETIME'] <= split_date]
            df_trainB = rand_select(df_train_all, num_per_day = 4)

            print("HMI train: ", df_trainB.__len__())
            print("HMI test: ", df_testB.__len__())

    if pair_test_set == True:
        print("Pairing MDI and HMI test sets.")
        df_testA, df_testB = pair_mdi_hmi('test', df_testA, df_testB)
        print("MDI test: ", df_testA.__len__())
        print("HMI test: ", df_testB.__len__())
        if filter_test_set != 0:
            print(f"Selecting 1/{filter_test_set} images from the test set.")
            df_testA, df_testB = filter_hmi_mdi_test(filter_test_set, df_testA, df_testB)
    
    df_trainA['npy_filename'].to_csv(os.path.join(save_path, 'trainA.csv'), index=False, header=False)
    df_trainB['npy_filename'].to_csv(os.path.join(save_path, 'trainB.csv'), index=False, header=False)
    df_testA.to_csv(os.path.join(save_path, 'testA.csv'), index=False, header=False)
    df_testB.to_csv(os.path.join(save_path, 'testB.csv'), index=False, header=False)
    
    print('#' * 30)
    print("Split.")
    print(df_trainA.__len__(), "MDI train files")
    print(df_testA.__len__(), "MDI test files")
    print(df_trainB.__len__(), "HMI train files")
    print(df_testB.__len__(), "HMI test files")
    return
    
if __name__ == '__main__':
    main()