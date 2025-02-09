#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import os

from eval import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-datasets", help="Arr of file path of the model datasets", nargs='+', type=str)
    parser.add_argument("-oversample", help="Whether to do oversampling or not", default=False, action='store_true')
    parser.add_argument("-undersample", help="Whether to do oversampling or not", default=False, action='store_true')
    parser.add_argument("-output", help="Path to save the dataset (folder)", type=str, required=True)
    args = parser.parse_args()
    
    base_path = os.path.abspath(args.output)
    create_output_dir(base_path)
    
    dataset_paths = []
    if args.datasets:
        dataset_paths += args.datasets
    print("Concatenate,", "undersample" if args.undersample else "not undersample", "oversample" if args.oversample else "not oversample", "these datasets\n", dataset_paths)

    if not args.datasets or len(dataset_paths) <= 1:
        print("ERROR: Invalid argument. \n(sample: python concatenate_model_dataset.py -datasets ./dataset/nvme1n1/*.per_10k/profile_v1.feat_v6.readonly.dataset/classification_nn/algo_4_ks_test_thpt/100_dataset/normal/drift_data_v0.csv ./dataset/nvme1n1/*.per_10k/profile_v1.feat_v6.readonly.dataset/classification_nn/algo_8_js_thpt/100_dataset/normal/drift_data_v0.csv'")
        exit()
    
    concatenated_df = pd.DataFrame()
    final_dataset = []
    
    for data_path in dataset_paths:
        temp_df = pd.read_csv(data_path, dtype={
            'p0': np.float64,
            'p10': np.float64,
            'p20': np.float64,
            'p30': np.float64,
            'p40': np.float64,
            'p50': np.float64,
            'p60': np.float64,
            'p70': np.float64,
            'p80': np.float64,
            'p90': np.float64,
            'p100': np.float64,
            'drift': np.int32,
            'roc_auc': np.float64
        }, index_col=0)
        
        if temp_df.shape[0] > 1:
            final_dataset.append(data_path)
            if not args.undersample and not args.oversample:
                concatenated_df = pd.concat([concatenated_df, temp_df], ignore_index=True)
                concatenated_df.reset_index(inplace=True, drop=True)
                
            else:
                temp_x = temp_df.copy(deep=True).drop(columns=['drift'])
                temp_y = temp_df.copy(deep=True).drop(columns=[col for col in temp_df.columns if col != 'drift'])
                
                if args.undersample:
                    undersampler = RandomUnderSampler(random_state=42, sampling_strategy='majority')
                    x_samp, y_samp = undersampler.fit_resample(temp_x, temp_y)
                
                if args.oversample:
                    oversampler = SMOTE(random_state=42, sampling_strategy='minority')
                    
                    if args.undersample:
                        x_samp, y_samp = oversampler.fit_resample(x_samp, y_samp)
                    else:
                        x_samp, y_samp = oversampler.fit_resample(temp_x, temp_y)
                
                samp_df = pd.DataFrame()
                for col in temp_df.columns:
                    if col == 'drift':
                        samp_df[col] = y_samp
                    else:
                        samp_df[col] = x_samp[col]
                concatenated_df = pd.concat([concatenated_df, samp_df], ignore_index=True)
                concatenated_df.reset_index(inplace=True, drop=True)
    
    file_path = os.path.join(base_path, 'drift_data.csv')
    concatenated_df.to_csv(file_path, index=False)
    print("=====> Output file:", file_path)
    
    final_dataset.append("undersample" if args.undersample else "not undersample")
    final_dataset.append("oversample" if args.oversample else "not oversample")
    stats_path = os.path.join(base_path, 'drift_data.stats')
    write_stats(stats_path, '\n'.join(final_dataset))