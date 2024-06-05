#!/usr/bin/env python3

import argparse
import subprocess
from subprocess import call
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", help="Path to the dataset", type=str)
    parser.add_argument("-datasets", help="Path of the datasets", nargs='+', type=str)
    parser.add_argument("-model", help="The filename of the model (.py)",type=str)
    args = parser.parse_args()
    if (not args.dataset and not args.datasets and not args.model and not args.train_eval_split):
        print("    ERROR: You must provide these arguments: -dataset <the labeled trace> -model <the model name> -train_eval_split <the split ratio> ")
        exit(-1)

    arr_dataset = []
    if args.datasets:
        arr_dataset += args.datasets
    elif args.dataset:
        arr_dataset.append(args.dataset)
    print("trace_profiles = " + str(arr_dataset))
    
    for dataset_path in arr_dataset:
        print("\nTraining on " + str(dataset_path))
        command = "python ./" + args.model + ".py -dataset " + dataset_path
        subprocess.call(command, shell=True)

# Example how to run:
# python train.py -dataset /mnt/extra/flashnet/model_collection/3_continual_learning/dd_model_experiments/dataset/nvme1n1/alibaba.per_10k/profile_v1.feat_v6.readonly.dataset/classification_nn/algo_7_kl_thpt/100_dataset/drift_data.csv -model flashnet_dd_randforest