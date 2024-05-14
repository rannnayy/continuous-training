#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_predict
from sklearn.cluster import KMeans
import joblib
import argparse

from eval import *

def train_model(dataset_path):
    model_name = "kmeans"
    print("Start training the "+ model_name +" model...")
    
    dataset = pd.read_csv(dataset_path)
    
    x_train = dataset.copy(deep=True).drop(columns=["drift", "roc_auc"], axis=1)
    y_train = dataset['drift'].copy(deep=True)

    # Data normalization with sklearn
        # fit scaler on training data
    # Model Directory
    base_dataset_dir = os.path.dirname(dataset_path)
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stored_model', model_name)
    create_output_dir(base_dir)
    model_dir = os.path.join(base_dir, model_name+'.joblib')
    norm_dir = os.path.join(base_dir, model_name+'_'+'norm.joblib')

    # Train the model
    norm = RobustScaler().fit(x_train.values)
    
    # transform training data
    x_train_norm = norm.transform(x_train.values)
    joblib.dump(norm, norm_dir)
    
    clt = KMeans(n_clusters=2, random_state=42, n_init="auto")
    clt.fit(x_train_norm)
    
    # Save the model
    joblib.dump(clt, model_dir)
    print(" =====> Saved in", model_dir)
    print(" =====> Saved in", norm_dir)

    # Evaluation
    y_pred = clt.predict(x_train_norm)
    dataset['pred_drift'] = y_pred

    # Print confusion matrix and stats
    evaluate(y_train, y_pred, base_dir, model_name)
    evaluate(y_train, y_pred, base_dataset_dir, model_name)
    
    # # Save cluster centers
    # cluster_centers = clt.cluster_centers_
    # cluster_centers_labels = np.unique(clt.labels_).tolist().sort()
    # cluster_centers.extend(cluster_centers_labels)
    # for i in cluster_centers_labels:
    #     cluster_centers.append(clt.labels_.tolist().count(i))
    # write_stats(os.path.join(base_dataset_dir, 'cluster_centers.txt'), '\n'.join(cluster_centers))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", help="Path to the dataset", type=str)
    args = parser.parse_args()
    if (not args.dataset):
        print("    ERROR: You must provide these arguments: -dataset <the labeled trace>")
        exit(-1)

    train_model(args.dataset)
    
# How to run:
# python flashnet_dd_kmeans.py -dataset /mnt/extra/flashnet/model_collection/3_continual_learning/dd_model_experiments/dataset/nvme1n1/alibaba.per_10k/profile_v1.feat_v6.readonly.dataset/classification_nn/algo_4_ks_test_thpt/100_dataset/drift_data.csv