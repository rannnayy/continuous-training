#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import argparse

from eval import *

def train_model(dataset_path):
    model_name = "nn"
    print("Start training the "+ model_name +" model...")
    
    dataset = pd.read_csv(dataset_path)
    if dataset.shape[1] == 13:
        dataset_ver = 'v0'
    else:
        dataset_ver = 'v1'
    
    x_train = dataset.copy(deep=True).drop(columns=["drift", "f1"], axis=1)
    y_train = dataset['drift'].copy(deep=True)

    # Model Directory
    base_dataset_dir = os.path.dirname(dataset_path)
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stored_model', model_name, dataset_ver)
    create_output_dir(base_dir)
    model_dir = os.path.join(base_dir, model_name+'.tf')

    # Data normalization
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(x_train.values))
    
    # Train model
    clf = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(512, activation='relu', input_dim=x_train.values.shape[1]),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    clf.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
    clf.fit(
        x_train.values,
        y_train.values,
        validation_split=0.2,
        verbose=1, epochs=20
    )
    
    # Save the model
    clf.save(model_dir)
    print(" =====> Saved in", model_dir)

    # Evaluation
    y_pred = (clf.predict(x_train.values) > 0.5).flatten()
    dataset['pred_drift'] = y_pred

    # Print confusion matrix and stats
    evaluate(y_train, y_pred, base_dir, model_name)
    evaluate(y_train, y_pred, base_dataset_dir, model_name)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", help="Path to the dataset", type=str)
    args = parser.parse_args()
    if (not args.dataset):
        print("    ERROR: You must provide these arguments: -dataset <the labeled trace>")
        exit(-1)

    train_model(args.dataset)
    
# How to run:
# python flashnet_dd_randforest.py -dataset /mnt/extra/flashnet/model_collection/3_continual_learning/dd_model_experiments/dataset/nvme1n1/alibaba.per_10k/profile_v1.feat_v6.readonly.dataset/classification_nn/algo_4_ks_test_thpt/100_dataset/drift_data.csv