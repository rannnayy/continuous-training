#!/usr/bin/env python3

import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import os
import numpy as np
import time
import tensorflow as tf
# import sklearnex
# patch_sklearn()
import resource
import psutil
from timeit import default_timer

tf.config.set_visible_devices([], 'GPU')

from model.classification.nn import NN as NeuralNetworkClf
from model.classification.randforest import RandomForest as RandomForestClf
from model.regression.nn import NN as NeuralNetworkReg
from model.regression.nurd import Reweight
# from model.regression.randforest import RandomForest as RandomForestReg

DATA_TRAIN_DURATION_MIN = 5
DATA_RETRAIN_DURATION_MIN = 5
DATA_EVAL_DURATION_MIN = 1
DATA_MONITOR_PERIOD_MIN = 1
DATA_MONITOR_DURATION_MIN = 1
BATCH_SIZE = 256

# For Overhead Evaluation
EVAL_TIME = 0
EVAL_MEMORY_USAGE = 0
EVAL_CPU_TIME = 0
EVAL_COUNTER = 0
TRAIN_TIME = 0
TRAIN_MEMORY_USAGE = 0
TRAIN_CPU_TIME = 0
TRAIN_COUNTER = 0

def create_output_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path

def write_stats(filePath, statistics):
    with open(filePath, "w") as text_file:
        text_file.write(statistics)
    print("===== output file : " + filePath)

def eval(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_values = [0 for i in range(4)]
    i = 0
    for row in cm:
        for val in row:
            cm_values[i] = val
            i += 1
    TN, FP, FN, TP = cm_values[0], cm_values[1], cm_values[2], cm_values[3]
    return roc_auc_score(y_test, y_pred), average_precision_score(y_test, y_pred), f1_score(y_test, y_pred), accuracy_score(y_test, y_pred), FP/(FP+TN+0.1), FN/(TP+FN+0.1)

models = {
    'nn_reg': NeuralNetworkReg,
    'nn_clf': NeuralNetworkClf,
    # 'rf_reg': RandomForestReg,
    'rf_clf': RandomForestClf
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", help="Path to the dataset folder", type=str, required=True)
    parser.add_argument("-dataset_name", help="Name of dataset file", type=str, required=True)
    parser.add_argument("-data_train_duration_min", help="Data duration used for training", type=int, default=DATA_TRAIN_DURATION_MIN)
    parser.add_argument("-data_retrain_duration_min", help="Data duration used for retraining (If not set, the window will be set equal to data_train_duration_min)", type=int, default=DATA_RETRAIN_DURATION_MIN)
    parser.add_argument("-data_eval_duration_min", help="Data duration used for each evaluation", type=int, default=DATA_EVAL_DURATION_MIN)
    parser.add_argument("-eval_period", help="Period of data for evaluation (hour)", type=float, default=1)
    parser.add_argument("-model_name", help="Model's name upon saving", type=str, required=True)
    parser.add_argument("-output", help="Output CSV file name upon saving", type=str, required=True)
    args = parser.parse_args()

    path = args.path
    dataset_name = args.dataset_name
    data_train_duration_min = args.data_train_duration_min
    data_retrain_duration_min = args.data_retrain_duration_min
    data_eval_duration_min = args.data_eval_duration_min

    # Prepare Output Directory
    timestamp = int(time.time_ns())
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), 'results', str(timestamp))
    output_stats = os.path.join(output_dir, args.output)
    output_vars = os.path.join(output_dir, 'parameters.txt')
    output_drift_data = os.path.join(output_dir, 'drift_data.csv')
    create_output_dir(output_dir)

    # Regarding file
    files = [folder for folder in os.listdir(path) if ('chunk' in folder and os.path.isdir(os.path.join(path, folder)))]
    num_files = len(files)
    prefix = files[0].split('_')[0]

    print("The prefix of all folders are", prefix)
    print("There are", num_files, "files")

    # Regarding duration
    data_train_duration_ms = data_train_duration_min * 60 * 1000
    print("Taking", data_train_duration_ms, "ms data to train")

    data_eval_duration_ms = data_eval_duration_min * 60 * 1000
    print("Taking", data_eval_duration_ms, "ms data for every evaluation")

    pending_training_data = None
    num_rows = []
    stats_df = pd.DataFrame(columns=['mode', 'minute', 'us', 'roc_auc', 'pr_auc', 'f1_score', 'accuracy', 'fnr', 'fpr', 'retrain'])
    drift_data = pd.DataFrame(columns=['p0', 'p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90', 'p100', 'drift', 'f1'])

    output_cycle = os.path.join(output_dir, 'train')
    create_output_dir(output_cycle)

    curr_ts = data_train_duration_ms
    eval_mode = False
    index_data_train = 0
    index_data_1min = 0
    print(1, int(args.eval_period*12)+1)
    for i in range(1, int(args.eval_period*12)+1):
        print("="*20, i, "="*20)
        dataset_path = os.path.join(path, prefix + "_" + str(i), dataset_name)
        print("Dataset Path =>", dataset_path)

        dataset_new = pd.read_csv(dataset_path)
        if i > 1:
            dataset = pd.concat([dataset, dataset_new], ignore_index=True)
        else:
            dataset = dataset_new.copy(deep=True)
        dataset_new = None
        dataset.reset_index(inplace=True, drop=True)

        if not eval_mode:
            # Train Mode
            if (dataset['ts_record'] >= (data_train_duration_ms)).idxmax() > 0:
                index_data_train = (dataset['ts_record'] >= (data_train_duration_ms)).idxmax()
                print("Taking", i * data_train_duration_ms)
                print("Number of data for train", index_data_train)
                dataset_train = dataset[:index_data_train]
                dataset = dataset[index_data_train:]
                x = dataset_train.copy(deep=True).drop(columns=["ts_record", "reject"], axis=1)
                y = dataset_train["reject"].copy(deep=True)

                # Get training throughput data for DD model dataset
                # initial_thpt = dataset_train['size']/dataset_train['latency']
                # summary_initial_thpt = np.array([int(np.percentile(initial_thpt, x)) for x in range(0, 101, 10)])
                
                # Train if model doesn't exist
                # if not (os.path.isfile(os.path.join(os.path.dirname(output_dir), args.model_name + '_norm.joblib')) and os.path.isfile(os.path.join(os.path.dirname(output_dir), args.model_name + ('.keras' if 'nn_' in model_algo else '.joblib')))):
                # Specific output directory
                start_time = default_timer()
                train_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                cpu_times = psutil.cpu_times()
                model_instance = Reweight()
                model_instance.train(x, y, True, os.path.join(output_cycle, args.model_name + '_logreg.joblib'), os.path.join(output_cycle, args.model_name + '_tree.joblib'), False)
                TRAIN_TIME += default_timer()-start_time
                TRAIN_MEMORY_USAGE += (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0 - train_mem)
                TRAIN_CPU_TIME += psutil.cpu_times().user-cpu_times.user
                TRAIN_COUNTER += 1
                print("train", len(y))
                # else:
                #     print("Just copy...")
                #     shutil.copy(os.path.join(os.path.dirname(output_dir), args.model_name + '_norm.joblib'), os.path.join(output_cycle, args.model_name + '_norm.joblib'))
                #     shutil.copy(os.path.join(os.path.dirname(output_dir), args.model_name + ('.keras' if 'nn_' in model_algo else '.joblib')), os.path.join(output_cycle, args.model_name + ('.keras' if 'nn_' in model_algo else '.joblib')))
                #     model_instance = models[model_algo](batch_size, x, y, 'BatchNorm')
                #     model_instance.dnn_model = tf.keras.models.load_model(os.path.join(output_cycle, args.model_name + ('.keras' if 'nn_' in model_algo else '.joblib')))
                #     model_instance.norm = joblib.load(os.path.join(output_cycle, args.model_name + '_norm.joblib'))
                
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
                y_pred = model_instance.pred(x_test)
                roc_auc, pr_auc, f1, acc, fnr, fpr = eval(y_test.values, y_pred)
                stats_df.loc[len(stats_df)] = ['train', -1, 0, roc_auc, pr_auc, f1, acc, fnr, fpr, False]

                eval_mode = True

        if eval_mode:
            # Eval Mode
            # One chunk has 5 mins of data
            for j in range(1, 6):
                print("*"*20)
                curr_ts += data_eval_duration_ms
                index_data_1min = (dataset['ts_record'] >= ((i-1)*5 + j) * data_eval_duration_ms).idxmax()

                print("Taking", ((i-1)*5 + j) * data_eval_duration_ms)
                print("Number of data for #", ((i-1)*5) + j, "eval", index_data_1min)
                dataset_1min = dataset[:index_data_1min]
                dataset = dataset[index_data_1min:]
                dataset.reset_index(drop=True, inplace=True)
                # print(dataset_1min['ts_record'].head().tolist())
                # print(dataset_1min['ts_record'].tail().tolist())
                # print(dataset_1min.shape)

                # Prepare evaluation data
                x = dataset_1min.copy(deep=True).drop(columns=["ts_record", "reject"], axis=1)
                y = dataset_1min["reject"].copy(deep=True)
                print("eval", i, j, len(y))

                # Evaluation
                if len(x) > 0:
                    y_pred = model_instance.pred(x)
                    roc_auc, pr_auc, f1, acc, fnr, fpr = eval(y.values, y_pred)
                    stats_df.loc[len(stats_df)] = ['test', ((i-1)*5) + j, ((i-1)*5 + j) * data_eval_duration_ms, roc_auc, pr_auc, f1, acc, fnr, fpr, False]
                
    stats_df.to_csv(output_stats)
    print("Output file =>", output_stats)

    drift_data.to_csv(output_drift_data)
    print("Output drift data =>", output_drift_data)

    params = []
    params.append("-path = "+str(path))
    params.append("-dataset_name = "+str(dataset_name))
    params.append("-data_train_duration_min = "+str(data_train_duration_min))
    params.append("-data_retrain_duration_min = "+str(data_retrain_duration_min))
    params.append("-data_eval_duration_min = "+str(data_eval_duration_min))
    params.append("-output = "+str(args.output))
    params.append("-eval_period = "+str(args.eval_period))
    params.append("-model_name = "+str(args.model_name))
    params.append("-output = "+str(args.output))
    params.append(" =====> Output = "+str(output_dir))
    write_stats(output_vars, '\n'.join(params))