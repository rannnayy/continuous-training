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

def time_dd(time, now_time):
    if now_time % time == 0:
        return True
    return False

def monitor(y_then, y_now, dd_algo, extra_params=[]):
    if 'time' in dd_algo:
        return time_dd(int(dd_algo.split('time_')[1].split('min')[0]), extra_params[0])
    if dd_algo not in ['heuristics-based-labeler', 'model-cluster', 'model-networks']:
        return drift_detectors[dd_algo](y_then, y_now)
    else:
        if dd_algo == 'heuristics-based-labeler':
            return drift_detectors[dd_algo](y_then, y_now, extra_params[0], extra_params[1])
        else:
            return drift_detectors[dd_algo](y_then, y_now, extra_params[0])

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
    parser.add_argument("-roc_auc_threshold", help="ROC-AUC threshold for retraining (if using simple retraining condition, not specifying -model_algo)", type=float)
    parser.add_argument("-batch_size", help="Training batch size", type=int, default=BATCH_SIZE)
    parser.add_argument("-no_retrain", help="Add flag if no retraining is needed", action="store_true", default=False)
    parser.add_argument("-oracle", help="Add flag if using oracle mode", action="store_true", default=False)
    parser.add_argument("-model_algo", help="Machine Learning algorithm to use", choices=['nn_reg', 'nn_clf', 'rf_reg', 'rf_clf'], type=str, required=True)
    parser.add_argument("-model_name", help="Model's name upon saving", type=str, required=True)
    parser.add_argument("-dd_algo", help="Drift detection algorithm to use", type=str, default='')
    parser.add_argument("-output", help="Output CSV file name upon saving", type=str, required=True)
    args = parser.parse_args()
    
    if (not args.no_retrain and args.dd_algo == ''):
        print('Retrain has to choose the DD algo!')
        exit()

    path = args.path
    dataset_name = args.dataset_name
    data_train_duration_min = args.data_train_duration_min
    data_retrain_duration_min = args.data_retrain_duration_min
    data_eval_duration_min = args.data_eval_duration_min
    roc_auc_threshold = args.roc_auc_threshold
    batch_size = args.batch_size
    model_algo = args.model_algo

    # Prepare Output Directory
    timestamp = int(time.time_ns())
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), 'results', str(timestamp))
    output_stats = os.path.join(output_dir, args.output+'.csv')
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

    print("Batch size =", batch_size)

    pending_training_data = None
    num_rows = []
    stats_df = pd.DataFrame(columns=['mode', 'minute', 'us', 'roc_auc', 'pr_auc', 'f1_score', 'accuracy', 'fnr', 'fpr', 'retrain'])
    drift_data = pd.DataFrame(columns=['p0', 'p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90', 'p100', 'drift', 'f1'])

    output_cycle = os.path.join(output_dir, 'train')
    create_output_dir(output_cycle)

    # Logging
    params = []
    if args.oracle:
        models = []

    # Start Training and Evaluating
    curr_ts = data_train_duration_ms
    count_retrain = 0
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
                x = dataset_train.copy(deep=True).drop(columns=["ts_record", "reject", "latency"], axis=1)
                y = dataset_train["reject"].copy(deep=True)

                # Get training throughput data for DD model dataset
                initial_thpt = dataset_train['size']/dataset_train['latency']
                summary_initial_thpt = np.array([int(np.percentile(initial_thpt, x)) for x in range(0, 101, 10)])
                if args.dd_algo == 'heuristics-based-labeler':
                    initial_lat = dataset_train['latency']
                
                # Train if model doesn't exist
                # if not (os.path.isfile(os.path.join(os.path.dirname(output_dir), args.model_name + '_norm.joblib')) and os.path.isfile(os.path.join(os.path.dirname(output_dir), args.model_name + ('.keras' if 'nn_' in model_algo else '.joblib')))):
                # Specific output directory
                start_time = default_timer()
                train_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                cpu_times = psutil.cpu_times()
                model_instance = models[model_algo](batch_size, x, y, 'BatchNorm')
                model_instance.train(x, y, True, os.path.join(output_cycle, args.model_name + '_0_norm.joblib'), os.path.join(output_cycle, args.model_name + ('_0.keras' if 'nn_' in model_algo else '_0.joblib')), False)
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
                
                if args.oracle:
                    models.append(model_instance)

                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
                y_pred = model_instance.pred(x_test)
                roc_auc, pr_auc, f1, acc, fnr, fpr = eval(y_test.values, [1 if y == True else 0 for y in y_pred])
                stats_df.loc[len(stats_df)] = ['train', -1, 0, roc_auc, pr_auc, f1, acc, fnr, fpr, False]

                eval_mode = True

        if eval_mode:
            # Eval Mode
            # One chunk has 5 mins of data
            for j in range(1, 6):
                print("*"*20, j)
                curr_ts += data_eval_duration_ms
                index_data_1min = (dataset['ts_record'] >= ((i-1)*5 + j) * data_eval_duration_ms).idxmax()

                if index_data_1min > 0:

                    print("Taking", ((i-1)*5 + j) * data_eval_duration_ms)
                    print("Number of data for #", ((i-1)*5) + j, "eval", index_data_1min)
                    dataset_1min = dataset[:index_data_1min]
                    dataset = dataset[index_data_1min:]
                    dataset.reset_index(drop=True, inplace=True)
                    # print(dataset_1min['ts_record'].head().tolist())
                    # print(dataset_1min['ts_record'].tail().tolist())
                    # print(dataset_1min.shape)

                    # Prepare evaluation data
                    x = dataset_1min.copy(deep=True).drop(columns=["ts_record", "reject", "latency"], axis=1)
                    y = dataset_1min["reject"].copy(deep=True)
                    print("eval", i, j, len(y))

                    # Evaluation
                    start_eval_time = default_timer()
                    eval_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                    cpu_eval_times = psutil.cpu_times()
                    y_pred = model_instance.pred(x)
                    EVAL_TIME += default_timer()-start_eval_time
                    EVAL_MEMORY_USAGE += (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0 - eval_mem)
                    EVAL_CPU_TIME += psutil.cpu_times().user-cpu_eval_times.user
                    EVAL_COUNTER += len(x)

                    roc_auc, pr_auc, f1, acc, fnr, fpr = eval(y.values, y_pred)
                    stats_df.loc[len(stats_df)] = ['test', ((i-1)*5) + j, ((i-1)*5 + j) * data_eval_duration_ms, roc_auc, pr_auc, f1, acc, fnr, fpr, False]
                    if args.oracle:
                        for model_id, model in enumerate(models):
                            y_pred = model.pred(x)
                            roc_auc, pr_auc, f1, acc, fnr, fpr = eval(y.values, y_pred)
                            stats_df.loc[len(stats_df)] = ['test_'+str(model_id), ((i-1)*5) + j, ((i-1)*5 + j) * data_eval_duration_ms, roc_auc, pr_auc, f1, acc, fnr, fpr, False]        

                    # Save dataset for future retraining (only eligible if data_retrain_duration_min mins of data passed)
                    if not args.no_retrain: # If do retraining
                        if len(num_rows) == data_retrain_duration_min:
                            num_to_remove = num_rows.pop(0)
                            pending_training_data = pending_training_data[num_to_remove:]
                        if len(num_rows) == 0:
                            pending_training_data = dataset_1min.copy(deep=True)
                        else:
                            pending_training_data = pd.concat([pending_training_data, dataset_1min], ignore_index=True)
                        num_rows.append(len(y_pred))

                        # Dataset Generation for models
                        current_thpt = pending_training_data['size']/pending_training_data['latency']
                        summary_current_thpt = np.array([int(np.percentile(current_thpt, x)) for x in range(0, 101, 10)])
                        if args.dd_algo == 'heuristics-based-labeler':
                            current_lat = pending_training_data['latency']

                        # Retraining
                        if args.oracle:
                            do_retrain = True
                        else:
                            if args.dd_algo == 'heuristics-based-labeler':
                                do_retrain = monitor(initial_lat, initial_thpt, args.dd_algo, [current_lat, current_thpt])
                            elif 'time' in args.dd_algo:
                                do_retrain = monitor(initial_thpt, current_thpt, args.dd_algo, [((i-1)*5) + j])
                            else:
                                do_retrain = monitor(initial_thpt, current_thpt, args.dd_algo)
                        if do_retrain:
                            x = pending_training_data.copy(deep=True).drop(columns=["ts_record", "reject", "latency"], axis=1)
                            y = pending_training_data["reject"].copy(deep=True)
                            # Specific output directory
                            count_retrain += 1
                            start_time = default_timer()
                            train_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                            cpu_times = psutil.cpu_times()
                            model_instance.train(x, y, True, os.path.join(output_cycle, args.model_name + '_'+str(count_retrain) + '_norm.joblib'), os.path.join(output_cycle, args.model_name + (('_'+str(count_retrain) + '.keras') if 'nn_' in model_algo else ('_'+str(count_retrain) + '.joblib'))), True if not args.oracle else False)
                            TRAIN_TIME += default_timer()-start_time
                            TRAIN_MEMORY_USAGE += (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0 - train_mem)
                            TRAIN_CPU_TIME += psutil.cpu_times().user-cpu_times.user
                            TRAIN_COUNTER += 1
                            
                            if args.oracle:
                                models.append(model_instance)
                            
                            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
                            y_pred = model_instance.pred(x_test)
                            roc_auc, pr_auc, f1, acc, fnr, fpr = eval(y_test.values, y_pred)
                            
                            stats_df.loc[len(stats_df)] = ['validation_'+str(count_retrain), ((i-1)*5) + j, ((i-1)*5 + j) * data_eval_duration_ms, roc_auc, pr_auc, f1, acc, fnr, fpr, do_retrain]

                            # Update the data
                            initial_thpt = current_thpt
                            summary_initial_thpt = summary_current_thpt
                            if args.dd_algo == 'heuristics-based-labeler':
                                initial_lat = current_lat

                        # Calculate difference of throughput percentiles
                        difference_thpt_per_percentile = [abs(i-c) for i, c in zip(summary_initial_thpt, summary_current_thpt)]
                        # The label
                        difference_thpt_per_percentile.append(do_retrain)
                        difference_thpt_per_percentile.append(f1)
                        # Log drift data
                        drift_data.loc[len(drift_data)] = difference_thpt_per_percentile
                    else:
                        # If not retrain, still log the data
                        if len(num_rows) == data_retrain_duration_min:
                            num_to_remove = num_rows.pop(0)
                            pending_training_data = pending_training_data[num_to_remove:]
                        if len(num_rows) == 0:
                            pending_training_data = dataset_1min.copy(deep=True)
                        else:
                            pending_training_data = pd.concat([pending_training_data, dataset_1min], ignore_index=True)
                        num_rows.append(len(y_pred))

                        # Dataset Generation for models
                        current_thpt = pending_training_data['size']/pending_training_data['latency']
                        summary_current_thpt = np.array([int(np.percentile(current_thpt, x)) for x in range(0, 101, 10)])

                        # Calculate difference of throughput percentiles
                        difference_thpt_per_percentile = [abs(i-c) for i, c in zip(summary_initial_thpt, summary_current_thpt)]
                        # The label
                        difference_thpt_per_percentile.append(False)
                        difference_thpt_per_percentile.append(f1)
                        # Log drift data
                        drift_data.loc[len(drift_data)] = difference_thpt_per_percentile
                
    stats_df.to_csv(output_stats)
    print("Output file =>", output_stats)

    drift_data.to_csv(output_drift_data)
    print("Output drift data =>", output_drift_data)

    params.append("-path = "+str(path))
    params.append("-dataset_name = "+str(dataset_name))
    params.append("-data_train_duration_min = "+str(data_train_duration_min))
    params.append("-data_retrain_duration_min = "+str(data_retrain_duration_min))
    params.append("-data_eval_duration_min = "+str(data_eval_duration_min))
    params.append("-roc_auc_threshold = "+str(roc_auc_threshold))
    params.append("-batch_size = "+str(batch_size))
    params.append("-no_retrain = "+str(args.no_retrain))
    params.append("-model_algo = "+str(model_algo))
    params.append("-output = "+str(args.output))
    params.append(" =====> Output = "+str(output_dir))
    params.append("Retrained = "+str(count_retrain))
    params.append("Training time total = "+str(TRAIN_TIME)+" s")
    params.append("Training time single = "+str(TRAIN_TIME/TRAIN_COUNTER)+" s")
    params.append("Training memory usage = "+str(TRAIN_MEMORY_USAGE)+" MB")
    params.append("Training CPU times usage = "+str(TRAIN_CPU_TIME)+" sCPU")
    params.append("Training counter = "+str(TRAIN_COUNTER))
    params.append("Inference time total = "+str(EVAL_TIME)+" s")
    params.append("Inference time single = "+str(EVAL_TIME/EVAL_COUNTER)+" s")
    params.append("Inference memory usage = "+str(EVAL_MEMORY_USAGE)+" MB")
    params.append("Inference CPU times usage = "+str(EVAL_CPU_TIME)+" sCPU")
    params.append("Inference counter = "+str(EVAL_COUNTER))
    write_stats(output_vars, '\n'.join(params))