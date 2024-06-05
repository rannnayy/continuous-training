#!/usr/bin/env python3

import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
import time
import tensorflow as tf
import joblib
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
from model.classification.decisiontree import DecisionTree

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

S = {}
N = {}

class MatchMaker:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, max_leaf_nodes=100)

    def train(self, x, y, sample=True, save=True, model_path=None):
        print("Sampling...")
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=10000, random_state=42, stratify=y)
        print("After sampling =", len(x_train), set(y_train))
        self.model.fit(x_train, y_train)
        print("Done training Matchmaker")
        if save:
            joblib.dump(self.model, model_path)
    
    def get_decision_path(self, tree, x):
        path_matrix = tree.decision_path(x)
        path_keys = []
        for i in range(len(x)):
            node_index = path_matrix.indices[path_matrix.indptr[i] : path_matrix.indptr[i + 1]]
            path_keys.append('_'.join([str(val) for val in node_index]))
        return path_keys
    
    def generate(self, x, batch_index):
        # batch_index = [(start:end), (start:end)]
        global S, N
        S = {}
        for tree_id, tree in enumerate(self.model.estimators_):
            N = {}
            for batch_id, batch in enumerate(batch_index):
                if batch[1] > batch[0]:
                    data = x[batch[0]:batch[1]]
                    if len(data) > 500:
                        # print(tree_id, "= Sampling for batch", batch_id, "originally", len(data))
                        temp_df = pd.DataFrame(data)
                        temp_df = temp_df.sample(n=100, random_state=42)
                        data = temp_df.values
                        # print("Resulting length", len(data))
                    keys_decision_paths = self.get_decision_path(tree, data)
                    for key in keys_decision_paths:
                        if key not in N.keys():
                            N[key] = {}
                        if batch_id not in N[key].keys():
                            N[key][batch_id] = 0
                        N[key][batch_id] += 1 # fill in N[k][t] where k = node, t = batch ID
            S[tree_id] = {}
            for key in N.keys():
                S[tree_id][key] = {}
                for batch_id in N[key].keys():
                    S[tree_id][key][batch_id] = sum([1 if (batch_id != batch_apostrophe and N[key][batch_id] > N[key][batch_apostrophe]) else 0 for batch_apostrophe in N[key].keys()])

    def borda_count(self, rcs, rcd):
        # rcs and rcd in form of dictionary
        # sort by dictionary value and get the key
        # in descending order
        sorted_rcs_dict = {k: v for k, v in sorted(rcs.items(), key=lambda item: item[1], reverse=True)}
        sorted_rcd_dict = {k: v for k, v in sorted(rcd.items(), key=lambda item: item[1], reverse=True)}
        sorted_rcs = list(sorted_rcs_dict.keys())
        sorted_rcd = list(sorted_rcd_dict.keys())
        # print(sorted_rcs)
        # print(sorted_rcd)
        # count the borda scores for each unique element in rcs and rcd (the unique elements are same,
        # which is the batch ID)
        # Formula: len(list)-list.index(item)-1
        # r_star[batch_id] = Formula(rcs) + Formula(rcd)
        r_star = []
        for batch_id in range(len(sorted_rcs)):
            r_star.append((len(sorted_rcs) - sorted_rcs.index(batch_id) - 1) + (len(sorted_rcd) - sorted_rcd.index(batch_id) - 1))
        return r_star # not sorted
    
    def inference(self, x_sample, batch_index, rcd):
        global S
        # print("x_sample length =", len(x_sample))
        # print("x_sample =", x_sample)
        vote_batch_chosen = {}
        for id_sample in range(len(x_sample)):
            x = x_sample[id_sample:id_sample+1]
            sit = []
            for tree_id, tree in enumerate(self.model.estimators_):
                key = self.get_decision_path(tree, x.values)
                ki = []
                for batch_id, batch in enumerate(batch_index):
                    # print(batch_id, tree_id, key[0])
                    if (key[0] not in S[tree_id].keys()) or (batch_id not in S[tree_id][key[0]].keys()):
                        ki.append(0)
                    else:
                        ki.append(S[tree_id][key[0]][batch_id])
                sit.append(ki)
            rcs = {}
            for batch_id, batch in enumerate(batch_index):
                rcs[batch_id] = sum([sit[tree_id][batch_id] for tree_id in range(len(self.model.estimators_))])
            final_ranking = self.borda_count(rcs, rcd)
            chosen_batch = final_ranking.index(max(final_ranking)) # get the max value from borda count
            if chosen_batch not in vote_batch_chosen.keys():
                vote_batch_chosen[chosen_batch] = 1
            else:
                vote_batch_chosen[chosen_batch] += 1
        most_voted_sorted = {k: v for k, v in sorted(vote_batch_chosen.items(), key=lambda item: item[1], reverse=True)}
        most_voted_chosen = list(most_voted_sorted.keys())[0]

        return most_voted_chosen

models = {
    'nn_reg': NeuralNetworkReg,
    'nn_clf': NeuralNetworkClf,
    # 'rf_reg': RandomForestReg,
    'rf_clf': RandomForestClf,
    'dt_clf': DecisionTree,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", help="Path to the dataset folder", type=str, required=True)
    parser.add_argument("-dataset_name", help="Name of dataset file", type=str, required=True)
    parser.add_argument("-data_train_duration_min", help="Data duration used for training", type=int, default=DATA_TRAIN_DURATION_MIN)
    parser.add_argument("-data_retrain_duration_min", help="Data duration used for retraining (If not set, the window will be set equal to data_train_duration_min)", type=int, default=DATA_RETRAIN_DURATION_MIN)
    parser.add_argument("-data_eval_duration_min", help="Data duration used for each evaluation", type=int, default=DATA_EVAL_DURATION_MIN)
    parser.add_argument("-eval_period", help="Period of data for evaluation (hour)", type=float, default=1)
    parser.add_argument("-train_period", help="Period of data for training (hour)", type=float, default=0.5)
    parser.add_argument("-batch_size", help="Training batch size", type=int, default=BATCH_SIZE)
    parser.add_argument("-no_retrain", help="Add flag if no retraining is needed", action="store_true", default=False)
    parser.add_argument("-model_algo", help="Machine Learning algorithm to use", choices=['nn_reg', 'nn_clf', 'rf_reg', 'rf_clf', 'dt_clf'], type=str, required=True)
    parser.add_argument("-model_name", help="Model's name upon saving", type=str, required=True)
    parser.add_argument("-output", help="Output CSV file name upon saving", type=str, required=True)
    args = parser.parse_args()
    
    path = args.path
    dataset_name = args.dataset_name
    data_train_duration_min = args.data_train_duration_min
    data_retrain_duration_min = args.data_retrain_duration_min
    data_eval_duration_min = args.data_eval_duration_min
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
    list_of_models = []

    # Start Training and Evaluating
    curr_ts = data_train_duration_ms
    index_data_train = 0
    index_data_1min = 0
    old_index_data_1min = 0
    eval_mode = False
    rcd = [] # concept drift
    batch_index = []
    count_retrain = 0

    # Init RF model
    RF = MatchMaker()

    print(1, int(args.eval_period*12)+1)

    # 1 batch = 1 minute

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

        # Eval Mode
        # One chunk has 5 mins of data
        for j in range(1, 6):
            print("*"*20, j)
            curr_ts += data_eval_duration_ms
            index_data_1min = (dataset['ts_record'] >= ((i-1)*5 + j) * data_eval_duration_ms).idxmax()

            if index_data_1min > old_index_data_1min:

                print("Taking", ((i-1)*5 + j) * data_eval_duration_ms)
                print("Number of data for #", ((i-1)*5) + j, "eval", index_data_1min)
                dataset_1min = dataset[old_index_data_1min:index_data_1min]
                # dataset = dataset[index_data_1min:]
                # dataset.reset_index(drop=True, inplace=True)

                # Prepare evaluation data
                x = dataset_1min.copy(deep=True).drop(columns=["ts_record", "reject", "latency"], axis=1)
                y = dataset_1min["reject"].copy(deep=True)
                print("eval", i, j, len(y))

                # Training Period
                if i <= args.train_period * 12:
                    batch_index.append([old_index_data_1min, index_data_1min])

                    # Get training throughput data for DD model dataset
                    initial_thpt = dataset_1min['size']/dataset_1min['latency']
                    summary_initial_thpt = np.array([int(np.percentile(initial_thpt, x)) for x in range(0, 101, 10)])
                    
                    # Train
                    start_time = default_timer()
                    train_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                    cpu_times = psutil.cpu_times()
                    model_instance = models[model_algo](batch_size, x, y)
                    model_instance.train(x, y, True, None, os.path.join(output_cycle, args.model_name + ('_' + str((i-1)*5 + j) + '.keras' if 'nn_' in model_algo else '_' + str((i-1)*5 + j) + '.joblib')), False)
                    TRAIN_TIME += default_timer()-start_time
                    TRAIN_MEMORY_USAGE += (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0 - train_mem)
                    TRAIN_CPU_TIME += psutil.cpu_times().user-cpu_times.user
                    TRAIN_COUNTER += 1
                    print("train", len(y))
                    
                    list_of_models.append(model_instance)

                    miu_val = sum([(1-val)**2 for val in model_instance.pred_proba(x, y)])
                    rcd.append(1-(miu_val/len(y)))

                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
                    y_pred = model_instance.pred(x_test)
                    roc_auc, pr_auc, f1, acc, fnr, fpr = eval(y_test.values, y_pred)
                    stats_df.loc[len(stats_df)] = ['train_' + str((i-1)*5 + j), -1, 0, roc_auc, pr_auc, f1, acc, fnr, fpr, False]

                    if i == args.train_period * 12 and j == 5:
                        # Train RF model
                        # Prepare whole batch data
                        x = dataset.copy(deep=True).drop(columns=["ts_record", "reject", "latency"], axis=1)
                        y = dataset["reject"].copy(deep=True)
                        print("train whole batch", i, j, len(y))

                        RF.train(x.values, y.values, True, True, os.path.join(output_dir, 'matchmaker.joblib'))
                        RF.generate(x.values, batch_index)

                else:
                    # Evaluation
                    start_eval_time = default_timer()
                    eval_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                    cpu_eval_times = psutil.cpu_times()
                    # Get sample to know which model batch to use
                    model_index = RF.inference(x[:5], batch_index, {rcd_index:rcd[rcd_index] for rcd_index in range(len(rcd))})
                    model_instance = list_of_models[model_index]
                    y_pred = model_instance.pred(x)
                    EVAL_TIME += default_timer()-start_eval_time
                    EVAL_MEMORY_USAGE += (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0 - eval_mem)
                    EVAL_CPU_TIME += psutil.cpu_times().user-cpu_eval_times.user
                    EVAL_COUNTER += len(x)

                    roc_auc, pr_auc, f1, acc, fnr, fpr = eval(y.values, y_pred)
                    stats_df.loc[len(stats_df)] = ['test', ((i-1)*5) + j, ((i-1)*5 + j) * data_eval_duration_ms, roc_auc, pr_auc, f1, acc, fnr, fpr, False]
                    
                    # Dataset Generation for models
                    current_thpt = dataset_1min['size']/dataset_1min['latency']
                    summary_current_thpt = np.array([int(np.percentile(current_thpt, x)) for x in range(0, 101, 10)])
                
                    # Calculate difference of throughput percentiles
                    difference_thpt_per_percentile = [abs(i-c) for i, c in zip(summary_initial_thpt, summary_current_thpt)]
                    # The label
                    difference_thpt_per_percentile.append(False)
                    difference_thpt_per_percentile.append(f1)
                    # Log drift data
                    drift_data.loc[len(drift_data)] = difference_thpt_per_percentile
                    
                old_index_data_1min = index_data_1min
                
    stats_df.to_csv(output_stats)
    print("Output file =>", output_stats)

    drift_data.to_csv(output_drift_data)
    print("Output drift data =>", output_drift_data)

    params.append("-path = "+str(path))
    params.append("-dataset_name = "+str(dataset_name))
    params.append("-data_train_duration_min = "+str(data_train_duration_min))
    params.append("-data_retrain_duration_min = "+str(data_retrain_duration_min))
    params.append("-data_eval_duration_min = "+str(data_eval_duration_min))
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