#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import joblib
import numpy as np
import pandas as pd
import warnings
from scipy.optimize import minimize
import scipy.stats
from scipy.special import log_ndtr
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, classification_report, average_precision_score, make_scorer, mean_squared_error, r2_score, mean_absolute_error, mean_squared_error, f1_score

tf.config.set_visible_devices([], 'GPU')

############################ From NURD

flashnet_feat_col = [
    "size",
    "queue_len",
    "prev_queue_len_1",
    "prev_queue_len_2",
    "prev_queue_len_3",
    "prev_latency_1",
    "prev_latency_2",
    "prev_latency_3",
    "prev_throughput_1",
    "prev_throughput_2",
    "prev_throughput_3",
]

ori_feat_col = [
    "size",
    "offset"
]

task_cols = [
    "latency",
    "size",
    "queue_len",
    "prev_queue_len_1",
    "prev_queue_len_2",
    "prev_queue_len_3",
    "prev_latency_1",
    "prev_latency_2",
    "prev_latency_3",
    "prev_throughput_1",
    "prev_throughput_2",
    "prev_throughput_3",
]

ori_task_cols = [
    "latency",
    "size",
    "offset"
]

def split_left_right_censored(x, y, cens):
    counts = cens.value_counts()
    if -1 not in counts and 1 not in counts:
        warnings.warn("No censored observations; use regression methods for uncensored data")
    xs = []
    ys = []

    for value in [-1, 0, 1]:
        if value in counts:
            split = cens == value
            y_split = np.squeeze(y[split].values)
            x_split = x[split].values

        else:
            y_split, x_split = None, None
        xs.append(x_split)
        ys.append(y_split)
    return xs, ys


def tobit_neg_log_likelihood(xs, ys, params):
    x_left, x_mid, x_right = xs
    y_left, y_mid, y_right = ys

    b = params[:-1]
    # s = math.exp(params[-1])
    s = params[-1]

    to_cat = []

    cens = False
    if y_left is not None:
        cens = True
        left = (y_left - np.dot(x_left, b))
        to_cat.append(left)
    if y_right is not None:
        cens = True
        right = (np.dot(x_right, b) - y_right)
        to_cat.append(right)
    if cens:
        concat_stats = np.concatenate(to_cat, axis=0) / s
        log_cum_norm = scipy.stats.norm.logcdf(concat_stats)  # log_ndtr(concat_stats)
        cens_sum = log_cum_norm.sum()
    else:
        cens_sum = 0

    if y_mid is not None:
        mid_stats = (y_mid - np.dot(x_mid, b)) / s
        mid = scipy.stats.norm.logpdf(mid_stats) - math.log(max(np.finfo('float').resolution, s))
        mid_sum = mid.sum()
    else:
        mid_sum = 0

    loglik = cens_sum + mid_sum

    return - loglik


def tobit_neg_log_likelihood_der(xs, ys, params):
    x_left, x_mid, x_right = xs
    y_left, y_mid, y_right = ys

    b = params[:-1]
    # s = math.exp(params[-1]) # in censReg, not using chain rule as below; they optimize in terms of log(s)
    s = params[-1]

    beta_jac = np.zeros(len(b))
    sigma_jac = 0

    if y_left is not None:
        left_stats = (y_left - np.dot(x_left, b)) / s
        l_pdf = scipy.stats.norm.logpdf(left_stats)
        l_cdf = log_ndtr(left_stats)
        left_frac = np.exp(l_pdf - l_cdf)
        beta_left = np.dot(left_frac, x_left / s)
        beta_jac -= beta_left

        left_sigma = np.dot(left_frac, left_stats)
        sigma_jac -= left_sigma

    if y_right is not None:
        right_stats = (np.dot(x_right, b) - y_right) / s
        r_pdf = scipy.stats.norm.logpdf(right_stats)
        r_cdf = log_ndtr(right_stats)
        right_frac = np.exp(r_pdf - r_cdf)
        beta_right = np.dot(right_frac, x_right / s)
        beta_jac += beta_right

        right_sigma = np.dot(right_frac, right_stats)
        sigma_jac -= right_sigma

    if y_mid is not None:
        mid_stats = (y_mid - np.dot(x_mid, b)) / s
        beta_mid = np.dot(mid_stats, x_mid / s)
        beta_jac += beta_mid

        mid_sigma = (np.square(mid_stats) - 1).sum()
        sigma_jac += mid_sigma

    combo_jac = np.append(beta_jac, sigma_jac / s)  # by chain rule, since the expression above is dloglik/dlogsigma

    return -combo_jac


class TobitModel:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.ols_coef_ = None
        self.ols_intercept = None
        self.coef_ = None
        self.intercept_ = None
        self.sigma_ = None

    def fit(self, x, y, cens, verbose=False):
        """
        Fit a maximum-likelihood Tobit regression
        :param x: Pandas DataFrame (n_samples, n_features): Data
        :param y: Pandas Series (n_samples,): Target
        :param cens: Pandas Series (n_samples,): -1 indicates left-censored samples, 0 for uncensored, 1 for right-censored
        :param verbose: boolean, show info from minimization
        :return:
        """
        x_copy = x.copy()
        if self.fit_intercept:
            x_copy.insert(0, 'intercept', 1.0)
        else:
            x_copy.scale(with_mean=True, with_std=False, copy=False)
        init_reg = LinearRegression(fit_intercept=False).fit(x_copy, y)
        b0 = init_reg.coef_
        y_pred = init_reg.predict(x_copy)
        resid = y - y_pred
        resid_var = np.var(resid)
        s0 = np.sqrt(resid_var)
        params0 = np.append(b0, s0)
        xs, ys = split_left_right_censored(x_copy, y, cens)

        result = minimize(lambda params: tobit_neg_log_likelihood(xs, ys, params), params0, method='BFGS',
                          jac=lambda params: tobit_neg_log_likelihood_der(xs, ys, params), options={'disp': verbose})
        if verbose:
            print(result)
        self.ols_coef_ = b0[1:]
        self.ols_intercept = b0[0]
        if self.fit_intercept:
            self.intercept_ = result.x[1]
            self.coef_ = result.x[1:-1]
        else:
            self.coef_ = result.x[:-1]
            self.intercept_ = 0
        self.sigma_ = result.x[-1]
        return self

    def predict(self, x):
        return self.intercept_ + np.dot(x, self.coef_)

    def score(self, x, y, scoring_function=mean_absolute_error):
        y_pred = np.dot(x, self.coef_)
        return scoring_function(y, y_pred)

############## utils_ts.py
def fun_df_cumsum(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Convert interval-wise time series data to cumulative time series data.
        Args:
            df: interval-wise time series data   
        return:
            df_new: cumulative time series data    
    '''
    fields = list(df)
    total_period = sum(df.ET-df.ST)  ## total trace period as latency
    task_weight = (df.ET-df.ST)/total_period
    task_weight = np.diag(task_weight.values)
    df_new = task_weight.dot(df)
    df_new = pd.DataFrame(data=df_new, columns=fields)
    return df_new.cumsum(), total_period

def fun_cum_vec(v: np.array)-> np.array:
    '''
    Get cumulated columns for plotting CDF
    '''
    vid = np.argmax(v==1)
    if np.sum(v)!=0:
        v[vid:]=1
    else:
        v[-1]=1
    return v

def get_TPR_FPR(pred, alpha, n_stra):
    '''
    Get true positive and false positive rate from prediction matrix/list
    pred: prediction. 
    alpha: threshold.
    n_stra: number of true stragglers.
    '''
    aa = np.array(pred)
    ppdd1 = np.apply_along_axis(max, 0, aa)
    aa = aa >= alpha
    aa = aa.astype(int) 
    aa_cum = np.apply_along_axis(fun_cum_vec, 0, aa)
    stra_cum = aa_cum[:, :n_stra]
    gap_cum = aa_cum[:, n_stra:]
    tpr = sum(stra_cum[-2])*100/len(stra_cum[-2])
    fpr = sum(gap_cum[-2])*100/len(gap_cum[-2])
    
    true = np.asarray([1]*len(stra_cum[-2]) + [0] * len(gap_cum[-2]))
    auc = roc_auc_score(true, ppdd1)
    
    ppdd2 = np.concatenate([stra_cum[-2], gap_cum[-2]])    
    f1 = f1_score(true, ppdd2)
    
    tn, fp, fn, tp = confusion_matrix(true, ppdd2).ravel()
    fnr = fn/(fn+tp)
    
    return tpr, fpr, fnr, auc, f1

def get_PCT_new(kl_pred: list, y_true: np.array, n_stra:int, alpha: float, bins: list) -> list:
    """Get recall between prediction and groundtruth

    Args:    
      y_true: groundtruth real value array.
      n_stra: number of true stragglers.
      kl_pred: prediction matrix.
      bins: bin size interval.

    Returns:
      recall: recall for [90,95), [95, 99), [99+]
    """
    aa = np.array(kl_pred)[:,:n_stra]
    aa = aa >= alpha
    aa = aa.astype(int)
    aa_cum = np.apply_along_axis(fun_cum_vec, 0, aa)
    y_pred_b = aa_cum[-2]
    y_pos_sort = np.argsort(y_true)
    s95  = sum(y_pred_b[y_pos_sort[0:bins[0]].tolist()])*100/(bins[0])
    s99  = sum(y_pred_b[y_pos_sort[bins[0]:bins[1]].tolist()])*100/(bins[1]-bins[0])
    s99p = sum(y_pred_b[y_pos_sort[bins[1]:bins[2]].tolist()])*100/(bins[2]-bins[1])

    return [s95, s99, s99p]

def get_PCT(kl_pred: list, y_true: np.array, alpha: float, bins: list) -> list:
    """Get recall between prediction and groundtruth

    Args:    
      y_true: groundtruth real value array.
      kl_pred: prediction matrix.
      bins: bin size interval.

    Returns:
      recall: recall for [90,95), [95, 99), [99+]
    """
    aa = np.array(kl_pred)
    aa = aa >= alpha
    aa = aa.astype(int)
    aa_cum = np.apply_along_axis(fun_cum_vec, 0, aa)
    y_pred_b = aa_cum[-2]
    y_pos_sort = np.argsort(y_true)
    s95  = sum(y_pred_b[y_pos_sort[0:bins[0]].tolist()])*100/(bins[0])
    s99  = sum(y_pred_b[y_pos_sort[bins[0]:bins[1]].tolist()])*100/(bins[1]-bins[0])
    s99p = sum(y_pred_b[y_pos_sort[bins[1]:bins[2]].tolist()])*100/(bins[2]-bins[1])

    return [s95, s99, s99p]

def get_FPR(pred: list, alpha: float) -> float:
    '''
    Get false positive rate from prediction matrix/list
    pred: prediction 
    alpha: threshold
    '''
    pred_b = [1 if i >= alpha else 0 for i in pred]
    return sum(pred_b)*100/len(pred)


def get_TPR(kl_pred: list, alpha: float) -> float: 
    '''
    Get true positive rate from prediction matrix/list
    kl_input: input prediction list
    alpha: threshold
    '''
    aa = np.array(kl_pred)
    aa = aa >= alpha
    aa = aa.astype(int)
    aa_cum = np.apply_along_axis(fun_cum_vec, 0, aa)
    return sum(aa_cum[-2])*100/len(aa_cum[-2])


def fun_cum_tpr(kl_input: list, alpha: float) -> np.array:
    '''
    Get cumulative probability from result matrix/list
    kl_input: input result list
    alpha: threshold
    tl_list: true task length for different tasks
    '''
    aa=np.array(kl_input)
    aa = aa >= alpha
    aa = aa.astype(int)
    aa_cum = np.apply_along_axis(fun_cum_vec, 0, aa)
    aa_cum_pr = np.sum(aa_cum, axis=1)/aa.shape[1]
    # return aa_cum_pr * 100
    return np.insert(aa_cum_pr,0,0)*100.0

############### train_nurd.py

# =========================== Hyperparameters ==================================
# using default hyperparameters descirbed in NURD repo
random_state = 42
gamma = 0
pt = 0.2    # Training set size (in NURD repo, in argument description the pt is default to be 0.2)
tail = 0.9

# 0 - 100
# p0 - p20: trainning set (train on the fast IO)
# p20 - p100: testing set  (predict on the slow IO)
# ==============================================================================

def transform_ground_truth(prediction_result, alpha):
    '''
        Return the label directly in a list to use FlashNet's metrics calculation
        Input:
            1. GT: ground truth (a.k.a. y_test/y_true)
            2. prediction_result: y_pred
            3. alpha: similar to IP threshold, to detemine slow vs fast
        Output:
            1. y_true, y_pred
    '''
    total_size = len(prediction_result)
    # print(total_size)
    y_pred = []
    for i in range(total_size):
        Predict_lable = -1
        if prediction_result[i] >= alpha:
            Predict_lable = 1   # slagger
        else:
            Predict_lable = 0
        y_pred.append(Predict_lable)
    
    return y_pred

def eval_nurd(GT, prediction_result, alpha):
    '''
        Return a dict of: [
            accuracy,
            precision, (if the ground truth is 'tail', it would be positive)
            recall,
        ]
    '''
    total_size = len(GT)
    acc_num = 0
    TP = 0     # true positive
    FP = 0     # false positive
    TN = 0     # true negative
    FN = 0     # false negative
    for i in range(total_size):
        GT_label = -1   
        if GT[i] >= alpha:
            GT_label = 1  # slagger
        else:
            GT_label = 0

        Predict_lable = -1
        if prediction_result[i] >= alpha:
            Predict_lable = 1   # slagger
        else:
            Predict_lable = 0
        if Predict_lable == GT_label and GT_label != -1:
            acc_num += 1

        # 1. judge if true positive
        if Predict_lable == 1 and GT_label == 1:
            TP += 1
        # 2. judge if false positive
        elif Predict_lable == 1 and GT_label == 0:
            FP += 1
        # 3. judge if true negative
        elif Predict_lable == 0 and GT_label == 0:
            TN += 1
        # 4. judge if false negative
        elif Predict_lable == 0 and GT_label == 1:
            FN += 1

    acc = acc_num / total_size
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return [acc, precision, recall]

############################

class Reweight:
    def __init__(self):
        # Initialize Model
        self.model_tree = GradientBoostingRegressor(random_state=random_state)
        self.model_logreg = LogisticRegression(random_state=random_state, solver='lbfgs')

    def train(self, x_train, y_train, save=True, modellogreg_path=None, modeltree_path=None, retrain=False):
        print("Preparing Data")
        dataset = x_train
        dataset = dataset.copy(deep=True).reset_index(drop=True)
        job_rawsize = dataset.shape[0]  ## Code change: Get number of tasks in a job (Here, we treat each IO request a job (which only contians one task))
        print("size of dataset: {}".format(job_rawsize))

        ## Get cumulative time series data
        list_task = [] 
        list_tp = []  ## list of total period
        list_task_compact = []  ## list of last row
        for i in range(job_rawsize):
            task = dataset.loc[i:i, :]   # code change: we treat each IO an task within a job (and a job only contain one task)
            task_new = task.copy(deep=True) # .drop(columns=["latency"], axis=1) get the dataframe of this task (except for latency)
            tp_new = sum(task.copy(deep=True)["latency"])  # get the latency of this task
            list_tp.append(tp_new)
            list_task_compact.append(task_new.iloc[-1].tolist())
            list_task.append(task_new)
        print("Finish appending tasks")

        ## Construct new non-time series data
        df_task_compact = dataset[flashnet_feat_col]
        df_task_compact['latency'] = pd.Series(np.asarray(list_tp), index=df_task_compact.index)
        df_sel = df_task_compact[task_cols]
        self.norm_min = df_sel.min()
        self.norm_max = df_sel.max()
        job = (df_sel-self.norm_min)/(self.norm_max-self.norm_min)   # apply min-max normalization to the jobs.
        job = job.dropna(axis='columns') 
        job_raw = job.reset_index(drop=True)

        ## Normalize task at different time points using final row
        list_task_nn = []
        ts_size = 0  ## max task size in a job
        cn_train = [i for i in list(job) if i not in ['latency']]  
        df_sel_min = df_sel[cn_train].min()
        df_sel_max = df_sel[cn_train].max()

        for i in range(len(list_task)):
            task = list_task[i][cn_train]
            task = (task-df_sel_min)/(df_sel_max-df_sel_min)
            if ts_size < task.shape[0]:
                ts_size = task.shape[0]
            list_task_nn.append(task)

        ## Split training and testing data
        latency = job_raw.latency.values
        ## Parameter to tune propensity score
        lat_sort = np.sort(latency)
        print("# tail :  {}".format(tail))
        cutoff = int(tail*latency.shape[0])
        print("# cutoff:  {}".format(cutoff))
        self.alpha = lat_sort.tolist()[cutoff]    # the value at p90
        print("# alpha:  {}".format(self.alpha))

        cutoff_pt = int(pt * latency.shape[0])
        alpha_pt = lat_sort.tolist()[cutoff_pt]  # the value at p4
        train_idx_init = job.index[job['latency'] < self.alpha].tolist()
        test_idx_init = job.index[job['latency'] >= self.alpha].tolist()
        train_idx_removed = job.index[(job['latency'] >= alpha_pt) & (job['latency'] < self.alpha)].tolist()
        print("# true tail: {}".format(len(test_idx_init)))

        train_idx = list(set(train_idx_init) - set(train_idx_removed))
        test_idx = test_idx_init + train_idx_removed  ## test_idx = stra_idx + gap_idx
        print("# removed: {}".format(len(train_idx_removed)))

        job = job_raw.copy() 
        job_train = job.iloc[train_idx]
        job_test = job.iloc[test_idx]
        job_test_stra = job.iloc[test_idx_init]
        job_test_gap = job.iloc[train_idx_removed]
        print("# train: {}".format(job_train.shape[0]))
        print("# test:  {}".format(job_test.shape[0]))
        print("# test stra:  {}".format(job_test_stra.shape[0]))
        print("# test gap:  {}".format(job_test_gap.shape[0]))

        X_train = job_train.to_numpy()[:,1:]
        Y_train = job_train.to_numpy()[:,0]
        # X_test = job_test.to_numpy()[:,1:]
        # Y_test = job_test.to_numpy()[:,0]
        X_test_stra = job_test_stra.to_numpy()[:,1:]
        Y_test_stra = job_test_stra.to_numpy()[:,0]
        X_test_gap = job_test_gap.to_numpy()[:,1:]
        Y_test_gap = job_test_gap.to_numpy()[:,0]

        job.loc[train_idx_init, 'Label'] = 0
        job.loc[test_idx_init, 'Label'] = 1
        # y_test_true = job.loc[test_idx, 'Label'].values ## binary groundtruth for testing tasks
        # y_stra_true = job.loc[test_idx_init, 'latency'].values ## groundtruth for straggler

        # Get latency bins, [90,95), [95, 99), [99+]
        cutoff95 = int(0.95 * latency.shape[0])
        alpha95 = lat_sort.tolist()[cutoff95]
        cutoff99 = int(0.99 * latency.shape[0])
        alpha99 = lat_sort.tolist()[cutoff99]
        # test95_idx = job.index[(job['latency'] >= alpha) & (job['latency'] < alpha95)].tolist()
        # test99_idx = job.index[(job['latency'] >= alpha95) & (job['latency'] < alpha99)].tolist()
        # test99p_idx = job.index[(job['latency'] >= alpha99)].tolist()
        # BI = np.cumsum([len(test95_idx), len(test99_idx), len(test99p_idx)])
        # print("# latency bins: {}".format(BI))

        ## Padding zero rows to unify task size
        list_task_norm = list_task_nn
        test_idx_gap = [i for i in test_idx if i not in test_idx_init]
        # list_task_nn_stra = [list_task_nn[i] for i in test_idx_init]  ## only straggler tasks
        # list_task_nn_gap = [list_task_nn[i] for i in test_idx_gap] ## nonstragglers in testing
        # list_task_nn_test = [list_task_nn[i] for i in test_idx]  ## for all test tasks

        # print("list_task_nn_stra: {}".format(list_task_nn_stra))
        # print("list_task_nn_stra.shape[0]: {}".format(list_task_nn_stra.shape[0]))
    
        ## Only care about tasks that are stragglers
        list_task_norm_stra = [list_task_norm[i] for i in test_idx_init]
        list_task_norm_gap = [list_task_norm[i] for i in test_idx_gap]
        list_task_norm_test = [list_task_norm[i] for i in test_idx]
        print(len(list_task_norm_stra),len(list_task_norm_gap),len(list_task_norm_test))

        list_task_final_stra = list_task_norm_stra
        list_task_final_gap = list_task_norm_gap

        list_task_final_test = list_task_final_stra + list_task_final_gap

        ## Get final test data
        X_test_stra_final = X_test_stra
        X_test_gap_final = X_test_gap
        X_test_final = np.concatenate((X_test_stra_final, X_test_gap_final))

        Y_test_stra_final = Y_test_stra
        Y_test_gap_final = Y_test_gap
        Y_test_final = np.concatenate((Y_test_stra_final, Y_test_gap_final))

        BI_95 = sum(((Y_test_stra_final>=self.alpha) & (Y_test_stra_final < alpha95)) * 1)
        BI_99 = sum(((Y_test_stra_final>=self.alpha) & (Y_test_stra_final < alpha99)) * 1)
        BI_99p = sum(((Y_test_stra_final>=self.alpha)) * 1)
        BI_new = np.asarray([BI_95, BI_99, BI_99p])

        # num_stra, num_gap = X_test_stra_final.shape[0], X_test_gap_final.shape
        print("Final size", X_test_stra_final.shape, X_test_gap_final.shape, X_test_final.shape)
        print("Final BI", BI_new)

        X_train_up, X_test_up, Y_train_up, Y_test_up = X_train, X_test_final, Y_train, Y_test_final
        # Y_train_pu = (Y_train_up<alpha)*1

        # lt_stra, lt_gap = len(list_task_final_stra), len(list_task_final_gap)  ## straggler/non-straggler size in testing
        # list_task_final_gap_down = list_task_final_gap
        list_task_final_test_down = list_task_final_test
        # full_idx = range(len(list_task_final_test_down))
        # eps = 0.05
        # pl_gb, pl_ipwnc, pl_ipw = [],[],[]

        print("Training", "."*20)

        # retrieve all the final test records(?), and store in np_tn_nz
        tn = [i.iloc[0].values for i in list_task_final_test_down]
        np_tn = np.asarray(tn)    
        # np_tn_nzidx = (np.where(np_tn.any(axis=1))[0]).tolist()
        np_tn_nz = np_tn[~np.all(np_tn == 0, axis=1)]
        
        cen_train = np.mean(X_train_up, axis=0)
        cen_test = np.mean(np_tn_nz, axis=0)
        rho = sum(cen_train**2)/sum((cen_train-cen_test)**2)
        
        self.delta = 1/(1+rho) - gamma
        print("delta: {}".format(self.delta))
        
        ## Base      
        self.model_tree = self.model_tree.fit(X_train_up, Y_train_up)   # 1. Train with finished task using GBDT
        # p_gb_curr = self.model_tree.predict(np_tn_nz).tolist()    # apply to the test data.
        # p_gb = [0] * len(full_idx)
        # for j in range(len(np_tn_nzidx)):
        #     p_gb[np_tn_nzidx[j]] = p_gb_curr[j]
        # pl_gb.append(p_gb)        
        
        ## IPW-NC  (proposed NURD in the paper.) (NC stands for not including reweighting based on latency space)
        # Apply ** reweighting **
        # zero_reweight_n = 0
        X = np.asarray(np.concatenate((X_train_up, np_tn_nz)))
        y = np.asarray([0] * X_train_up.shape[0] + [1] * np_tn_nz.shape[0])
        self.model_logreg = self.model_logreg.fit(X, y)
        # ps = self.model_logreg.predict_proba(X)
        # ps0 = ps[X_train_up.shape[0]:,0].copy()   # 2. Use logistic regression to estimate the propensity score.
        # p_ipwnc_curr = []

        # # Train Phase:
        # # 1. Train on fast I/O (p0-p20) with GradientBoostingRegressor, get prediction result on testing set in 'p_gb_curr'.
        # # 2. Since they train on fast I/O, they consider the current prediction on testing set is biased. So, need to do reweighting.
        # #    final_pred_testing = 'p_gb_curr' / reweighting
        # # 3. 'reweighting' is got by LogisticRegression() on all data(training & testing set) => 'reweighting' = 'ps0'
        # # 4. final prediction latency: 'p_ipwnc_curr' = 'p_gb_curr' / 'ps0'
        # for x, z in zip(p_gb_curr, ps0.tolist()):    # Handle the devided by 0 situation, give up reweight
        #     if z == 0:
        #         zero_reweight_n += 1
        #         p_ipwnc_curr.append(x)
        #     else:
        #         p_ipwnc_curr.append(x/z)    

        # # p_ipwnc_curr = [x/z for x, z in zip(p_gb_curr, ps0.tolist())]
        # p_ipwnc = [0] * len(full_idx)
        # for j in range(len(np_tn_nzidx)):
        #     p_ipwnc[np_tn_nzidx[j]] = p_ipwnc_curr[j]    
        # pl_ipwnc.append(p_ipwnc)      
        
        # ## IPW (proposed NURD in the paper.)
        # ps1 = ps[X_train_up.shape[0]:,0].copy()    
        # for i in range(len(ps1)):
        #     ps1[i] = max(eps, min(ps1[i]+self.delta, 1))     # 3. with eps to do 
                
        # p_ipw_curr = [x/(z + 0.000001) for x, z in zip(p_gb_curr, ps1.tolist())]
        # p_ipw = [0] * len(full_idx)
        # for j in range(len(np_tn_nzidx)):
        #     p_ipw[np_tn_nzidx[j]] = p_ipw_curr[j]    
        # pl_ipw.append(p_ipw)  

        # y_true, y_pred = transform_ground_truth(Y_test_up[np_tn_nzidx], p_ipwnc_curr, self.alpha)
        
        if save:
            joblib.dump(self.model_tree, modeltree_path)
            joblib.dump(self.model_logreg, modellogreg_path)

        print("Done Training", "."*20)

    def pred(self, x_test):
        # print("x_test", len(x_test))
        if len(x_test) > 0:
            x_test = x_test[flashnet_feat_col]
            x_test.reset_index(drop=True, inplace=True)
            x_test = (x_test-self.norm_min)/(self.norm_max-self.norm_min)
            x_test = x_test.to_numpy()[:,1:]
            # print("transform", x_test[0])
            p_gb = self.model_tree.predict(x_test).tolist()    # apply to the test data.
            # print("p_gb", p_gb[:5])
            
            ## IPW-NC  (proposed NURD in the paper.) (NC stands for not including reweighting based on latency space)
            # Apply ** reweighting **
            ps = self.model_logreg.predict_proba(x_test)
            ps0 = ps[:,0].copy()
            # print("ps", ps[:5])
            # print("ps0", ps0[:5])

            eps = 0.05
            ## IPW (proposed NURD in the paper.)
            ps_test = []
            for i in range(len(x_test)):
                ps_test.append(max(eps, min(ps0[i]+self.delta, 1)))     # 3. with eps to do 
            # print("ps_test", ps_test[:5])

            p_ipwnc = [x/(z + 0.000001) for x, z in zip(p_gb, ps_test)]
            # print("ipwnc", p_ipwnc[:5])

            y_pred = transform_ground_truth(p_ipwnc, self.alpha)
            # print("pred", y_pred[:5])
        
        else:
            y_pred = []

        return y_pred