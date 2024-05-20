# Kalmogorov-Smirnov Test for detecting data drift

import numpy as np
from scipy import stats

def compute_cdf(x):
    x = np.sort(x)
    return 1. * np.arange(len(x)) / float(len(x) - 1)

def cdf(x, data_concat):
    return [np.round(stats.percentileofscore(x, value)/100, 1) for value in data_concat]

def compute_ks_statistic(cdf1, cdf2):
    return np.max(np.abs(np.subtract(cdf1, cdf2)))

def dd_ks_test(data_initial, data_current):
    # function to detect drift
    # OLD
    # data_concat = np.sort(np.concatenate((data_initial, data_current)))
    # cdf_initial = cdf(data_initial, data_concat)
    # cdf_current = cdf(data_current, data_concat)
    
    # difference = compute_ks_statistic(cdf_initial, cdf_current)
    
    # critical_value = 1.36 * np.sqrt(len(data_initial)**-1 + len(data_current)**-1)
    # drift = difference > critical_value

    # FASTER
    ks_stats = stats.ks_2samp(data_initial, data_current).pvalue
    drift = ks_stats < 0.05
    
    return drift