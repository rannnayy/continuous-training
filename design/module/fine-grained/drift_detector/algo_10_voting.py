import pandas as pd
import numpy as np

from drift_detection.algo_0_ip import dd_ip
from drift_detection.algo_1_lat_slope import dd_lat_slope
from drift_detection.algo_2_thpt_slope import dd_thpt_slope
from drift_detection.algo_3_lat_thpt_heuristics import dd_lat_thpt_heuristics
from drift_detection.algo_4_ks_test import dd_ks_test
from drift_detection.algo_5_page_hinkley import dd_page_hinkley
from drift_detection.algo_6_psi import dd_psi
from drift_detection.algo_7_kullback_leibler import dd_kl
from drift_detection.algo_8_jensen_shannon_dist import dd_js
from drift_detection.algo_9_model import dd_model

DD_ALGO = {
    'algo_0_ip': dd_ip,
    'algo_1_lat_slope': dd_lat_slope,
    'algo_2_thpt_slope': dd_thpt_slope,
    'algo_3_lat_thpt_heuristics': dd_lat_thpt_heuristics,
    'algo_4_ks_test': dd_ks_test,
    'algo_5_page_hinkley': dd_page_hinkley,
    'algo_6_psi': dd_psi,
    'algo_7_kl': dd_kl,
    'algo_8_js': dd_js,
    'algo_9_model': dd_model
}

def dd_voting(initial_lat, curr_lat, initial_size, curr_size, initial_thpt, curr_thpt, initial_reject, pred_reject, num_latest, dd, mode):
    count_drift = 0
    count_notdrift = 0
    
    for drift_detection_algo in dd:
        if 'algo_0_ip' in drift_detection_algo:
            isDrift = dd_ip(int(drift_detection_algo.split('_')[-1]), initial_lat, curr_lat)
            
        elif drift_detection_algo == 'algo_1_lat_slope':
            isDrift = dd_lat_slope(pd.concat([initial_lat, curr_lat]), num_latest)
            
        elif drift_detection_algo == 'algo_2_thpt_slope':
            isDrift = dd_thpt_slope(pd.concat([initial_thpt, curr_thpt]), num_latest)
        
        elif drift_detection_algo == 'algo_3_lat_thpt_heuristics':
            isDrift = dd_lat_thpt_heuristics(initial_lat, initial_thpt, curr_lat, curr_thpt)
        
        elif 'algo_9_model' in drift_detection_algo:
            p_thpt_diff = [abs(i-c) for i, c in zip(initial_thpt, curr_thpt)]
            
            dd_model_name = drift_detection_algo.rsplit('_', 1)[1]
            isDrift = dd_model(dd_model_name, p_thpt_diff)
            
        else:
            variable = drift_detection_algo.split('_')[-1]
            
            if variable == 'lat':
                isDrift = DD_ALGO[drift_detection_algo.rsplit('_', 1)[0]](initial_lat, curr_lat)
            elif variable == 'thpt':
                isDrift = DD_ALGO[drift_detection_algo.rsplit('_', 1)[0]](initial_thpt, curr_thpt)
            elif variable == 'reject':
                isDrift = DD_ALGO[drift_detection_algo.rsplit('_', 1)[0]](initial_reject, pred_reject)
        
        count_drift += 1 if isDrift else 0
        count_notdrift += 1 if not isDrift else 0
    
    if mode == 'majority':
        return count_drift > count_notdrift
    elif mode == 'all':
        return count_notdrift == 0