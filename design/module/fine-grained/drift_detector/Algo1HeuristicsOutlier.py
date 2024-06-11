#!/usr/bin/env python3

# Tolerable spikes
SPIKE_THRESHOLD = 1 # in percentage

def count_iqr(data):
    q75 = data.quantile(0.75)
    q25 = data.quantile(0.25)
    iqr = q75 - q25
    iqr_1_5 = 1.5*iqr
    lower_fence = q25 - iqr_1_5
    upper_fence = q75 + iqr_1_5
    return lower_fence, upper_fence

def dd_lat_slope(data_initial, data_current):
    lower_fence_initial, upper_fence_initial = count_iqr(data_initial)
    lower_fence_current, upper_fence_current = count_iqr(data_current)
    
    spike_initial = len([i for i in data_initial if i > upper_fence_initial or i < lower_fence_initial])/len(data_initial)
    spike_current = len([i for i in data_current if i > upper_fence_current or i < lower_fence_current])/len(data_current)
    
    if (abs(spike_current-spike_initial) > SPIKE_THRESHOLD/100):
        # TODO: Adjust with elbow method later!
        return True
    
    return False