#!/usr/bin/env python3

# Tolerable spikes
SPIKE_THRESHOLD = 0.25 # in percentage

def count_quartile(data):
    q75 = data.quantile(0.75)
    q25 = data.quantile(0.25)
    return q25, q75

def dd_lat_slope(data_initial, data_current):
    lower_fence_initial, upper_fence_initial = count_quartile(data_initial)
    lower_fence_current, upper_fence_current = count_quartile(data_current)
    
    spike_initial = len([i for i in data_initial if i > upper_fence_initial or i < lower_fence_initial])/len(data_initial)
    spike_current = len([i for i in data_current if i > upper_fence_current or i < lower_fence_current])/len(data_current)
    
    if (abs(spike_current-spike_initial) > SPIKE_THRESHOLD/100):
        # TODO: Adjust with elbow method later!
        return True
    
    return False