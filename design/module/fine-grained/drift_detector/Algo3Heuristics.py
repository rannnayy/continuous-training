#!/usr/bin/env python3

LAT_INCREASE_RATE = 1.5
THPT_DROP_RATE = 1.7

def dd_heuristics(initial_lat, initial_thpt, current_lat, current_thpt):
    latency_increase = current_lat.mean()/initial_lat.mean()
    throughput_drop = initial_thpt.mean()/current_thpt.mean()
    
    if (latency_increase >= LAT_INCREASE_RATE or throughput_drop >= THPT_DROP_RATE):
        # print("====================> Data Drift Detected")
        return True

    return False