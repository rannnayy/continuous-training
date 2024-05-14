#!/usr/bin/env python3

import sys
sys.path.append('../../commonutils')
import default_ip_finder

def dd_update_ip_0(data_initial, data_current):
    # Always update IP
    ip_current, _ = default_ip_finder.tangent_based(data_current)
    
    return ip_current

def dd_update_ip_1(data_initial, data_current):
    # Algo_0_ip_5 ++
    ip_initial, _ = default_ip_finder.tangent_based(data_initial)
    ip_current, _ = default_ip_finder.tangent_based(data_current)
    
    initial_to_current_rejection = [data > ip_initial for data in data_current]
    current_to_current_rejection = [data > ip_current for data in data_current]
    
    len_initial_reject = initial_to_current_rejection.count(True)
    len_current_reject = current_to_current_rejection.count(True)
    
    if len_initial_reject > len_current_reject:
        return ip_current
    else:
        return ip_initial

DD_IP_FUNCTIONS = {
    0: dd_update_ip_0,
    1: dd_update_ip_1
}

def dd_update_ip(algo, data_initial, data_current):
    return DD_IP_FUNCTIONS[algo](data_initial, data_current)