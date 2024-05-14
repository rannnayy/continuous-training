#!/usr/bin/env python3

import sys
import os
sys.path.append('../../utils')
print(os.getcwd())
import default_ip_finder

def dd_ip(data_initial, data_current):
    # Get the IP of initial and latest data
    ip_initial, _ = default_ip_finder.tangent_based(data_initial)
    ip_current, _ = default_ip_finder.tangent_based(data_current)
    
    # Intuition: if the latest data doesn't shift from the initial data
    # it means that the IP will still be more or less the same.
    # That's why, the number of rejections will be much or less the same too
    # Now, get the rejection using the old IP towards old data and new data
    initial_to_current_rejection = [data > ip_initial for data in data_current]
    current_to_current_rejection = [data > ip_current for data in data_current]
    
    # Now, count the rejections of each category
    len_initial_reject = initial_to_current_rejection.count(True)
    len_current_reject = current_to_current_rejection.count(True)

    # It is a drift if the number of rejection becomes scarce
    return len_initial_reject > len_current_reject