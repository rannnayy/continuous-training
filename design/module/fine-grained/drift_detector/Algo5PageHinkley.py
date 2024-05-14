# Modified Page-Hinkley algorithm for drift detection

from statistics import mean

def calculate_new_mean(mean, n, new_value, n_new_value):
    return ((mean * n) + (new_value * n_new_value)) / (n + n_new_value)

def dd_page_hinkley(data_initial, data_current, alpha=1-0.0001, delta=0.005, threshold=100):
    increase = [b - a if (b - a > 0) else 0 for a, b in zip(data_initial, data_initial[1:])]
    decrease = [a - b if (a - b > 0) else 0 for a, b in zip(data_initial, data_initial[1:])]
    
    new_mean = calculate_new_mean(mean(data_initial), len(data_initial), mean(data_current), len(data_current))
    
    dev_increase = [x - new_mean for x in data_current]
    
    sum_increase = alpha * sum(increase) + sum(dev_increase) - delta
    sum_decrease = alpha * sum(decrease) + sum(dev_increase) + delta
    
    min_increase = min(increase)
    max_decresase = max(decrease)
    
    if sum_increase < min_increase:
        min_increase = sum_increase
    
    if sum_decrease > max_decresase:
        max_decresase = sum_decrease
    
    test_increase = sum_increase - min_increase
    test_decrease = max_decresase - sum_decrease
    
    return test_increase > threshold or test_decrease > threshold