import numpy as np

def port_input(ls, small=False):
    if not small:
        return np.array([int(np.percentile(ls, x)) for x in range(0, 101, 10)])
    else:
        return np.array([int(np.percentile(ls, x)) for x in [1, 25, 50, 75, 99]])