# Jensen Shannon Distance based Drift Detection Algorithm

from scipy.spatial.distance import jensenshannon

def js_dist(p, q):
    return jensenshannon(p, q) ** 2

def dd_js(p, q):
    len_calc = min(len(p), len(q))
    p = p[-len_calc:].tolist()
    q = q[-len_calc:].tolist()
    js_val = js_dist(p, q)
    
    if js_val < 0.1:
        return False
    elif js_val < 0.2 and js_val >= 0.1:
        return False
    elif js_val >= 0.2:
        return True