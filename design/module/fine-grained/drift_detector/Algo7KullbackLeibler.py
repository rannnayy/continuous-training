from math import log
 
# calculate the kl divergence
def kl_divergence(p, q):
    # return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)) if p!=0 and q!=0)
    return sum(p * log(p / q) for p, q in zip(p, q) if p != 0 and q != 0)

def dd_kl(p, q):
    len_calc = min(len(p), len(q))
    p = p[-len_calc:].tolist()
    q = q[-len_calc:].tolist()
    kl_val = kl_divergence(p, q)
    
    if kl_val < 0.1:
        return False
    elif kl_val < 0.2 and kl_val >= 0.1:
        return True
    elif kl_val >= 0.2:
        return True