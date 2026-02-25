import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    p = np.array(p)
    q = np.array(q)
    q = q + eps
    D_KL = 0
    for i in range(len(p)):
        if p[i] == 0:
            p[i] += eps
        D_KL += (p[i] * np.log(p[i] / q[i]))
    return D_KL