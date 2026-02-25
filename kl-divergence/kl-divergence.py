import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    q = q + eps
    D_KL = 0
    for i in range(len(p)):
        if p[i] == 0:
            p[i] += eps
        D_KL += p[i] * np.log(p[i] / q[i])
    return D_KL

    # Alternative approach, more numpy heavy
    # p = np.array(p, dtype=float)
    # q = np.array(q, dtype=float)
    # q += eps
    # p /= np.sum(p)
    # q /= np.sum(q)
    # # Handling the case when p[i] = 0 by assigning 0 to them
    # D_KL = np.sum(np.where(p != 0, p * np.log(p / q), 0))
    # return D_KL