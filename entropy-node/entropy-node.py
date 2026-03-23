import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.array(y, dtype=float)
    n = len(y)
    if n == 0:
        return 0.0
    _, class_frequencies = np.unique(y, return_counts=True)
    p_i = class_frequencies / n
    p_i = p_i[p_i > 0]
    return -np.sum(p_i * np.log2(p_i))