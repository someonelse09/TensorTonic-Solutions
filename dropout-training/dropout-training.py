import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x, dtype=float)
    if not rng is None:
        random_values = rng.random(x.shape)
    else:
        random_values = np.random.random(x.shape)
    keep = random_values < (1 - p)
    dropout_pattern = keep.astype(float) * (1 / (1 - p))
    output = x * dropout_pattern
    return output, dropout_pattern
    
    