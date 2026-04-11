import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    max_len = max_len if max_len is not None else max(len(seq) for seq in seqs)
    result = np.zeros((len(seqs), max_len), dtype = int)
    row = 0
    for seq in seqs:
        if len(seq) == max_len:
            result[row, :] = np.array(seq)

        elif len(seq) > max_len:
            result[row, :] = np.array(seq[:max_len])

        else:
            result[row, :len(seq)] = np.array(seq)
            d = [pad_value] * (max_len - len(seq))
            result[row, len(seq):] = np.array(d)
        row += 1

    return result


"""
Another cleaner approach:

import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    # 1. Determine max_len once
    if max_len is None:
        if not seqs: return np.array([])
        max_len = max(len(seq) for seq in seqs)
    
    # 2. Preallocate with pad_value directly
    result = np.full((len(seqs), max_len), pad_value, dtype=int)
    
    # 3. Vectorized filling
    for row, seq in enumerate(seqs):
        length = len(seq)
        if length > 0:
            # Only slice up to max_len if needed
            result[row, :min(length, max_len)] = seq[:max_len]
            
    return result
"""