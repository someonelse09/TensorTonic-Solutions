import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    PE = np.zeros((seq_length, d_model))
    for pos in range(seq_length):
        for i in range(d_model):
            if i % 2 == 1:
                PE[pos, i] += np.cos(pos / (10 ** (8 * i / d_model)))
            else:
                PE[pos, i] += np.sin(pos / (10 ** (8 * i / d_model)))
    return PE

    """
    Alternative approaches
    # 1
    pos = np.arange(seq_length)[:, np.newaxis]
    i = np.arange(d_model)
    div_term = 10 ** (8 * i / d_model)

    pe = np.where(i % 2 == 0, np.sin(pos / div_term), np.cos(pos / div_term))
    return pe

    # 2
    pos = np.arange(seq_length)
    i = np.arange(d_model)
    pe = np.zeros((seq_length, d_model))
    div_term = 10 ** (8 * i / d_model)

    pe[:, 0::2] = np.sin(pos[:, np.newaxis] / div_term[0::2]) # even indices
    pe[:, 1::2] = np.cos(pos[:, np.newaxis] / div_term[1::2]) # odd indices
    """