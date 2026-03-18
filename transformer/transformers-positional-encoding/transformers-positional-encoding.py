import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    PE = np.zeros((seq_length, d_model))
    for pos in range(seq_length):
        for i in range(d_model):
            if (i % 2 == 1):
                PE[pos, i] += np.cos(pos / (10 ** (8 * i / d_model)))
            else:
                PE[pos, i] += np.sin(pos / (10 ** (8 * i / d_model)))
    return PE