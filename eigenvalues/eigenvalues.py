import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try:
        matrix = np.asarray(matrix)
    except (ValueError, TypeError):
        return None
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    n = matrix.shape[0]
    eig_values = np.linalg.eigvals(matrix,)
    sorted_indices = np.lexsort((eig_values.imag, eig_values.real))
    eig_values = eig_values[sorted_indices]
    return eig_values
    
    