import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    if not isinstance(A, list) or not isinstance(A[0], list):
        return None
    if len(A) != len(A[0]) or np.linalg.det(A) == 0:
        return None
    return np.linalg.inv(A)    
