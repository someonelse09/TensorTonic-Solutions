import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # X has shape (N, D) from question decription
    X = np.asarray(X)

    # Return None for invalid input (N < 2 or not 2D)
    if X.ndim != 2:
        return None
    N, D = X.shape
    if N < 2:
        return None
    # Center the data by subtracting the mean from each column (feature)
    mu = np.mean(X, axis=0)
    X_centered = X - mu

    # Applying the formula: (1 / N - 1) * X^T * X
    covariance_matrix = (1 / (N - 1)) * (X_centered.T @ X_centered)

    return covariance_matrix