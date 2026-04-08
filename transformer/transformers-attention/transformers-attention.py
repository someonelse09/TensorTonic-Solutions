import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    #  softmax(Q @ K.T / math.sqrt(d_k)) @ V
    d_k = Q.shape[2]
    # Computing Scores tensor
    # We transpose the last two dims (-2 and -1) of K to align them for multiplication
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Applying Softmax to scores
    # This turns scores into values between 0 and 1 that sum to 1
    weights = F.softmax(scores, dim=-1)

    return torch.matmul(weights, V)