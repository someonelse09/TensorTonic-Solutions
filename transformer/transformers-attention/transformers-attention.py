import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    d_k = Q.shape[2]
    #  (Q @ K.T / math.sqrt(d_k)) @ V
    return F.scaled_dot_product_attention(Q, K, V)
    