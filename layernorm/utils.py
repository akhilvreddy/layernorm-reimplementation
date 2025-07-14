import torch

def compute_mean(x, dim=-1, keepdim=True):
    """
    Computes the mean along the specified dimension - I took advantage of PyTorch's in built functionality here.
    
    Args:
        x (Tensor): input tensor
        dim (int): dimension to reduce
        keepdim (bool): whether to keep the reduced dimension

    Returns:
        Tensor: mean values
    """
    return x.mean(dim=dim, keepdim=keepdim)

def compute_variance(x, dim=-1, keepdim=True):
    """
    Computes the variance (unbiased=False) along the specified dimension.
    
    Args:
        x (Tensor): input tensor
        dim (int): dimension to reduce
        keepdim (bool): whether to keep the reduced dimension

    Returns:
        Tensor: variance values
    """
    mean = compute_mean(x, dim=dim, keepdim=True)
    return ((x - mean) ** 2).mean(dim=dim, keepdim=keepdim)