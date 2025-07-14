import torch
import torch.nn as nn
from layernorm.utils import compute_mean, compute_variance

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        """
        Sets up layer normalization parameters, just like the original implementation.
        
        Args:
            normalized_shape (int or tuple): input shape from an expected input of size (…, normalized_shape)
            eps (float): a value added to the denominator for numerical stability
        """
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

        # learnable affine parameters (always used in layernorm)
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        """
        Applies layer normalization on the input tensor

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: normalized + scaled + shifted output
        """
        mean = compute_mean(x, dim=-1, keepdim=True)
        var = compute_variance(x, dim=-1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta