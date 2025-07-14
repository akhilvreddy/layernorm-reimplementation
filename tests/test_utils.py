import torch
from layernorm.utils import compute_mean, compute_variance

# I just wrote 2 simple test to check if mean and variance computation is working

def test_compute_mean():
    x = torch.randn(4, 10)
    expected = x.mean(dim=-1, keepdim=True)
    actual = compute_mean(x, dim=-1, keepdim=True)
    assert torch.allclose(actual, expected, atol=1e-6)

def test_compute_variance():
    x = torch.randn(4, 10)
    expected = x.var(dim=-1, unbiased=False, keepdim=True)
    actual = compute_variance(x, dim=-1, keepdim=True)
    assert torch.allclose(actual, expected, atol=1e-6)