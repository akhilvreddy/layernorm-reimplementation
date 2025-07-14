import torch
import torch.nn as nn
from layernorm.layers import LayerNorm

# comparing my implementation with pytorch's implementation
def test_layernorm_equivalence():
    shape = (8, 16)
    x = torch.randn(shape)

    my_ln = LayerNorm(16)
    torch_ln = nn.LayerNorm(16)

    torch_ln.weight.data = my_ln.gamma.data.clone()
    torch_ln.bias.data = my_ln.beta.data.clone()

    out1 = my_ln(x)
    out2 = torch_ln(x)

    assert torch.allclose(out1, out2, atol=1e-5), "Outputs do not match"

# have to make sure our shapes are not collapsing
def test_layernorm_shapes():
    x = torch.randn(4, 10, 32)
    ln = LayerNorm(32)
    out = ln(x)
    assert out.shape == x.shape, "Output shape mismatch"

# similar test as above but for a bunch of different shapes, just to make sure
def test_layernorm_no_crash_on_different_shapes():
    shapes = [(2, 8), (4, 16), (1, 10, 64)]
    for shape in shapes:
        x = torch.randn(*shape)
        ln = LayerNorm(shape[-1])
        out = ln(x)
        assert out.shape == x.shape