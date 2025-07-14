# LayerNorm from Scratch

A simple reimplementation of `LayerNorm` using PyTorch’s low-level API. This repo reproduces the key behavior of `torch.nn.LayerNorm`, including:

- Per-sample normalization across features (not across batch)
- Learnable scale (`gamma`) and shift (`beta`) parameters
- Numerical stability via `eps`
- Full support for backpropagation with autograd

This project was built to fully understand how Layer Normalization works under the hood (independent of PyTorch's built-in layer).


See `notebook/layernorm_demo.ipynb` for a demo of the implementation.

---

### Tests

All tests pass via `pytest` and the test suite checks for:

- Output equivalence to PyTorch's `nn.LayerNorm`
- Correct shape handling across multiple input shapes
- Mean/variance calculation with `utils.py`

