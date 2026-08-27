# Standalone MeshRegressor

This directory can be copied as a whole into another Python project. Its only
runtime dependency is PyTorch.

```python
import torch
from mesh_regressor_package import MeshRegressor

model = MeshRegressor()
vertices = model(torch.randn(2, 21, 256))
assert vertices.shape == (2, 778, 3)
```

The regressor keeps the `21 -> 84 -> 336 -> 778` upsampling path and uses the
final 778 tokens as queries over the three earlier feature scales before the
`64 -> 3` prediction layer.

Copy weights from an instantiated simpleHand `MeshHead`:

```python
model.load_from_mesh_head(hand_net.mesh_head)
```

Load from a saved simpleHand checkpoint:

```python
checkpoint = torch.load("checkpoint.pth", map_location="cpu")
model.load_from_checkpoint(checkpoint, prefix="mesh_head.")
```

Legacy weights without `cross_stage_attn.*` initialize the unchanged backbone
and output head while the new cross-stage attention keeps its initialization.
Other missing or shape-mismatched parameters still raise `ValueError`.

For a distributed checkpoint whose keys start with `module.mesh_head.`, pass
that string as `prefix` instead.

Run the packaged tests from the project root:

```bash
python -m pytest -q mesh_regressor_package/test_mesh_regressor.py
```
