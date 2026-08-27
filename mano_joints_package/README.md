# SimpleHand MANO mesh-to-joints package

该目录独立封装了 SimpleHand 的以下转换：

```text
778 个预测顶点 -> 16 个 MANO 关节 -> 补充 5 个指尖
-> 按 SimpleHand 顺序重排 -> 21 个三维关节
```

目录中的实现不引用原项目的 `models` 包。运行时依赖 PyTorch、NumPy，
并使用随目录打包的 `data/MANO_RIGHT_C.pkl`。

## 函数式调用

```python
import torch
from mano_joints_package import mesh_to_joints

vertices = torch.randn(2, 778, 3)
joints = mesh_to_joints(vertices)
assert joints.shape == (2, 21, 3)
```

## 作为 PyTorch 模块调用

```python
from mano_joints_package import MeshToJoints

converter = MeshToJoints().to(vertices.device)
joints = converter(vertices)
```

`MeshToJoints` 会把 `[16, 778]` 的 `J_regressor` 注册为 buffer，因此它会随
模块移动设备并包含在 `state_dict()` 中。该转换没有可训练参数，但保留梯度，
可以直接接在网格预测网络后参与反向传播。

## 测试

在包含本目录的上一级目录运行：

```bash
conda run -n vit python -m pytest -q mano_joints_package/test_mesh_to_joints.py
```
