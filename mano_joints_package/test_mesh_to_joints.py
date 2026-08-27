import pytest
import torch

from mano_joints_package import MeshToJoints, mesh_to_joints


def test_function_and_module_forward_backward():
    torch.manual_seed(7)
    vertices = torch.randn(2, 778, 3, requires_grad=True)

    function_output = mesh_to_joints(vertices)
    module_output = MeshToJoints()(vertices)

    assert function_output.shape == (2, 21, 3)
    assert torch.allclose(function_output, module_output, atol=0.0, rtol=0.0)

    module_output.sum().backward()
    assert vertices.grad is not None
    assert torch.isfinite(vertices.grad).all()


@pytest.mark.parametrize(
    "bad_shape",
    [(778, 3), (2, 777, 3), (2, 778, 2), (2, 21, 3)],
)
def test_invalid_shape(bad_shape):
    with pytest.raises(ValueError, match=r"\[B, 778, 3\]"):
        mesh_to_joints(torch.randn(*bad_shape))
