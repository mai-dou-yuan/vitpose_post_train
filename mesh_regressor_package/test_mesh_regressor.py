import torch
import pytest

from mesh_regressor_package import MeshRegressor


def test_forward_backward_and_shape():
    model = MeshRegressor()
    joint_tokens = torch.randn(2, 21, 256, requires_grad=True)
    vertices = model(joint_tokens)
    assert vertices.shape == (2, 778, 3)
    vertices.sum().backward()
    assert joint_tokens.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_checkpoint_loading():
    torch.manual_seed(7)
    source_model = MeshRegressor().eval()
    checkpoint = {
        "state_dict": {
            f"mesh_head.{key}": value
            for key, value in source_model.state_dict().items()
        }
    }
    loaded_model = MeshRegressor().eval()
    loaded_model.load_from_checkpoint(checkpoint)
    joint_tokens = torch.randn(2, 21, 256)
    with torch.no_grad():
        expected = source_model(joint_tokens)
        actual = loaded_model(joint_tokens)
    assert torch.allclose(actual, expected)


def test_cross_stage_attention_uses_expected_history_shapes():
    model = MeshRegressor().eval()
    projection_inputs = {}
    hooks = []
    for name in ("proj21", "proj84", "proj336"):
        layer = getattr(model.cross_stage_attn, name)
        hooks.append(
            layer.register_forward_pre_hook(
                lambda _module, inputs, projection=name: projection_inputs.__setitem__(
                    projection, tuple(inputs[0].shape)
                )
            )
        )

    with torch.no_grad():
        model(torch.randn(2, 21, 256))
    for hook in hooks:
        hook.remove()

    assert projection_inputs == {
        "proj21": (2, 21, 256),
        "proj84": (2, 84, 256),
        "proj336": (2, 336, 128),
    }


def test_legacy_checkpoint_keeps_new_attention_initialization():
    torch.manual_seed(11)
    source_model = MeshRegressor()
    legacy_state = {
        key: value.detach().clone()
        for key, value in source_model.state_dict().items()
        if not key.startswith("cross_stage_attn.")
    }

    torch.manual_seed(19)
    loaded_model = MeshRegressor()
    initial_attention_state = {
        key: value.detach().clone()
        for key, value in loaded_model.cross_stage_attn.state_dict().items()
    }
    loaded_model.load_from_checkpoint(
        {"state_dict": {f"mesh_head.{key}": value for key, value in legacy_state.items()}}
    )

    for key, value in legacy_state.items():
        assert torch.equal(loaded_model.state_dict()[key], value)
    for key, value in initial_attention_state.items():
        assert torch.equal(loaded_model.cross_stage_attn.state_dict()[key], value)


def test_legacy_loading_still_rejects_other_missing_or_mismatched_weights():
    model = MeshRegressor()
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    del state["proj_1.weight"]
    with pytest.raises(ValueError, match="proj_1.weight"):
        model.load_from_checkpoint(
            {"state_dict": {f"mesh_head.{key}": value for key, value in state.items()}}
        )

    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    state["cross_stage_attn.proj21.weight"] = torch.empty(1, 1)
    with pytest.raises(ValueError, match="cross_stage_attn.proj21.weight"):
        model.load_from_checkpoint(
            {"state_dict": {f"mesh_head.{key}": value for key, value in state.items()}}
        )
