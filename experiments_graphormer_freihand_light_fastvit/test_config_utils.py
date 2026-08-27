import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import Lightning before config_utils loads the repository-local `datasets`
# package; transformers probes that package name during Lightning import.
from experiments_graphormer_freihand_light_fastvit import lightning_module  # noqa: F401
from experiments_graphormer_freihand_light_fastvit.config_utils import (
    build_dataloader,
)


def make_dataset():
    return TensorDataset(torch.arange(4))


def test_prefetch_factor_accepts_explicit_value_or_false():
    explicit_loader = build_dataloader(
        make_dataset(), 2, 1, False, prefetch_factor=4
    )
    default_loader = build_dataloader(
        make_dataset(), 2, 1, False, prefetch_factor=False
    )
    pytorch_default_loader = DataLoader(
        make_dataset(), batch_size=2, num_workers=1, persistent_workers=True
    )

    assert explicit_loader.prefetch_factor == 4
    assert default_loader.prefetch_factor == pytorch_default_loader.prefetch_factor


def test_prefetch_factor_is_ignored_without_workers():
    loader = build_dataloader(make_dataset(), 2, 0, False, prefetch_factor=4)
    assert loader.num_workers == 0
    assert loader.prefetch_factor is None


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "4"])
def test_prefetch_factor_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="false or a positive integer"):
        build_dataloader(make_dataset(), 2, 1, False, prefetch_factor=value)
