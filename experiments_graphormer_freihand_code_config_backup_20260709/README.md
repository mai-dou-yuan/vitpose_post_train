# FreiHAND Graphormer Experiment

This directory provides a standalone training/testing entry for FreiHAND while
reusing the repository's existing `FreiHANDDataset` and `PoseLightningModule`
implementations.

## Commands

Train:

```bash
python experiments_graphormer_freihand/train.py --config experiments_graphormer_freihand/configs/freihand_graphormer.yaml
```

Test:

```bash
python experiments_graphormer_freihand/test.py --config experiments_graphormer_freihand/configs/freihand_graphormer.yaml --checkpoint /path/to/model.ckpt
```
