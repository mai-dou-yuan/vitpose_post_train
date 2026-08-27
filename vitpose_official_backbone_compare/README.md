# Official ViTPose++-B backbone comparison

This directory compares the official ViTPose `ViTMoE` implementation with the
dependency-light extraction in `../pretrain_vitpose_pkl_and_call/`.  Neither
path constructs or calls a keypoint/heatmap head.

The official source checkout is pinned by its local Git commit.  The tested
commit is `c050ed29112da7704797cc1a65af0234b525010d`.  The tested
environment is named `vitpose_official` and follows the official README stack:
Python 3.8, PyTorch 1.9.0, torchvision 0.10.0, CUDA runtime 11.1,
mmcv-full 1.3.9, MMPose/ViTPose 0.24.0, timm 0.4.9, and einops.

## Reproduce the environment

```bash
export http_proxy=http://127.0.0.1:10808
export https_proxy=http://127.0.0.1:10808
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy

conda create -n vitpose_official python=3.8 -y
conda install -n vitpose_official pytorch=1.9.0 torchvision=0.10.0 \
  cudatoolkit=11.1 -c pytorch -c conda-forge -y
conda run -n vitpose_official python -m pip install \
  mmcv-full==1.3.9 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
conda run -n vitpose_official python -m pip install -v \
  -e ./vitpose_official_backbone_compare/official_vitpose \
  timm==0.4.9 einops numpy==1.23.5
```

This machine globally exposes CUDA 11.7 through `LD_LIBRARY_PATH`.  That path
must not override the environment's CUDA 11.1 libraries.  The tested isolated
environment therefore has:

```bash
conda env config vars set -n vitpose_official \
  LD_LIBRARY_PATH=/home/duanmu/anaconda3/envs/vitpose_official/lib
```

## Run

Load only the official backbone:

```bash
conda run -n vitpose_official python \
  vitpose_official_backbone_compare/load_official_vitpose_plus_backbone.py \
  --device auto --dataset-source 5 --freeze
```

Export a reference with the existing `vit` environment, then compare in the
official environment:

```bash
conda run -n vit python \
  vitpose_official_backbone_compare/export_existing_reference.py

conda run -n vitpose_official python \
  vitpose_official_backbone_compare/compare_backbones.py
```

The comparison checks all 293 backbone tensors, all six expert routes, every
transformer block for expert 5, and a cross-environment feature artifact.  The
canonical output is `[B, 768, 16, 12]`; Graphormer tokens are obtained with
`feature_map.flatten(2).transpose(1, 2)`, giving `[B, 192, 768]`.

"Same" is reported at two levels: parameter tensors must be bitwise identical;
floating-point features must satisfy `atol=2e-5, rtol=1e-5` and relative L2
error at most `2e-6`.  Exact feature equality is also printed, but is not
expected across different PyTorch/BLAS implementations or different expert
evaluation orders.
