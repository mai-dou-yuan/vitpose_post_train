# FreiHAND Graphormer with trainable ViTPose++-B

This directory provides a standalone training/testing entry for FreiHAND while
using the backbone-only ViTPose++-B implementation from
`pretrain_vitpose_pkl_and_call`. The local multi-task checkpoint is loaded
without constructing its original neck or keypoint head; training does not
download model weights. A small experiment-local Initial 2D Coordinate Head is
applied after the 768-to-256 feature projection.

The existing FreiHAND dataset still supplies square `[0, 1]` RGB crops. The
backbone adapter resizes them to the checkpoint's canonical `256x192` input,
applies its configured ImageNet normalization, and routes every sample to the
WholeBody expert (`dataset_source=5`). The resulting `[B, 768, 16, 12]` map is
projected to the unchanged Graphormer memory dimension.

## Initial 2D references

The projected `[B,256,16,12]` feature map passes through a 16-channel pointwise
and depthwise-convolutional bottleneck, then adaptive pooling retains a compact
`4x3` spatial grid. A two-layer MLP directly regresses 42 values, reshaped to
21 `(x,y)` pairs and bounded by sigmoid in crop-normalized `[0,1]` coordinates.
No per-joint heatmaps are constructed. These coordinates are explicitly
converted to pixels of the dataset crop (normally `224x224`) before Stage 0
Local CA; the `16x12` feature indices and ViTPose's internal `256x192` pixels
are never used as crop pixels.

The refinement flow is:

- Stage 0: Initial 2D crop pixels -> Local CA -> Stage 0 3D prediction.
- Stage 1: Stage 0 3D + camera-coordinate wrist -> `cam_k` projection -> Local CA.
- Stage 2: unchanged Full CA; no local reference coordinates.

After every refinement stage, the latest joint tokens pass through the shared
SimpleHand mesh regressor to produce 778 MANO vertices. Its final 778 vertex
tokens use cross-attention to retrieve the 21/84/336-token intermediate mesh
features before vertex prediction. A registered MANO joint
regressor plus fingertip lookup converts those vertices to 21 joints. Predictions
and FreiHAND mesh/joint targets are wrist-relative and retain FreiHAND's meter
units and joint order. The old token-coordinate 3D head remains in the forward
path only for later local-attention references; it has no direct coordinate loss.

Only the final stage is supervised. All three terms use L1 loss:
`1.0 * loss_2d + 10.0 * loss_3d + 10.0 * loss_vert`. The 2D term directly
supervises the normalized Initial 2D Head and therefore guides Stage 0 sampling;
the 3D terms supervise mesh-derived joints and mesh vertices. Training,
validation and test log every component and the weighted total separately.

Train and test commands create two CSV logs with the same `version_N` id:

- `outputs/lightning_logs/version_N/metrics.csv` keeps all losses, metrics and
  diagnostics, including final MPVPE/PA-MPVPE and per-stage MPJPE/PA-MPJPE.
- `outputs/lightning_logs_summary/version_N/metrics.csv` is the compact view.
  It contains only final MPJPE/PA-MPJPE, each refinement stage's
  MPJPE/PA-MPJPE, and final MPVPE/PA-MPVPE (plus `epoch`/`step` indices).

Training metrics in the compact CSV are epoch averages; validation and test
metrics are dataset averages. MPVPE uses the same wrist-relative meter-space
vertices as mesh supervision, while PA-MPVPE applies one similarity Procrustes
alignment per sample before measuring vertex error.

Older checkpoints can initialize the new architecture through the non-strict,
weights-only loader. It prints every missing/unexpected key; for the expected
legacy case every missing tensor belongs to the new 2D/mesh modules (including
the mesh cross-stage attention) and there are no unexpected tensors. Training
then starts a fresh optimizer because the legacy optimizer parameter groups do
not contain the new modules. Exact current
checkpoints still resume full trainer state. Testing rejects legacy checkpoints
because a randomly initialized coordinate head would make Stage 0 results
invalid.

## Training regularization

The default FreiHAND config enables train-only augmentation:

- center/scale crop jitter and camera-Z rotation with synchronized 2D joints,
  3D joints, mesh vertices and camera-intrinsic updates;
- moderate color jitter and one randomly selected blur/noise/JPEG degradation;
- one light, hand-centered occlusion rectangle;
- grouped background sampling: one of FreiHAND's four image variants is drawn
  for each of the 32,560 distinct poses on every epoch.

Validation and test images keep the original deterministic preprocessing.
`training.prefetch_factor` accepts a positive integer or `false`; `false`
leaves the DataLoader at PyTorch's default, and the setting is ignored when
`training.num_workers` is `0`. The
learning-rate warm-up length is configured by `training.lr_warmup_epochs`
(`0` disables it). Independently, ViTPose is frozen for
`training.backbone_freeze_epochs` and then fine-tuned with a lower learning
rate than the pose decoder. Validation/test
logs also include scale-aligned MPJPE, axis MAE, bone length error and
predicted/ground-truth scale ratio for diagnosing metric-scale errors.


## Commands

Train:

```bash
conda run -n vit python -m experiments_graphormer_freihand_light_fastvit.train --config experiments_graphormer_freihand_light_fastvit/configs/freihand_graphormer.yaml
```

Test:

```bash
conda run -n vit python -m experiments_graphormer_freihand_light_fastvit.test --config experiments_graphormer_freihand_light_fastvit/configs/freihand_graphormer.yaml --checkpoint /path/to/model.ckpt
```

Offline integration check (strict weight loading, forward, backward and
optimizer membership):

```bash
conda run -n vit python -m experiments_graphormer_freihand_light_fastvit.verify_vitpose_integration
```

Checkpoints produced by the former FastViT version are not architecture
compatible with this ViTPose++-B version; start a new run or resume only from a
checkpoint produced by this code.
