"""Backbone-only config matched to the official multi-task ViTPose++-B weight.

The source checkpoint was released for six training datasets in this order:
COCO, AIC, MPII, AP-10K, APT-36K and COCO-WholeBody.  ViTPose++ routes the
last ``part_features`` channels through the expert selected by dataset_source.

The image_size convention follows MMPose: data_cfg stores [width, height],
whereas the backbone img_size stores (height, width).
"""

model = dict(
    backbone=dict(
        type="ViTMoE",
        img_size=(256, 192),
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        ratio=1,
        last_norm=True,
        norm_eps=1e-6,
        num_expert=6,
        part_features=192,
        use_checkpoint=False,
    )
)

data_cfg = dict(image_size=[192, 256])

# Official pipeline: LoadImageFromFile(channel_order='rgb') -> TopDownAffine
# (UDP) -> ToTensor (uint8 / 255) -> NormalizeTensor.
preprocess_cfg = dict(
    channel_order="RGB",
    input_range="0_1",
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    interpolation="bilinear",
    expects_topdown_crop=True,
    use_udp=True,
)

dataset_experts = {
    "coco": 0,
    "aic": 1,
    "mpii": 2,
    "ap10k": 3,
    "apt36k": 4,
    "wholebody": 5,
}

# WholeBody is the closest pretraining task to a cropped hand input.  This is
# a routing choice, not a learned hand-only expert; callers can override it per
# sample by passing dataset_source to forward().
default_dataset_source = dataset_experts["wholebody"]

