import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.checkpoint import checkpoint


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pl_system_v6_graphormer import PoseLightningModule


MODEL_INIT_KEYS = (
    "lr",
    "num_joints",
    "local_model_dir",
    "feature_dim",
    "layers",
    "upsample_dim",
    "num_refine_layers",
    "use_gradient_checkpointing",
)


class GraphormerOcclusionWrapper(PoseLightningModule):
    def __init__(self, *args, gnll_warmup_epochs: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gnll_warmup_epochs = gnll_warmup_epochs
        self.hparams["gnll_warmup_epochs"] = gnll_warmup_epochs

    def set_gnll_warmup(self, warmup_epochs: int) -> None:
        self.gnll_warmup_epochs = int(warmup_epochs)
        self.hparams["gnll_warmup_epochs"] = int(warmup_epochs)

    def forward(self, x: torch.Tensor, hand_back: torch.Tensor) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        batch_size = x.shape[0]

        features_dict = self.vitmodel(x)
        extracted_features = []
        for layer_id in self.layers:
            feat = features_dict[layer_id]
            patch_tokens = feat[:, 1:, :].transpose(1, 2)
            _, channels, num_patches = patch_tokens.shape
            height = width = int(math.sqrt(num_patches))
            extracted_features.append(patch_tokens.view(batch_size, channels, height, width))

        upsampled_features = [head(feat) for head, feat in zip(self.upsample_heads, extracted_features)]
        global_feature_map = self.fuse_block(upsampled_features)
        pos_embed_map = self.pos_embed_layer(global_feature_map)

        curr_tokens = self.joint_tokens.expand(batch_size, -1, -1)
        query_pos = self.joint_token_pos.expand(batch_size, -1, -1)

        all_stage_preds = []
        all_stage_logvars = []
        all_stage_tokens = []

        for i in range(self.num_refine_layers):
            if self.training and self.use_gradient_checkpointing:
                curr_tokens = checkpoint(self.layers_sa[i], curr_tokens, query_pos, use_reentrant=False)
                curr_tokens = checkpoint(
                    self.layers_ca[i],
                    curr_tokens,
                    global_feature_map,
                    query_pos,
                    pos_embed_map,
                    use_reentrant=False,
                )
            else:
                curr_tokens = self.layers_sa[i](x=curr_tokens, pos=query_pos)
                curr_tokens = self.layers_ca[i](
                    tgt=curr_tokens,
                    memory=global_feature_map,
                    query_pos=query_pos,
                    memory_pos=pos_embed_map,
                )

            all_stage_tokens.append(curr_tokens)
            raw_pred = self.pose_3d_head_PR(curr_tokens)
            all_stage_preds.append(raw_pred[..., :3])
            all_stage_logvars.append(raw_pred[..., 3:])

        results["pose3d"] = all_stage_preds[-1]
        results["pose3d_logvar"] = all_stage_logvars[-1]
        results["joint_token"] = all_stage_tokens[-1]
        results["all_stage_pose3d"] = all_stage_preds
        results["all_stage_logvars"] = all_stage_logvars
        results["all_stage_tokens"] = all_stage_tokens
        return results


def extract_init_kwargs_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    hparams = checkpoint_data.get("hyper_parameters", {})
    init_kwargs = {key: hparams[key] for key in MODEL_INIT_KEYS if key in hparams}
    if "lr" not in init_kwargs:
        init_kwargs["lr"] = 1e-4
    return init_kwargs


def load_occlusion_model_from_checkpoint(
    checkpoint_path: str,
    *,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = False,
    override_kwargs: Optional[Dict[str, Any]] = None,
    gnll_warmup_epochs: int = 0,
) -> GraphormerOcclusionWrapper:
    init_kwargs = extract_init_kwargs_from_checkpoint(checkpoint_path)
    if override_kwargs:
        init_kwargs.update(override_kwargs)
    model = GraphormerOcclusionWrapper(**init_kwargs, gnll_warmup_epochs=gnll_warmup_epochs)

    checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint_data["state_dict"]
    if state_dict and not any(key.startswith("vitmodel.") for key in state_dict):
        for prefix in ("model.", "student.", "expert."):
            prefixed = f"{prefix}vitmodel."
            if any(key.startswith(prefixed) for key in state_dict):
                state_dict = {
                    key[len(prefix):]: value
                    for key, value in state_dict.items()
                    if key.startswith(prefix)
                }
                break
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"[OcclusionWrapper] Missing keys while loading {checkpoint_path}: {missing}")
    if unexpected:
        print(f"[OcclusionWrapper] Unexpected keys while loading {checkpoint_path}: {unexpected}")
    model.set_gnll_warmup(gnll_warmup_epochs)
    return model
