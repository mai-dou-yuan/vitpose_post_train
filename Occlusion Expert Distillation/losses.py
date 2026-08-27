from typing import Optional

import torch
import torch.nn.functional as F


VISIBLE_LABEL = 0


def build_self_occ_mask(batch) -> torch.Tensor:
    visibility_label = batch["visibility_label"]
    in_view_mask = batch["in_view_mask"].bool()
    occ_mask = (visibility_label != VISIBLE_LABEL) & in_view_mask
    assert occ_mask.ndim == 2, f"Expected [B, 21] occ mask, got {occ_mask.shape}"
    return occ_mask


def build_visible_mask(batch) -> torch.Tensor:
    visibility_label = batch["visibility_label"]
    in_view_mask = batch["in_view_mask"].bool()
    visible_mask = (visibility_label == VISIBLE_LABEL) & in_view_mask
    assert visible_mask.ndim == 2, f"Expected [B, 21] visible mask, got {visible_mask.shape}"
    return visible_mask


def compute_gnll_loss_masked(
    pred_mu: torch.Tensor,
    pred_logvar: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    current_epoch: int = 0,
    warmup_epochs: int = 0,
    beta: float = 10.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    assert pred_mu.shape == gt.shape, f"pred_mu {pred_mu.shape} vs gt {gt.shape}"
    assert pred_logvar.shape == gt.shape, f"pred_logvar {pred_logvar.shape} vs gt {gt.shape}"

    robust_dist = F.smooth_l1_loss(pred_mu, gt, reduction="none", beta=beta)

    if current_epoch < warmup_epochs:
        loss = robust_dist
    else:
        pred_logvar = torch.clamp(pred_logvar, min=-5.0, max=5.0)
        precision = torch.exp(-pred_logvar)
        loss = precision * robust_dist + 0.5 * pred_logvar

    if mask is None:
        return loss.mean()

    mask_f = mask.float().unsqueeze(-1)
    valid_count = mask_f.sum()
    if valid_count < 1:
        return pred_mu.sum() * 0.0
    return (loss * mask_f).sum() / (valid_count * pred_mu.shape[-1] + eps)


def compute_expert_gnll_weighted(
    pred_mu: torch.Tensor,
    pred_logvar: torch.Tensor,
    gt: torch.Tensor,
    occ_mask: torch.Tensor,
    current_epoch: int = 0,
    warmup_epochs: int = 0,
    w_occ: float = 1.0,
    w_vis: float = 0.1,
    beta: float = 10.0,
    eps: float = 1e-6,
    vis_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    components = compute_expert_gnll_components(
        pred_mu=pred_mu,
        pred_logvar=pred_logvar,
        gt=gt,
        occ_mask=occ_mask,
        current_epoch=current_epoch,
        warmup_epochs=warmup_epochs,
        w_occ=w_occ,
        w_vis=w_vis,
        beta=beta,
        eps=eps,
        vis_mask=vis_mask,
    )
    return components["total"]


def compute_expert_gnll_components(
    pred_mu: torch.Tensor,
    pred_logvar: torch.Tensor,
    gt: torch.Tensor,
    occ_mask: torch.Tensor,
    current_epoch: int = 0,
    warmup_epochs: int = 0,
    w_occ: float = 1.0,
    w_vis: float = 0.1,
    beta: float = 10.0,
    eps: float = 1e-6,
    vis_mask: Optional[torch.Tensor] = None,
):
    assert pred_mu.shape == gt.shape, f"pred_mu {pred_mu.shape} vs gt {gt.shape}"
    assert pred_logvar.shape == gt.shape, f"pred_logvar {pred_logvar.shape} vs gt {gt.shape}"

    robust_dist = F.smooth_l1_loss(pred_mu, gt, reduction="none", beta=beta)
    if current_epoch < warmup_epochs:
        loss_matrix = robust_dist
    else:
        pred_logvar = torch.clamp(pred_logvar, min=-5.0, max=5.0)
        precision = torch.exp(-pred_logvar)
        loss_matrix = precision * robust_dist + 0.5 * pred_logvar

    occ_mask_f = occ_mask.float().unsqueeze(-1)
    if vis_mask is None:
        vis_mask_f = (~occ_mask).float().unsqueeze(-1)
    else:
        vis_mask_f = vis_mask.float().unsqueeze(-1)

    occ_count = occ_mask_f.sum()
    vis_count = vis_mask_f.sum()
    zero = pred_mu.sum() * 0.0
    loss = zero
    loss_occ = zero
    loss_vis = zero

    if occ_count >= 1:
        loss_occ = (loss_matrix * occ_mask_f).sum() / (occ_count * pred_mu.shape[-1] + eps)
        loss = loss + w_occ * loss_occ

    if vis_count >= 1:
        loss_vis = (loss_matrix * vis_mask_f).sum() / (vis_count * pred_mu.shape[-1] + eps)
        loss = loss + w_vis * loss_vis

    return {
        "occ_raw": loss_occ,
        "vis_raw": loss_vis,
        "occ_weighted": w_occ * loss_occ,
        "vis_weighted": w_vis * loss_vis,
        "total": loss,
    }


def diagonal_gaussian_kl_occ_stable(
    mu_e: torch.Tensor,
    logvar_e: torch.Tensor,
    mu_s: torch.Tensor,
    logvar_s: torch.Tensor,
    occ_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    mu_e = mu_e.detach()
    logvar_e = logvar_e.detach()

    logvar_e = torch.clamp(logvar_e, min=-5.0, max=5.0)
    logvar_s_safe = torch.clamp(logvar_s, min=-5.0, max=5.0)

    term1 = logvar_s_safe - logvar_e
    term2 = torch.exp(torch.clamp(logvar_e - logvar_s_safe, min=-20.0, max=20.0))
    term3 = (mu_e - mu_s).pow(2) * torch.exp(torch.clamp(-logvar_s_safe, min=-20.0, max=20.0))

    kl_per_dim = 0.5 * (term1 + term2 + term3 - 1.0)
    kl_per_joint = kl_per_dim.sum(dim=-1)

    occ_mask_f = occ_mask.float()
    valid_count = occ_mask_f.sum()
    if valid_count < 1:
        return mu_s.sum() * 0.0
    return (kl_per_joint * occ_mask_f).sum() / (valid_count + eps)


def token_distill_occ_safe(
    token_s: torch.Tensor,
    token_e: torch.Tensor,
    occ_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    token_e = token_e.detach()
    token_s = F.layer_norm(token_s, token_s.shape[-1:])
    token_e = F.layer_norm(token_e, token_e.shape[-1:])

    loss_per_joint = (token_s - token_e).pow(2).mean(dim=-1)
    occ_mask_f = occ_mask.float()
    valid_count = occ_mask_f.sum()
    if valid_count < 1:
        return token_s.sum() * 0.0
    return (loss_per_joint * occ_mask_f).sum() / (valid_count + eps)
