# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import random
import math

def compute_valid_patches_from_crops(
    crops: torch.Tensor,
    n_tokens: int,
    patch_size: int,
    min_nonzero_frac: float = 0.02,
    eps: float = 0.0,
) -> torch.Tensor:
    """
    Compute a boolean mask of 'valid' patches based on occupancy.

    Args:
        crops: (B, C, H, W) tensor of global crops
        n_tokens: number of patch tokens per image (N_patches)
        min_nonzero_frac: minimum fraction of non-zero pixels in a patch
                          to consider it valid.
        eps: threshold for treating a pixel as non-zero.

    Returns:
        valid: (B, N_patches) bool tensor: True = patch has enough non-zero pixels.
    """
    B, C, H, W = crops.shape
    crops2 = crops.detach()

    # Binary map of "non-zero pixels" (any channel exceeds eps in abs value)
    # percentage based : 2% worked for ViT backbone
    nonzero = (crops2.abs() > eps).any(dim=1, keepdim=True).float()  # (B, 1, H, W)

    # Average occupancy per patch using avg_pool2d
    occ = F.avg_pool2d(
        nonzero,
        kernel_size=patch_size,
        stride=patch_size,
    )  # (B, 1, H/ps, W/ps)

    # Flatten to (B, N_patches)
    occ = occ.view(B, -1)

    # Threshold
    # percentage based : 2% worked for ViT backbone
    valid = occ >= min_nonzero_frac  # (B, N_patches) bool
    return valid

def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, patch_size, dtype, n_tokens=None, mask_generator=None):
    # dtype = torch.half  # TODO: Remove

    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    collated_local_crops = None
    if n_local_crops > 0:
        collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
        collated_local_crops = collated_local_crops.to(dtype)

    B = len(collated_global_crops)
    N = n_tokens
    # --------------------------------------------------------
    # NEW: compute valid patches (near-empty patches are invalid)
    # --------------------------------------------------------
    valid_patches = None
    if (mask_generator is not None) and (n_tokens is not None):
        # This computes which patch tokens correspond to "real signal"
        # based on non-zero pixel fraction in the global crops.
        valid_patches = compute_valid_patches_from_crops(
            collated_global_crops,
            n_tokens=N,
            patch_size=patch_size,
            #  min_nonzero_frac=0.02,  # <- tune this threshold (worked for ViT)
            min_nonzero_frac=0.01,  # <- tune this threshold (warp conv)
            eps=0.0,
        )  # (B, N) bool

    # --------------------------------------------------------
    # Original mask generation
    # --------------------------------------------------------
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)

    # --------------------------------------------------------
    # NEW: zero out masks on invalid (near-empty) patches
    # --------------------------------------------------------
    if valid_patches is not None:
        # make sure we're on same device
        valid_patches = valid_patches.to(collated_masks.device)
        collated_masks = collated_masks & valid_patches  # (B, N) bool

    #      #  # OPTIONAL: ensure each sample has at least one masked patch
    #      #  # (otherwise some rows could be all-False for very sparse images)
    #      #  for b in range(B):
    #      #      if not collated_masks[b].any():
    #      #          # fallback: mask the most "occupied" patch
    #      #          idx = valid_patches[b].float().argmax()
    #      #          collated_masks[b, idx] = True

    # Recompute indices and weights AFTER filtering

    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
