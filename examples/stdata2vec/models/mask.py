import numpy as np
import  torch
from collections import namedtuple
from typing import Optional, Callable
from fairseq.data.data_utils import compute_mask_indices
from fairseq.utils import index_put

MaskSeed = namedtuple("MaskSeed", ["seed", "update", "ids"])
MaskInfo = namedtuple("MaskInfo", ["x_unmasked", "mask", "ids_restore", "ids_keep"])

def make_maskinfo(self, x, mask, keep_masked_pct, shape=None):
    if shape is None:
        B, T, D = x.shape
    else:
        B, T, D = shape
    mask = mask.to(torch.uint8)
    #if torch.__version__>="2.0":
    #    mask = mask.to(torch.bool)
    ids_shuffle = mask.argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1).unsqueeze(-1).expand(-1, -1, D)

    len_keep = T - mask[0].sum()
    if keep_masked_pct > 0:
        len_keep += round((T - int(len_keep)) * keep_masked_pct)

    ids_keep = ids_shuffle[:, :len_keep]

    if shape is not None:
        x_unmasked = None
    else:
        ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        x_unmasked = torch.gather(x, dim=1, index=ids_keep)

    mask_info = MaskInfo(
        x_unmasked=x_unmasked,
        mask=mask,
        ids_restore=ids_restore,
        ids_keep=ids_keep,
    )
    return mask_info

def apply_mask(self, x, mask_info,encoder_zero_mask, mask_noise_std,mask_channel_prob):
    
    B, T, C = x.shape

    if mask_info is not None:
        mask = mask_info.mask
        if encoder_zero_mask:
            x = x * (1 - mask.type_as(x).unsqueeze(-1))
        else:
            num_masks = mask.sum().item()
            masks = x.new_empty(num_masks, x.size(-1)).normal_(
                0, mask_noise_std
            )
            x = index_put(x, mask, masks)
    if mask_channel_prob > 0:
        mask_channel = compute_mask_indices(
            (B, C),
            None,
            cfg.mask_channel_prob,
            cfg.mask_channel_length,
        )
        mask_channel = (
            torch.from_numpy(mask_channel)
            .to(x.device)
            .unsqueeze(1)
            .expand(-1, T, -1)
        )
        x = index_put(x, mask_channel, 0)
    return x

def compute_mask(
    self,
    x,
    padding_mask,
    mask_seed: Optional[MaskSeed],
    apply,
    precomputed_mask,
    mask_prob,
    mask_prob_min,
    inverse_mask,
    mask_dropout,
    add_masks,
    mask_length,
    keep_masked_pct,
    encoder_zero_mask, 
    mask_noise_std,
    mask_channel_prob,
    ):
    if precomputed_mask is not None:
        mask = precomputed_mask
        mask_info = make_maskinfo(x, mask)
    else:
        B, T, C = x.shape
        if (
            cfg.mask_prob_min is not None
            and cfg.mask_prob_min >= 0
            and cfg.mask_prob_min < mask_prob
        ):
            mask_prob = np.random.uniform(mask_prob_min, mask_prob)
        if mask_prob > 0:
            if mask_length == 1:
                mask_info = random_masking(x, mask_prob, mask_seed)
            else:
                if inverse_mask:
                    mask_prob = 1 - mask_prob

                mask = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    mask_prob,
                    cfg.mask_length,
                    min_masks=1,
                    require_same_masks=True,
                    mask_dropout=mask_dropout,
                    add_masks=add_masks,
                    seed=mask_seed.seed if mask_seed is not None else None,
                    epoch=mask_seed.update if mask_seed is not None else None,
                    indices=mask_seed.ids if mask_seed is not None else None,
                )

                mask = torch.from_numpy(mask).to(device=x.device)
                if inverse_mask:
                    mask = 1 - mask
                mask_info = make_maskinfo(x, mask, keep_masked_pct)
        else:
            mask_info = None

    if apply:
        x = apply_mask(x, mask_info, encoder_zero_mask, mask_noise_std, mask_channel_prob)
    return x, mask_info
      
