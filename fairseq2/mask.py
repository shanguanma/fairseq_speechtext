#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, cast

import torch
from torch import Tensor
import numpy as np


## (TODO) check two version mask.
## this function is from fairseq
def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
    add_masks: bool = False,
    seed: Optional[int] = None,
    epoch: Optional[int] = None,
    indices: Optional[torch.Tensor] = None,
    idc_select_ver: int = 1,  # 2 to reproduce mask_tokens_dataset
    num_mask_ver: int = 2,  # 2 to reproduce mask_tokens_dataset
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    if num_mask_ver == 1:
        all_num_mask = int(
            # add a random number for probabilistic rounding
            mask_prob * all_sz / float(mask_length)
            + np.random.rand()
        )
        all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if seed is not None and epoch is not None and indices is not None:
            seed_i = int(hash((seed, epoch, indices[i].item())) % 1e6)
        else:
            seed_i = None

        rng = np.random.default_rng(seed_i)

        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            assert sz >= 0, sz
        else:
            sz = all_sz

        if num_mask_ver == 1:
            if padding_mask is not None:
                num_mask = int(
                    # add a random number for probabilistic rounding
                    mask_prob * sz / float(mask_length)
                    + np.random.rand()
                )
                num_mask = max(min_masks, num_mask)
            else:
                num_mask = all_num_mask
        elif num_mask_ver == 2:
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + rng.random()
            )
            num_mask = max(min_masks, num_mask)
        else:
            raise ValueError()

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = rng.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = rng.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = rng.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)
        # print(f"sum(lengths) : {sum(lengths)}")
        # print(f"num_mask: {num_mask}, mask_length: {mask_length}") ## num_mask: int, mask_length: int
        if sum(lengths) == 0:
            if mask_type == "static":
                raise ValueError(f"this should never happens")
            else:
                lengths = [min(mask_length, sz - 1)]

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = rng.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = rng.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            if idc_select_ver == 1:
                min_len = min(lengths)
                if sz - min_len <= num_mask:
                    min_len = sz - num_mask - 1
                mask_idc = rng.choice(sz - min_len, num_mask, replace=False)
            elif idc_select_ver == 2:
                mask_idc = rng.choice(sz, num_mask, replace=False)
            else:
                raise ValueError()

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idc = np.unique(mask_idc[mask_idc < sz])
        if len(mask_idc) >= sz:
            raise ValueError(
                (
                    f"the entire sequence is masked. "
                    f"sz={sz}; mask_idc[mask_idc]; "
                    f"index={indices[i] if indices is not None else None}"
                )
            )
        mask_idcs.append(mask_idc)

    target_len = None
    if require_same_masks:
        if add_masks:
            target_len = max([len(m) for m in mask_idcs])
        else:
            target_len = min([len(m) for m in mask_idcs])

    for i, mask_idc in enumerate(mask_idcs):
        if target_len is not None and len(mask_idc) > target_len:
            mask_idc = rng.choice(mask_idc, target_len, replace=False)

        mask[i, mask_idc] = True

        if target_len is not None and len(mask_idc) < target_len:
            unmasked = np.flatnonzero(~mask[i])
            to_mask = rng.choice(unmasked, target_len - len(mask_idc), replace=False)
            mask[i, to_mask] = True

        if mask_dropout > 0:
            masked = np.flatnonzero(mask[i])
            num_holes = np.rint(len(masked) * mask_dropout).astype(int)
            to_drop = rng.choice(masked, num_holes, replace=False)
            mask[i, to_drop] = False

    return mask

## the below code is copy and modified from fairseq2
def to_padding_mask(seqs: Tensor, seq_lens: Optional[Tensor]) -> Optional[Tensor]:
    """Convert a sequence length array to a float padding mask.

    :param seqs:
        The sequences to mask. *Shape:* :math:`(N,S,*)`, where :math:`N` is the
        batch size, :math:`S` is the sequence length, and :math:`*` is any
        number of sequence-specific dimensions including none.
    :param seq_lens:
        An array where each element represents the length of the sequence at the
        same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is the
        batch size.

    :returns:
        The float padding mask. *Shape:* :math:`(N,S)`, where :math:`N` is the
        batch size and :math:`S` is the sequence length.
    """
    bool_mask = to_bool_padding_mask(seqs, seq_lens)
    if bool_mask is None:
        return None

    return to_float_mask(bool_mask, seqs.dtype if seqs.is_floating_point() else None)

def to_bool_padding_mask(seqs: Tensor, seq_lens: Optional[Tensor]) -> Optional[Tensor]:
    """Convert a sequence length array to a boolean padding mask.

    :param seqs:
        The sequences to mask. *Shape:* :math:`(N,S,*)`, where :math:`N` is the
        batch size, :math:`S` is the sequence length, and :math:`*` is any
        number of sequence-specific dimensions including none.
    :param seq_lens:
        An array where each element represents the length of the sequence at the
        same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is the
        batch size.

    :returns:
        The boolean padding mask. *Shape:* :math:`(N,S)`, where :math:`N` is the
        batch size and :math:`S` is the sequence length.
    """
    if seq_lens is None:
        return None

    batch_size, mask_seq_len = seqs.shape[:2]

    # No need to construct a mask if all sequences have the same length.
    if (seq_lens == mask_seq_len).all():
        return None

    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)

    return indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)


def to_float_mask(mask: Tensor, dtype = None) -> Tensor:
    """Convert a boolean mask to a float mask.

    :param mask:
        The mask tensor. *Shape:* Any.
    :param dtype:
        The floating-point type of the converted mask.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    return torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, -torch.inf)



def apply_padding_mask(seqs: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
    """Apply the specified padding mask to ``seqs``.

    :param seqs:
        The sequences to mask. *Shape:* :math:`(N,S,*)`, where :math:`N` is the
        the batch size, :math:`S` is the sequence length, and :math:`*` is any
        number of sequence-specific dimensions including none.
    :param padding_mask:
        The float padding mask to apply. *Shape:* :math:`(N_{msk},S)`, where
        :math:`N_{msk}` is the mask batch size and :math:`S` is the sequence
        length. :math:`N` can be a multiple of :math:`N_{msk}` in which case the
        mask will be tiled before being applied.

    :returns:
        The input sequences with mask applied. *Shape:* Same as ``seqs``.
    """
    if padding_mask is None:
        return seqs

    bool_mask = padding_mask.isinf()

    seq_batch_size, mask_batch_size = seqs.size(0), padding_mask.size(0)

    if seq_batch_size != mask_batch_size:
        if seq_batch_size % mask_batch_size != 0:
            raise ValueError(
                f"`seqs.size(0)` must be a multiple of `padding_mask.size(0)` ({mask_batch_size}), but is {seq_batch_size} instead."
            )
        bool_mask = bool_mask.repeat(seq_batch_size // mask_batch_size, 1)

    return seqs.masked_fill(bool_mask.unsqueeze(2), 0.0)


def compute_mask(
    shape: Tuple[int, int],
    span_len: int,
    max_mask_prob: float,
    row_lens: Optional[Tensor] = None,
    min_num_spans: int = 0,
    device = None,
) -> Optional[Tensor]:
    """Compute a random mask for the specified shape.

    :param shape:
        The two dimensional shape for which to compute a mask.
        for example: if shape=(batch_size, seq_len), it will return temporal_mask
                     it shape is same as (batch_size, seq_len)
                     if shape=(batch_size, model_dim),it will return  spatial_mask
                     it shape is same as (batch_size, model_dim).
    :param span_len:
        The length of each mask span.
    :param max_mask_prob:
        The maximum probability of masking an element among all elements in a
        row. Note that, due to mask span overlap, the effective probability
        might be smaller. The implementation also guarantees that there is
        always at least one unmasked element in each row.
    :param row_lens:
        The length of each row if ``shape`` is ragged.
    :param min_num_spans:
        The minimum number of mask spans per row.
    :param device:
        The device on which to initialize the mask.

    :returns:
        A boolean mask. *:Shape:* ``shape``.its shape is same as param shape
    """
    num_rows, max_row_len = shape

    if row_lens is None:
        # We only mask rows that are longer than the mask span length.
        if span_len >= max_row_len:
            raise ValueError(
                f"The size of the second dimension of `shape` must be greater than {span_len}, but is {max_row_len} instead."
            )

        row_lens = torch.full(
            (num_rows,), max_row_len, device=device, dtype=torch.int64
        )
    else:
        row_lens = row_lens.view(num_rows)

        # We only mask rows that are longer than the mask span length.
        if (span_len >= row_lens).any():
            raise ValueError(
                f"All lengths in `row_lens` must be greater than {span_len}, but at least one length is smaller. row_lens: {row_lens}"
            )

    indices = _compute_mask_spans(row_lens, span_len, max_mask_prob, min_num_spans)

    if indices is None:
        return row_lens.new_empty((0, 0))

    return _generate_mask(indices, max_row_len).to(device)


def _compute_mask_spans(
    row_lens: Tensor, span_len: int, max_mask_prob: float, min_num_spans: int
) -> Optional[Tensor]:
    """Compute random mask spans for the specified (ragged) shape."""
    device, dtype = row_lens.device, row_lens.dtype

    num_rows = row_lens.size(0)
    if num_rows == 0:
        return None

    # Compute the number of mask spans per row. We should always have at least
    # one unmasked element; this is why we substract 1 from `row_lens`.
    num_spans_per_row = (max_mask_prob / span_len) * (row_lens - 1)

    # Require the same number of mask spans for all rows.
    num_spans = cast(int, num_spans_per_row.type(dtype).min().item())

    if min_num_spans > num_spans:
        raise ValueError(
            f"`min_num_spans` is {min_num_spans}, but with the given `span_len` and `max_mask_prob` only {num_spans} mask span(s) can be generated."
        )

    if num_spans == 0:
        return None

    # The range of possible start indices for mask spans in form [0, max + 1).
    span_start_range = row_lens - span_len + 1

    # (R) -> (R x N)
    span_start_range = span_start_range.repeat_interleave(num_spans)

    # Unlike the fairseq implementation, we do sample with replacement, which is
    # more consistent with the overlap strategy.
    # (R x N)
    rand_scales = torch.rand(num_rows * num_spans, device=device)

    # By random scaling we effectively pick a random start index for each mask
    # span.
    span_offsets = span_start_range * rand_scales

    # The following ops convert the mask span offsets (i.e. start indices) to
    # mask spans (i.e. index ranges).
    # (R x N) -> (R, N)
    span_offsets = span_offsets.type(dtype).view(num_rows, -1)

    # (R, N) -> (R, N x L)
    span_offsets = span_offsets.repeat_interleave(span_len, dim=-1)

    # (L)
    indices = torch.arange(span_len, device=device, dtype=dtype)

    # (L) -> (R, N x L)
    indices = indices.repeat(num_spans).unsqueeze(0).expand(num_rows, -1)

    return span_offsets + indices


def _generate_mask(indices: Tensor, max_row_len: int) -> Tensor:
    """Generate a boolean mask by setting ``indices`` to ``True``."""
    float_mask = torch.zeros((indices.size(0), max_row_len), device=indices.device)

    # Set elements corresponding to masked indices to 1.
    float_mask.scatter_(1, indices, 1.0)

    # Since mask spans may overlap, rows might have varying number of masked
    # elements; therefore, we have to randomly unmask some of the elements to
    # ensure that all rows have the same amount of masking.
    min_num_masked = cast(int, torch.count_nonzero(float_mask, dim=-1).min().item())

    # We randomly pick `min_num_masked` masked elements from each row, which
    # effectively unmasks the remaining elements.
    indices = torch.multinomial(float_mask, num_samples=min_num_masked)

    # Now we construct the actual boolean mask which has the same number of
    # masked elements in each row.
    bool_mask = torch.full_like(float_mask, False, dtype=torch.bool)

    return bool_mask.scatter_(1, indices, True)


def apply_temporal_mask(x: Tensor, temporal_mask: Tensor) -> Tensor:
    """Apply the specified temporal mask to ``x``.
       
       :param x: 
             shape: (B,T,C)
       :param temporal_mask: 
             shape: (B,T), A boolean mask, all True elements will be masked.
       :returns:
             shape: (T_true_nums,C), T_true_nums: sum of true positions of all utterances in temporal_mask
    """

    return x[temporal_mask].unflatten(0, (x.size(0), -1))  # type: ignore[no-any-return]



## how to use the script:
if __name__ == "__main__":
    shape = (32, 512)

    mask = compute_mask(shape, span_len=10, max_mask_prob=0.65, device=device)

    assert mask is not None

    num_masked = torch.count_nonzero(mask, dim=-1)
    print(f"mask: {mask}")
    row_lens = torch.tensor([16, 14, 15, 16], device="cpu")
    
    mask1 = compute_mask(
        shape, span_len=4, max_mask_prob=1.0, device=device, row_lens=row_lens
    )
    print(f"mask1: {mask1}")

    seqs = torch.zeros((4, 6), device=device)

    seq_lens = torch.tensor([4, 2, 0, 5], device=device, dtype=torch.int32)

    mask2 = to_padding_mask(seqs, seq_lens)

    inf = -torch.inf

    expected_mask = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, inf, inf],
            [0.0, 0.0, inf, inf, inf, inf],
            [inf, inf, inf, inf, inf, inf],
            [0.0, 0.0, 0.0, 0.0, 0.0, inf],
        ],
        device=device,
    )

    assert mask2 is not None
    print(f"mask2: {mask2}")
    #assert_equal(mask2, expected_mask)
    seqs = torch.zeros((4, 6), device=device)

    mask3 = to_padding_mask(seqs, seq_lens=None)
    print(f"mask3: {mask}")
