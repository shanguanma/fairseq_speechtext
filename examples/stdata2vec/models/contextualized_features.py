import torch
from collections import namedtuple
from .mask import compute_mask

MaskSeed = namedtuple("MaskSeed", ["seed", "update", "ids"])
MaskInfo = namedtuple("MaskInfo", ["x_unmasked", "mask", "ids_restore", "ids_keep"])


def contextualized_features(
    self,
    x,
    padding_mask,
    convert_padding_mask,
    fixed_positional_encoder,
    relative_positional_encoder,
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
    get_alibi_bias,
    num_alibi_heads,
    alibi_scale,
    mask,
    remove_masked,
    prenet_depth,
    context_encoder, ## (TODO) md check,it  maybe function
    clone_batch: int = 1,
    mask_seeds: Optional[torch.Tensor] = None,
    precomputed_mask=None,
):

    if padding_mask is not None:
        padding_mask = convert_padding_mask(x, padding_mask)

    local_features = x
    if mask and clone_batch == 1:
        local_features = local_features.clone()

    orig_B, orig_T, _ = x.shape
    pre_mask_B = orig_B
    mask_info = None

    x_pos = None
    if fixed_positional_encoder is not None:
        x = x + fixed_positional_encoder(x, padding_mask)
    if mask:
        if clone_batch > 1:
            x = x.repeat_interleave(clone_batch, 0)
            if mask_seeds is not None:
                clone_hash = [
                    int(hash((mask_seeds.seed, ind)) % 1e10)
                    for ind in range(clone_batch - 1)
                ]
                clone_hash = torch.tensor([0] + clone_hash).long().view(1, -1)

                id = mask_seeds.ids
                id = id.repeat_interleave(clone_batch, 0)
                id = id.view(-1, clone_batch) + clone_hash.to(id)
                id = id.view(-1)
                mask_seeds = MaskSeed(
                    seed=mask_seeds.seed, update=mask_seeds.update, ids=id
                )
            if padding_mask is not None:
                padding_mask = padding_mask.repeat_interleave(clone_batch, 0)

        x, mask_info = compute_mask(
            x,
            padding_mask,
            mask_seed=mask_seeds,
            apply=self.relative_positional_encoder is not None or not remove_masked,
            precomputed_mask=precomputed_mask,
            mask_prob=mask_prob,
            mask_prob_min=mask_prob_min,
            inverse_mask=inverse_mask,
            mask_dropout=mask_dropout,
            add_masks=add_masks,
            mask_length=mask_length,
            keep_masked_pct=keep_masked_pct,
            encoder_zero_mask=encoder_zero_mask,
            mask_noise_std=mask_noise_std,
            mask_channel_prob=mask_channel_prob,
        )

    if relative_positional_encoder is not None:
            x_pos = relative_positional_encoder(x)
    masked_padding_mask = padding_mask
    if mask and remove_masked:
        x = mask_info.x_unmasked
        if x_pos is not None:
            x = x + gather_unmasked(x_pos, mask_info)

        if padding_mask is not None and padding_mask.any():
            masked_padding_mask = gather_unmasked_mask(padding_mask, mask_info)
            if not masked_padding_mask.any():
                masked_padding_mask = None
        else:
            masked_padding_mask = None

    elif x_pos is not None:
        x = x + x_pos

    alibi_bias = None
    if get_alibi_bias is not None:
        alibi_bias = get_alibi_bias(
            batch_size=pre_mask_B,
            time_steps=orig_T,
            heads=num_alibi_heads,
            dtype=torch.float32,
            device=x.device,
        )

        if alibi_scale is not None:
            alibi_scale = alibi_scale.clamp_min(0)
            if alibi_scale.size(0) == 1:
                alibi_bias = alibi_bias * alibi_scale.squeeze(0).type_as(alibi_bias)
                alibi_scale = None

        if clone_batch > 1:
            alibi_bias = alibi_bias.repeat_interleave(clone_batch, 0)

        if mask_info is not None and remove_masked:
            alibi_bias = masked_alibi(alibi_bias, mask_info)


    x = context_encoder(
        x,
        masked_padding_mask,
        alibi_bias,
        alibi_scale[: prenet_depth]
        if alibi_scale is not None
        else None,
    )
    ouput={
        "x": x,
        "local_features": local_features,
        "padding_mask": masked_padding_mask,
        "alibi_bias": alibi_bias,
        "alibi_scale": alibi_scale[prenet_depth :]
        if alibi_scale is not None and alibi_scale.size(0) > 1
        else alibi_scale,
        "encoder_mask": mask_info,
    return output

def gather_unmasked(x: torch.Tensor, mask_info: MaskInfo) -> torch.Tensor:
    return torch.gather(
        x,
        dim=1,
        index=mask_info.ids_keep,
    )


def gather_unmasked_mask(x: torch.Tensor, mask_info: MaskInfo) -> torch.Tensor:
    return torch.gather(
        x,
        dim=1,
        index=mask_info.ids_keep[..., 0],  # ignore the feature dimension
    )


def get_alibi(
    max_positions: int,
    attention_heads: int,
    dims: int = 1,
    distance: str = "manhattan",
):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        # In the paper, we only train models that have 2^a heads for some
        # a. This function has some good properties that only occur when
        # the input is a power of 2. To maintain that even when the number
        # of heads is not a power of 2, we use this workaround.
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    maxpos = max_positions
    attn_heads = attention_heads
    slopes = torch.Tensor(get_slopes(attn_heads))

    if dims == 1:
        # prepare alibi position linear bias. Note that wav2vec2 is non
        # autoregressive model so we want a symmetric mask with 0 on the
        # diagonal and other wise linear decreasing valuees
        pos_bias = (
            torch.abs(
                torch.arange(maxpos).unsqueeze(0) - torch.arange(maxpos).unsqueeze(1)
            )
            * -1
        )
    elif dims == 2:
        if distance == "manhattan":
            df = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)
        elif distance == "euclidean":
            df = lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        n = math.sqrt(max_positions)
        assert n.is_integer(), n
        n = int(n)

        pos_bias = torch.zeros((max_positions, max_positions))

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        new_x = i * n + j
                        new_y = k * n + l
                        pos_bias[new_x, new_y] = -df(i, j, k, l)

    else:
        raise Exception(f"unsupported number of alibi dims: {dims}")

    alibi_bias = slopes.unsqueeze(1).unsqueeze(1) * pos_bias.unsqueeze(0).expand(
        attn_heads, -1, -1
    )

    return alibi_bias

def get_alibi_bias(
    alibi_biases,
    batch_size,
    time_steps,
    heads,
    dtype,
    device,
    dims=1,
    distance="manhattan",
):
    cache_key = f"{dims}_{heads}_{distance}"

    buffered = alibi_biases.get(cache_key, None)

    target_size = heads * batch_size
    if (
        buffered is None
        or buffered.size(0) < target_size
        or buffered.size(1) < time_steps
        or buffered.dtype != dtype
        or buffered.device != device
    ):
        bt = max(time_steps, buffered.size(1) if buffered is not None else 0)
        bn = max(target_size, buffered.size(0) if buffered is not None else 0) // heads

        buffered = (
            get_alibi(bt, heads, dims=dims, distance=distance)
            .to(dtype=dtype, device=device)
            .repeat(bn, 1, 1)
        )

        alibi_biases[cache_key] = buffered

    b = buffered[:target_size, :time_steps, :time_steps]
    b = b.view(batch_size, heads, time_steps, time_steps)
    return b


def _learned_alibi_bias(
    alibi_bias,
    batch_size,
    time_steps,
    heads,
    scale,
    dtype,
    device,
):
    assert alibi_bias.size(1) == heads, alibi_bias.shape
    assert alibi_bias.dtype == dtype, alibi_bias.dtype
    assert alibi_bias.device == device, alibi_bias.device

    if alibi_bias.size(-1) < time_steps:
        psz = math.ceil((time_steps - alibi_bias.size(-1)) / 2)
        alibi_bias = F.pad(alibi_bias, (psz, psz, psz, psz), mode="replicate")

    alibi_bias = alibi_bias.expand(batch_size, -1, -1, -1) * scale
    return alibi_bias[..., :time_steps, :time_steps]


def masked_alibi(alibi_bias, mask_info):
    H = alibi_bias.size(1)

    orig_bias = alibi_bias

    index = mask_info.ids_keep.unsqueeze(1)[..., 0].unsqueeze(-1)
    alibi_bias = torch.gather(
        orig_bias,
        dim=-2,
        index=index.expand(-1, H, -1, mask_info.ids_restore.size(1)),
    )
    alibi_bias = torch.gather(
        alibi_bias,
        dim=-1,
        index=index.transpose(-1, -2).expand(-1, H, alibi_bias.size(-2), -1),
    )

    return alibi_bias

