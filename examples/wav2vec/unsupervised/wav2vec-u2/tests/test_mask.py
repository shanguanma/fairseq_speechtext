from fairseq.data.data_utils import compute_mask_indices

import torch



def apply_feature_mask(x, padding_mask):
    mask_emb = torch.nn.Parameter(
        torch.FloatTensor(2).uniform_()
    )

    B, T, C = x.shape
    mask_prob=0.8
    mask_length=3
    mask_selection="static"
    mask_other=0   
    no_mask_overlap=False
    mask_min_space=2 
    mask_indices = compute_mask_indices(
        (B, T),
        padding_mask,
        mask_prob,
        mask_length,
        mask_selection,
        mask_other,
        min_masks=2,
        no_overlap=no_mask_overlap,
        min_space=mask_min_space,
    )
    mask_indices = torch.from_numpy(mask_indices).to(x.device)
    x[mask_indices] = mask_emb
    return x, mask_indices

if __name__ == "__main__":
    x = torch.randn(1,10,2)
    padding = torch.randn(1,10)
    padding_mask = torch.BoolTensor(padding.shape).fill_(False)
    x_mask, mask_indices =  apply_feature_mask(x, padding_mask)
    print(f"input: {x}, its shape: {x.shape}")
    print(f"after masked input: {x_mask}, its shape: {x_mask.shape}")
    print(f"after masked input mask indices: {mask_indices}, its shape: {mask_indices.shape}")
