import torch
from torch import nn
import torch.nn.functional as F

from ts_vad.models.modules.modules import Conv1DBlock_v2, Conv1DBlock, Conv1D


class TCN(nn.Module):
    def __init__(
        self,
        spk_embed_dim,
        enc_out_size,
        conv_out_channels,
        conv_kernel_size,
        conv_block_norm,
        total_conv_block_num,
        conv_block_num,
        num_spk=None,
        spk_embed_input=False,
    ):
        super(TCN, self).__init__()
        self.spk_embed_input = spk_embed_input
        if spk_embed_input:
            self.conv_blocks = nn.ModuleList(
                [
                    Conv1DBlock_v2(
                        spk_embed_dim=spk_embed_dim,
                        in_channels=enc_out_size,
                        conv_channels=conv_out_channels,
                        kernel_size=conv_kernel_size,
                        norm=conv_block_norm,
                        dilation=1,
                    )
                    for _ in range(total_conv_block_num)
                ]
            )
        else:
            self.conv_blocks = nn.ModuleList(
                [
                    Conv1DBlock(
                        in_channels=enc_out_size,
                        conv_channels=conv_out_channels,
                        kernel_size=conv_kernel_size,
                        norm=conv_block_norm,
                        dilation=1,
                    )
                    for _ in range(total_conv_block_num)
                ]
            )

        self.conv_blocks_other = nn.ModuleList(
            [
                self._build_blocks(
                    num_blocks=conv_block_num,
                    in_channels=enc_out_size,
                    conv_channels=conv_out_channels,
                    kernel_size=conv_kernel_size,
                    norm=conv_block_norm,
                )
                for _ in range(total_conv_block_num)
            ]
        )

        if num_spk is not None:
            self.mask_module = nn.ModuleList(
                [Conv1D(enc_out_size, enc_out_size, 1) for _ in range(num_spk)]
            )

    def _build_blocks(self, num_blocks, **block_kwargs):
        """
        Build Conv1D block
        """
        blocks = [
            Conv1DBlock(**block_kwargs, dilation=(2**b)) for b in range(1, num_blocks)
        ]
        return nn.Sequential(*blocks)

    def forward(self, y, aux=None, include_mask=False):
        for conv_block, conv_block_other in zip(
            self.conv_blocks, self.conv_blocks_other
        ):
            if self.spk_embed_input:
                y = conv_block(y, aux)
            else:
                y = conv_block(y)
            y = conv_block_other(y)

        if include_mask:
            mask = [torch.nn.functional.relu(mm(y)) for mm in self.mask_module]
            return mask, y
        else:
            return y
