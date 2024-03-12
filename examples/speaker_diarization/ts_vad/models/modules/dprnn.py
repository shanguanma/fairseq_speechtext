import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPathRNN(nn.Module):
    def __init__(
        self,
        N,
        dprnn_output_channel,
        dprnn_hidden_state,
        dprnn_chunk_size,
        dprnn_layers,
        spk_embed_dim,
    ):
        # N, B, H, K, R
        super(DualPathRNN, self).__init__()
        self.dprnn_chunk_size, self.dprnn_layers = dprnn_chunk_size, dprnn_layers
        self.group_norm = nn.GroupNorm(1, N, eps=1e-8)
        self.bottleneck_conv1x1 = nn.Conv1d(N, dprnn_output_channel, 1, bias=False)

        self.dual_rnn = nn.ModuleList([])
        for _ in range(dprnn_layers):
            self.dual_rnn.append(
                DualRNNBlock(
                    dprnn_output_channel,
                    dprnn_hidden_state,
                    rnn_type="LSTM",
                    dropout=0,
                    bidirectional=True,
                )
            )

        self.prelu = nn.PReLU()
        self.av_conv = nn.Conv1d(
            dprnn_output_channel + spk_embed_dim, dprnn_output_channel, 1, bias=False
        )
        # self.mask_conv1x1 = nn.Conv1d(dprnn_output_channel, N, 1, bias=False)
        self._output_dim = dprnn_output_channel

    def forward(self, x, spk_emb):
        # M, N, D = x.size()

        # spk_emb, spk_pred = self.spk_encoder(aux, aux_len)
        spk_emb = torch.repeat_interleave(
            spk_emb.unsqueeze(2), repeats=x.size(2), dim=2
        )
        x = self.group_norm(x)  # [M, N, K]
        x = self.bottleneck_conv1x1(x)  # [M, B, K]

        x = torch.cat((x, spk_emb), 1)
        x = self.av_conv(x)

        x, gap = self._Segmentation(x, self.dprnn_chunk_size)  # [M, B, k, S]

        for i in range(self.dprnn_layers):
            x = self.dual_rnn[i](x)

        x = self._over_add(x, gap)

        x = self.prelu(x)
        # x = x.transpose(1, 2)
        # x = self.mask_conv1x1(x)

        # x = x.view(M, N, D) # [M, C*N, K] -> [M, C, N, K]
        # x = F.leaky_relu(x)
        return x

    def _padding(self, input, K):
        """
        padding the audio times
        K: chunks of length
        P: hop size
        input: [B, N, L]
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """
        the segmentation stage splits
        K: chunks of length
        P: hop size
        input: [B, N, L]
        output: [B, N, K, S]
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """
        Merge sequence
        input: [B, N, K, S]
        gap: padding length
        output: [B, N, L]
        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class DualRNNBlock(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        rnn_type="LSTM",
        dropout=0,
        bidirectional=False,
    ):
        super(DualRNNBlock, self).__init__()

        # RNN model
        self.intra_rnn = getattr(nn, rnn_type)(
            out_channels,
            hidden_channels,
            1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels,
            hidden_channels,
            1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Norm
        self.intra_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        self.inter_norm = nn.GroupNorm(1, out_channels, eps=1e-8)

        # Linear
        self.intra_linear = nn.Linear(
            hidden_channels * 2 if bidirectional else hidden_channels, out_channels
        )
        self.inter_linear = nn.Linear(
            hidden_channels * 2 if bidirectional else hidden_channels, out_channels
        )

    def forward(self, x):
        """
        x: [B, N, K, S]
        out: [Spks, B, N, K, S]
        """
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        # [BS, K, N]
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B * S * K, -1)).view(
            B * S, K, -1
        )
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)

        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        # [BK, S, H]
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B * S * K, -1)).view(
            B * K, S, -1
        )
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out
