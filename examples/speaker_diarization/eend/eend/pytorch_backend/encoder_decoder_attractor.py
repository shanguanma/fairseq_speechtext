#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

import torch
import torch.nn as nn
class LstmEncoderDedecoderAttractor(nn.Module):
    def __init__(self,input_size, num_layers=1, dropout=0.1):
        super(LstmEncoderDedecoderAttractor,self).__init__()
        self.encoder = nn.LSTM(input_size, input_size, num_layers,batch_first=True)
        self.decoder = nn.LSTM(input_size, input_size, num_layers,batch_first=True)

        self.linear = nn.Linear(input_size,1)
        self.dropout_layer = torch.nn.Dropout(p=dropout)

        self.input_size = input_size

    def forward(self,emb: torch.tensor, emb_length: torch.tensor,decoder_zero_input: torch.tensor):
        """Forward
        Args:
            emb(torch.tensor): output of encoder(i.e. transformer encoder or blstm encoder),shape(B,T,D)
            emb_length(torch.tensor): output frame length(actual number of frames without pad) of encoder(i.e. transformer encoder or blstm encoder), shape(B,)
            decoder_zeor_input(torch.tensor): zero input of decoder,shape(B, num_spk+1, D)
        Returns:
            attractors(torch.tensor): attractors of before linear transformer, shape(B,num_spk+1,D)
            attractors_prob(torch.tensor): attractor after linear transformer, shape(B,num_spk+1,1)

            # B: batch size, num_spk : number of speakers, D: dimension of feature
            # T: number of frames

        """
        assert emb.size(2)==self.input_size
        
        ## remove pad before pass into lstm network
        pack = torch.nn.utils.rnn.pack_padded_sequence(
            emb, lengths=emb_length.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hx, cx) = self.encoder(pack)
        attractors,(_,_) = self.decoder(decoder_zero_input, (hx,cx))#(B,S+1,D)
        attractors_prob = self.linear(attractors) # (B,S+1,1)
        # in order to use torch.nn.BCEWithLogitsLoss, because
        # This loss combines a Sigmoid layer and the BCELoss in one single class.
        # This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as,
        # by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.

        #prob = self.act(attractors_linear)#(B,S,1) #
        return attractors, attractors_prob
