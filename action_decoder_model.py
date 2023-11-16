# Action decoder class file

import torch
from torch import nn, Tensor
import math
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        # even indices are sine, odd indices are cosine
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[ batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    


class ActionDecoderModel(nn.Module):

    def __init__(self, action_dim: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, action_seq_len: int = 5,
                 mem_seq_len: int = 5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=action_seq_len).cuda()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_hid, dropout, batch_first=True).cuda()
        self.transformer_decoder = TransformerDecoder(decoder_layer, nlayers).cuda()
        self.d_model = d_model
        self.linear_action_in = nn.Linear(action_dim, d_model).cuda()
        # The output of the transformer decoder is a sequence of length action_seq_len-1 because it doesn't have sos token
        self.linear_action_out = nn.Linear(d_model, action_dim).cuda()
        self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(action_seq_len).cuda()
        # causal memory mask to prevent attending to future actions of size mem_seq_len x action_seq_len upper triangular part should be True (is masked)
        self.mem_mask = torch.triu(torch.ones(action_seq_len, mem_seq_len), diagonal=1).bool().cuda()

    def forward(self, actions: Tensor, memory: Tensor) -> Tensor:
        """
        
        """
        actions = self.linear_action_in(actions)
        actions = self.pos_encoder(actions)
        # If mixing tasks, then we will need padding masks in the batch
        output = self.transformer_decoder(
            tgt=actions, 
            memory=memory, 
            tgt_mask=self.tgt_mask,
            memory_mask=self.mem_mask
            )
        
        output = self.linear_action_out(output)
        return output