# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TransformerEncoder(nn.Module):
    """
    len_input: 指的是输入序列的长度

    """
    def __init__(self, len_input, d_model, dim_feedforward, n_head, norm=None):
        super().__init__()
        self.layer0 = TransformerEncoderLayer(len_input, len_input//2, d_model, n_head, dim_feedforward)
        self.layer1 = TransformerEncoderLayer(len_input//2, len_input//4, d_model, n_head, dim_feedforward)
        self.layer2 = TransformerEncoderLayer(len_input//4, len_input//8, d_model, n_head, dim_feedforward)
        self.layer3 = TransformerEncoderLayer(len_input//8, len_input//16, d_model, n_head, dim_feedforward)
        self.norm = norm

        self.linear = nn.Linear(len_input//16, len_input)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output0 = self.layer0(src)
        output1 = self.layer1(output0)
        output2 = self.layer2(output1)
        output3 = self.layer3(output2)
        output = output3

        if self.norm is not None:
            output = self.norm(output)
        output = output.permute(2, 1, 0)
        output = self.linear(output)
        output = output.permute(2, 1, 0)
        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, len_input, len_output, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.linear3 = nn.Linear(len_input, len_output)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        ## 查看attention权重
        # src2_, src2_weight = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)
        # weight = src2_weight[0, :, :].max(1)
        # import numpy as np
        # torch.set_printoptions(threshold=np.inf)
        # print(weight)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.permute(2, 1, 0)
        output_src = self.linear3(src)
        output_src = output_src.permute(2, 1, 0)

        ## 查看linear3权重
        # weight = self.linear3.weight[0, :]
        # print(weight)

        return output_src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


if __name__ == '__main__':
    input = torch.randn((450, 1, 256))
    encoder = TransformerEncoder(len_input=450, d_model=256, dim_feedforward=2048, n_head=4)
    output = encoder(input)

