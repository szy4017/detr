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
import random

from models.masking import Masking, batch_index_select

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, sta_query=False, deformable_decoder=False, sta_query_loc=None, sta_mask=False):
        super().__init__()

        # for encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # for decoder
        self.deformable_decoder = deformable_decoder
        self.sta_mask = sta_mask
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        if self.sta_mask:
            # self.sta_mask_transform = state_mask_embed()
            self.decoder = TransformerStateMaskDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                                       return_intermediate=return_intermediate_dec,
                                                       sta_query=sta_query,
                                                       embed_dim=256,
                                                       pruning_loc=[2, 4],
                                                       token_ratio=[0.8, 0.6])
        else:
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec,
                                              sta_query=sta_query)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.sta_query = sta_query
        self.sta_query_loc = sta_query_loc

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_sta_mask(self, hs_tgt, h, w):
        L, N, B, C = hs_tgt.shape
        hs_tgt = hs_tgt[-1, :, :, :]
        hs_tgt = hs_tgt.transpose(1, 0)
        hs_tgt = torch.reshape(hs_tgt, (B, 5, 10, C)).permute(0, 3, 1, 2)
        mask_feature = nn.functional.interpolate(hs_tgt, size=(h, w), mode='nearest', align_corners=None)
        mask_feature = self.sta_mask_transform(mask_feature)
        mask = mask_feature[:, 0, :, :] < mask_feature[:, 1, :, :]
        mask_flatten = mask.flatten(1)
        return mask_flatten

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        tgt_query_embed = query_embed['tgt'].weight.unsqueeze(1).repeat(1, bs, 1).to(src.device)
        tgt = torch.zeros_like(tgt_query_embed)  # target query
        if self.sta_query:
            sta_query_embed = query_embed['sta'].weight.unsqueeze(1).repeat(1, bs, 1).to(src.device)
            sta = torch.zeros_like(sta_query_embed)

        mask_flatten = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask_flatten, pos=pos_embed)

        if self.sta_query:
            if self.sta_query_loc == 'backbone':
                if self.sta_mask:
                    # sta_mask_flatten = self.get_sta_mask(hs_tgt, h, w)
                    # sta_mask_flatten = mask_flatten
                    # num = sta_mask_flatten.shape[-1]
                    # true_id = [random.sample(range(0, num), num//2)]
                    # sta_mask_flatten[:, true_id] = True
                    ## 测试，保存mask图
                    # save_mask = sta_mask_flatten.cpu().numpy()[0]
                    # save_mask = save_mask.reshape((h, w)).astype(int) * 255
                    # import cv2
                    # cv2.imwrite('mask.png', save_mask)

                    memory_target_mask = torch.zeros((bs, h * w, 1), dtype=src.dtype, device=src.device).bool()
                    hs_tgt, _ = self.decoder(tgt, memory, flag=0, memory_key_padding_mask=mask_flatten, pos=pos_embed,
                                             query_pos=tgt_query_embed, memory_target_mask=memory_target_mask)
                    hs_sta, mask_loss = self.decoder(sta, src, flag=1, query=hs_tgt, memory_key_padding_mask=mask_flatten,
                                                     pos=pos_embed,
                                                     query_pos=sta_query_embed, memory_target_mask=memory_target_mask)
                    return hs_tgt.transpose(1, 2), hs_sta.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h,
                                                                                                        w), mask_loss
                else:
                    hs_tgt = self.decoder(tgt, memory, memory_key_padding_mask=mask_flatten, pos=pos_embed,
                                          query_pos=tgt_query_embed)
                    hs_sta = self.decoder(sta, src, memory_key_padding_mask=mask_flatten, pos=pos_embed,
                                          query_pos=sta_query_embed)  # state query in backbone features
            else:
                hs_tgt = self.decoder(tgt, memory, memory_key_padding_mask=mask_flatten, pos=pos_embed,
                                      query_pos=tgt_query_embed)
                hs_sta = self.decoder(sta, memory, memory_key_padding_mask=mask_flatten, pos=pos_embed,
                                      query_pos=sta_query_embed)  # state query in decoder features
            return hs_tgt.transpose(1, 2), hs_sta.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        else:
            hs_tgt = self.decoder(tgt, memory, memory_key_padding_mask=mask_flatten, pos=pos_embed,
                                  query_pos=tgt_query_embed)
            return hs_tgt.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, sta_query=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.sta_query = sta_query

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerStateMaskDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 sta_query=True, sta_mask=True, embed_dim=256, pruning_loc=[2, 4], token_ratio=[0.6, 0.3]):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.sta_query = sta_query
        self.sta_mask = sta_mask
        self.pruning_loc = pruning_loc
        if self.sta_mask:
            self.atten_mask_predict = Masking(embed_dim, pruning_loc, token_ratio, num_layers)

    def forward(self, tgt, memory, flag, query=None,
                tgt_mask: Optional[Tensor] = None,
                memory_target_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        mask_loss_list = []
        pre_memory_target_mask = memory_target_mask
        for i, layer in enumerate(self.layers):
            if (i >= self.pruning_loc[0]) and flag == 1:
                if self.training:
                    post_memory_target_mask, target_mask_loss = self.atten_mask_predict(memory, query, (~pre_memory_target_mask).long(), i)
                    post_memory_target_mask = ~post_memory_target_mask.bool().transpose(2, 1)
                    post_memory_mask = post_memory_target_mask.repeat(8, 50, 1)
                    mask_loss_list.append(target_mask_loss)
                else:
                    post_memory_target_mask, keep_policy = self.atten_mask_predict(memory, query, (~pre_memory_target_mask).long(), i)
                    post_memory_target_mask = ~post_memory_target_mask.bool()
                    post_memory_mask = post_memory_target_mask.squeeze().repeat(8, 1, 1)
                    memory = batch_index_select(memory.transpose(1, 0), keep_policy).transpose(1, 0)
                    pos = batch_index_select(pos.transpose(1, 0), keep_policy).transpose(1, 0)
                    memory_key_padding_mask = batch_index_select(memory_key_padding_mask, keep_policy)
            else:
                post_memory_target_mask = pre_memory_target_mask
                post_memory_mask = post_memory_target_mask.transpose(2, 1)
                post_memory_mask = post_memory_mask.repeat(8, 50, 1)

            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=post_memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            pre_memory_target_mask = post_memory_target_mask
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), mask_loss_list

        return output.unsqueeze(0), mask_loss_list


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
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
        #src2_, src2_weight = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              #key_padding_mask=src_key_padding_mask)
        #weight = src2_weight[0, :, :].max(1)
        #import numpy as np
        #torch.set_printoptions(threshold=np.inf)
        #print(weight)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

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


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    ## 这里qkv计算分为两个部分，第一部分是object对于自身的qkv相关性计算，第二部分是object与encoder输出特征memory的qkv相关性计算，这里q是object，kv是memory
    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, input, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt = input

        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        else:
            return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                     tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class state_mask_embed(nn.Module):
    """ A simple multi-layer perceptron for class classification."""

    def __init__(self, output_dim=2):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 128, (3, 3), 1, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, (3, 3), 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 2, (3, 3), 1, 1)
        self.bn3 = nn.BatchNorm2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        out = torch.softmax(x, dim=1)
        return out


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        sta_query=args.sta_query,
        deformable_decoder=args.deformable_decoder,
        sta_query_loc=args.sta_query_loc,
        sta_mask=args.sta_mask
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
