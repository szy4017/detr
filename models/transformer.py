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
        if self.deformable_decoder is not True:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec, sta_query=sta_query)
        else:
            # for deformable decoder
            from .decoder_deformable import DeformableTransformerDecoderLayer, DeformableTransformerDecoder
            decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                              dropout, activation,
                                                              4, nhead, 4)
            self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
            self.reference_points = nn.Linear(d_model, 2)  # get reference points for deformable decoder

        self.sta_mask = sta_mask
        if self.sta_mask:
            self.sta_mask_transform = state_mask_embed()

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

    def deformable_decoder_forward(self, query, mask, feature, query_embed):
        bs, c, h, w = feature.shape
        mask_flatten = mask.flatten(1)
        reference_points = self.reference_points(query.transpose(1, 0)).sigmoid()
        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long, device=feature.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(mask)], 1)
        hs, inter_references = self.decoder(query.permute(1, 0, 2), reference_points, feature.permute(1, 0, 2),
                                                    spatial_shapes, level_start_index,
                                                    valid_ratios, query_embed.permute(1, 0, 2), mask_flatten)
        hs = hs.transpose(1, 2)
        return hs.transpose(1, 2)

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

        if self.deformable_decoder is not True:
            hs_tgt = self.decoder(tgt, memory, memory_key_padding_mask=mask_flatten, pos=pos_embed, query_pos=tgt_query_embed)

            if self.sta_query:
                if self.sta_query_loc == 'backbone':
                    if self.sta_mask:
                        # sta_mask_flatten = self.get_sta_mask(hs_tgt, h, w)
                        sta_mask_flatten = mask_flatten
                        num = sta_mask_flatten.shape[-1]
                        true_id = [random.sample(range(0, num), num//2)]
                        sta_mask_flatten[:, true_id] = True
                        ## 测试，保存mask图
                        # save_mask = sta_mask_flatten.cpu().numpy()[0]
                        # save_mask = save_mask.reshape((h, w)).astype(int) * 255
                        # import cv2
                        # cv2.imwrite('mask.png', save_mask)

                        hs_sta = self.decoder(sta, src, memory_key_padding_mask=sta_mask_flatten, pos=pos_embed,
                                            query_pos=sta_query_embed)    # state query in backbone features
                    else:
                        hs_sta = self.decoder(sta, src, memory_key_padding_mask=mask_flatten, pos=pos_embed,
                                            query_pos=sta_query_embed)    # state query in backbone features
                else:
                    hs_sta = self.decoder(sta, memory, memory_key_padding_mask=mask_flatten, pos=pos_embed,
                                          query_pos=sta_query_embed)    # state query in decoder features
                return hs_tgt.transpose(1, 2), hs_sta.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
            else:
                return hs_tgt.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        else:
            # for deformable decoder
            hs_tgt = self.deformable_decoder_forward(tgt, mask, memory, tgt_query_embed)

            if self.sta_query:
                if self.sta_query_loc == 'backbone':
                    hs_sta = self.deformable_decoder_forward(sta, mask, src, sta_query_embed)    # state query in backbone features
                else:
                    hs_sta = self.deformable_decoder_forward(sta, mask, memory, sta_query_embed)    # state query in decoder features
                return hs_tgt.transpose(1, 2), hs_sta.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
            else:
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

        # self.sta_query = sta_query
        # if self.sta_query:
        #     Self attention for state query
            # self.self_attn_sta = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            # self.multihead_attn_sta = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            # Feedforward model for state query
            # self.linear1_sta = nn.Linear(d_model, dim_feedforward)
            # self.dropout_sta = nn.Dropout(dropout)
            # self.linear2_sta = nn.Linear(dim_feedforward, d_model)

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

    # no use
    ## 增加state query
    # def forward_post_state(self, sta, memory,
    #                        sta_mask: Optional[Tensor] = None,
    #                        memory_mask: Optional[Tensor] = None,
    #                        sta_key_padding_mask: Optional[Tensor] = None,
    #                        memory_key_padding_mask: Optional[Tensor] = None,
    #                        pos: Optional[Tensor] = None,
    #                        query_pos: Optional[Tensor] = None):
    #     q = k = self.with_pos_embed(sta, query_pos) ## 给state query添加位置编码
    #     sta2 = self.self_attn_sta(q, k, value=sta, attn_mask=sta_mask,
    #                           key_padding_mask=sta_key_padding_mask)[0] ## 进行一次self attention操作，得到输出sta2
    #     sta = sta + self.dropout1(sta2) ## 对sta2进行drop out操作
    #     sta = self.norm1(sta)   ## 归一化处理
    #     sta2 = self.multihead_attn_sta(query=self.with_pos_embed(sta, query_pos),
    #                                key=self.with_pos_embed(memory, pos),
    #                                value=memory, attn_mask=memory_mask,
    #                                key_padding_mask=memory_key_padding_mask)[0] ## 再进行一次self attenrion操作，这里引入了memory变量，memory是decoder的输出，是编码的图像特征
    #     sta = sta + self.dropout2(sta2) ## 对sta2进行drop out操作
    #     sta = self.norm2(sta)   ## 归一化处理
    #     sta2 = self.linear2_sta(self.dropout_sta(self.activation(self.linear1_sta(sta))))   ## 对sta进行线性层特征转换，feed forward model
    #     sta = sta + self.dropout3(sta2) ## 对sta2进行drop out操作
    #     sta = self.norm3(sta)   #¥ 归一化处理
    #     return sta

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
