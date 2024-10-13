import torch.nn as nn
import torch
from torchinfo import summary
from .model_utils import *
import numpy as np
from math import sqrt
from sklearn.cluster import SpectralClustering
from collections import defaultdict
import warnings
from scipy import sparse as sp
# import hashlib
# import dgl
import torch.nn.parallel
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore", category=UserWarning)
from einops import repeat
from einops import rearrange


class sSelfAttentionLayer(nn.Module):
    def __init__(
            self, batch_size, num_nodes, model_dim, num_heads, dropout, in_steps=12, mask=False
    ):
        super().__init__()
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.eattn = AttentionLayer(model_dim, num_heads, dropout, mask)
        self.dattn = AttentionLayer(model_dim, num_heads, dropout, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

    def forward(self, X, dim=-2):
        x = X.transpose(dim, -2)
        patch_len = self.num_nodes
        global_patch = x.shape[2] - patch_len
        # x: (batch_size, ..., length, model_dim)
        residual = x[:, :, global_patch:, :]
        t = self.eattn(x[:, :, :global_patch, :], x[:, :, -patch_len:, :],
                       x[:, :, -patch_len:, :])  # (batch_size, ..., length, model_dim)
        # t = self.eattn(x, ttoken, ttoken)  # (batch_size, ..., length, model_dim)

        x[:, :, :global_patch, ] = t
        out = self.dattn(x[:, :, global_patch:, :], x, x)  # (batch_size, ..., length, model_dim)

        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = torch.cat((t, out), dim=-2)

        out = out.transpose(dim, -2)
        return out


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads, dropout, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
                             query @ key
                     ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)

        attn_score = self.dropout(attn_score)

        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, batch_size, num_nodes, model_dim, num_heads, dropout, in_steps=12, mask=False
    ):
        super().__init__()
        self.in_steps = in_steps
        self.dropout = dropout
        self.eattn = AttentionLayer(model_dim, num_heads, dropout, mask)
        self.dattn = AttentionLayer(model_dim, num_heads, dropout, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

    def forward(self, X, dim=-2):
        x = X.transpose(dim, -2)
        # ttoken=ttoken.transpose(dim, -2)
        patch_len = self.in_steps
        # x: (batch_size, ..., length, model_dim)
        residual = x[:, :, 3:, ]
        g = self.eattn(x[:, :, :3, :], x[:, :, -patch_len:, :],
                       x[:, :, -patch_len:, :])  # (batch_size, ..., length, model_dim)
        # g = self.eattn(x, ttoken, ttoken)  # (batch_size, ..., length, model_dim)

        x[:, :, :3, ] = g
        out = self.dattn(x[:, :, 3:, ], x, x)  # (batch_size, ..., length, model_dim)

        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = torch.cat((g, out), dim=-2)

        out = out.transpose(dim, -2)

        return out


class Temporal_Layer(nn.Module):
    def __init__(self, batch_size, in_steps, num_nodes, model_dim, num_heads_gtu, dropout):
        super(Temporal_Layer, self).__init__()
        self.gtu = GTU_Layer(batch_size, in_steps, num_nodes, model_dim, num_heads_gtu, dropout)
        self.skipcon = Skipcon(model_dim)

    def forward(self, X):
        HT_gtu = self.gtu(X)
        out = self.skipcon(HT_gtu, X)
        # del HT_gtu
        return out


class GTU_Layer(nn.Module):
    def __init__(self, batch_size, T, num_nodes, model_dim, num_heads_gtu, dropout, mask=False):
        super(GTU_Layer, self).__init__()
        self.gtu2 = GTU(model_dim, 2, kernel_size=2)
        self.gtu4 = GTU(model_dim, 4, kernel_size=4)
        self.gtu6 = GTU(model_dim, 6, kernel_size=6)
        self.dattn = AttentionLayer(model_dim, num_heads_gtu, dropout, mask)

    def forward(self, X):
        # X_STE = torch.cat((X, STE), dim=-1)  # 32,12,325,128
        # X = self.FC_T(X_STE)
        batch_size, in_steps, _, _ = X.shape

        x = X.permute(0, 3, 2, 1)  # 16,64,307,12

        x_ggu = []
        x_ggu.append(self.gtu2(x))  # B,F,N,T-2  6
        x_ggu.append(self.gtu4(x))  # B,F,N,T-4  3
        x_ggu.append(self.gtu6(x))  # B,F,N,T-6 2
        g_conv = torch.cat(x_ggu, dim=-1)  # B,F,N,3T-12

        patch_len = in_steps
        g_conv = g_conv.permute(0, 2, 3, 1)
        x = X.transpose(1, 2)
        g_conv = torch.cat((g_conv, x), dim=2)  # B,F,N,3T-12
        g_conv = self.dattn(g_conv, g_conv, g_conv)
        g_conv = g_conv[:, :, -patch_len:, :]
        x = g_conv.transpose(1, 2)

        return x


class Routeformer(nn.Module):
    def __init__(
            self,
            num_nodes,
            num_heads_gtu,
            num_heads,
            num_heads_s,
            num_tlayer,
            num_convlayer,
            num_slayer,
            num_trouter,
            num_srouter,
            emb_flag,
            in_steps=12,
            out_steps=12,
            steps_per_day=288,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=64,
            tod_embedding_dim=32,
            dow_embedding_dim=32,
            spatial_embedding_dim=64,
            feed_forward_dim=256,
            model_dim=64,
            num_layers=3,
            dropout=0.1,
            use_mixed_proj=True
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.model_dim = (model_dim
                          )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)  # 288,24
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        # if spatial_embedding_dim > 0:
        #     self.node_emb = nn.Parameter(
        #         torch.empty(in_steps, self.num_nodes, self.spatial_embedding_dim)
        #     )
        #     nn.init.xavier_uniform_(self.node_emb)

        self.tcat_token = nn.Parameter(torch.randn(1, num_trouter, num_nodes, model_dim))
        self.cls_token = nn.Parameter(torch.randn(1, in_steps, num_srouter, model_dim))

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)
        self.batch_size = 16
        self.TemporalBlock_1 = nn.ModuleList(
            [Temporal_Layer(self.batch_size, self.in_steps, self.num_nodes, model_dim, num_heads_gtu, dropout) for _ in
             range(num_convlayer)])

        self.FCX = nn.Linear(tod_embedding_dim + dow_embedding_dim + input_embedding_dim, model_dim)
        self.FCX_emb = nn.Linear(
            tod_embedding_dim + dow_embedding_dim + input_embedding_dim + self.spatial_embedding_dim, model_dim)
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.batch_size, num_nodes, model_dim, num_heads, dropout, in_steps=in_steps)
                for _ in range(num_tlayer)
            ]
        )
        self.attn_layers_s = nn.ModuleList(
            [
                sSelfAttentionLayer(self.batch_size, num_nodes, model_dim, num_heads_s, dropout, in_steps=in_steps)
                for _ in range(num_slayer)
            ]
        )
        self.num_convlayer = num_convlayer
        self.num_tlayer = num_tlayer
        self.num_slayer = num_slayer
        self.emb_flag = emb_flag
        self.skipcon = Skipcon(model_dim)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)

        x = history_data

        # res_x = x[..., 0].unsqueeze(-1)
        batch_size = x.size(0)
        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]

        x = x[..., : self.input_dim]  # 16，12，170，3
        Xo = self.input_proj(x)
        batchsize, in_steps, num_nodes, model_dim = Xo.shape

        features = []
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()  # 16，12，170，24=tod_embedding(288,24).[(16,12,170,288)]   288条tod
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)  
        if self.emb_flag:  
            spatial_emb = self.node_emb.expand(
                size=(batch_size, *self.node_emb.shape)
            )
            features.append(spatial_emb)

        STE = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim 152)

        X = torch.cat((Xo, STE), dim=-1)
        if self.emb_flag:
            X = self.FCX_emb(X)
        else:
            X = self.FCX(X)

        if self.num_tlayer > 0:
            patch_len = X.shape[1]
            tcat_token = repeat(self.tcat_token, '1 n t d -> b n t d', b=X.shape[0])
            X = torch.cat([tcat_token, X], dim=1)
            for attn in self.attn_layers_t:
                X = attn(X, dim=1)
            X = X[:, -patch_len:, :, :]

        if self.num_convlayer > 0:
            for net in self.TemporalBlock_1:
                X = net(X)

        if self.num_slayer > 0:
            batchrouter = repeat(self.cls_token, '1 t n d -> b t n d', b=X.shape[0])
            patch_len = X.shape[2]
            X = torch.cat([batchrouter, X], dim=2)
            for attn in self.attn_layers_s:
                X = attn(X, dim=2)
            X = X[:, :, -patch_len:, :]



        if self.use_mixed_proj:
            out = X.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )  # 16，170，1824
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )

            # out = self.mlp_2(out)
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = X.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out
