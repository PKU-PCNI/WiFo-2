# coding=utf-8
from functools import partial

import torch
import torch.nn as nn
import math
import numpy as np
from timm.models.vision_transformer import DropPath, Mlp
from einops import rearrange
import torch.nn.functional as F
from scipy.interpolate import RBFInterpolator
import scipy.io as sio

from Embed import DataEmbedding,  get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid, get_1d_sincos_pos_embed_from_grid_with_resolution
from scipy.spatial import Delaunay, cKDTree
from joblib import Parallel, delayed
from mask_strategy import *
import copy
import time
from scipy import interpolate

def WiFo2_model(args, **kwargs):


    if args.size == 'smallpro':
        model = WiFo2(
            embed_dim=256,
            depth=6,
            decoder_embed_dim=256,
            decoder_depth=4,  # 默认值为4
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=1,
            t_patch_size=args.t_patch_size,
            patch_size=args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb=args.pos_emb,
            no_qkv_bias=bool(args.no_qkv_bias),
            MoE=args.MoE,
            args=args,
            **kwargs,
        )
        return model
    elif args.size == 'smallpro-MoE':
        model = WiFo2(
            embed_dim=256,
            depth=6,
            decoder_embed_dim=256,
            decoder_depth=4,  # 默认值为4
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=4,
            t_patch_size=args.t_patch_size,
            patch_size=args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb=args.pos_emb,
            no_qkv_bias=bool(args.no_qkv_bias),
            MoE=args.MoE,
            args=args,
            **kwargs,
        )
        return model
    elif args.size == 'small-MoE-v16':
        model = WiFo2(
            embed_dim=256,
            depth=6,
            decoder_embed_dim=256,
            decoder_depth=4,  # 默认值为4
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=8,
            t_patch_size=args.t_patch_size,
            patch_size=args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb=args.pos_emb,
            no_qkv_bias=bool(args.no_qkv_bias),
            MoE=args.MoE,
            args=args,
            **kwargs,
        )
        return model
    elif args.size == 'basepro-MoE':
        model = WiFo2(
            embed_dim=512,
            depth=6,
            decoder_embed_dim = 512,
            decoder_depth=4, # 默认值为4
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=4,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            MoE= args.MoE,
            **kwargs,
        )
        return model
    elif args.size == 'basepro-MoE-v16':
        model = WiFo2(
            embed_dim=512,
            depth=6,
            decoder_embed_dim = 512,
            decoder_depth=4, # 默认值为4
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=8,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            MoE= args.MoE,
            **kwargs,
        )
        return model
    elif args.size == 'basepro':
        model = WiFo2(
            embed_dim=512,
            depth=6,
            decoder_embed_dim = 512,
            decoder_depth=4, # 默认值为4
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=4,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            MoE= args.MoE,
            **kwargs,
        )
        return model
    elif args.size == 'basepro1':
        model = WiFo2(
            embed_dim=512,
            depth=6,
            decoder_embed_dim = 512,
            decoder_depth=4, # 默认值为4
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=1,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            MoE= args.MoE,
            **kwargs,
        )
        return model
    elif args.size == 'base-Q':
        model = WiFo2(
            embed_dim=512,
            depth=4,
            decoder_embed_dim=512,
            decoder_depth=2,  # 默认值为4
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=2,
            t_patch_size=args.t_patch_size,
            patch_size=args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb=args.pos_emb,
            no_qkv_bias=bool(args.no_qkv_bias),
            args=args,
            MoE=args.MoE,
            **kwargs,
        )
        return model
    elif args.size == 'largeprov2-MoE':
        model = WiFo2(
            embed_dim=768,
            depth=8,
            decoder_embed_dim=768,
            decoder_depth=6,  # 默认值为4
            num_heads=12,
            decoder_num_heads=12,
            mlp_ratio=4,
            t_patch_size=args.t_patch_size,
            patch_size=args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb=args.pos_emb,
            no_qkv_bias=bool(args.no_qkv_bias),
            args=args,
            MoE=args.MoE,
            **kwargs,
        )
        return model
    elif args.size == 'largeprov2':
        model = WiFo2(
            embed_dim=768,
            depth=8,
            decoder_embed_dim=768,
            decoder_depth=6,  # 默认值为4
            num_heads=12,
            decoder_num_heads=12,
            mlp_ratio=1,
            t_patch_size=args.t_patch_size,
            patch_size=args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb=args.pos_emb,
            no_qkv_bias=bool(args.no_qkv_bias),
            args=args,
            MoE=args.MoE,
            **kwargs,
        )
        return model
    elif args.size == 'littlepro':
        model = WiFo2(
            embed_dim=128,
            depth=6,
            decoder_embed_dim=128,
            decoder_depth=4,  # 默认值为4
            num_heads=4,
            decoder_num_heads=4,
            mlp_ratio=1,
            t_patch_size=args.t_patch_size,
            patch_size=args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb=args.pos_emb,
            no_qkv_bias=bool(args.no_qkv_bias),
            MoE=args.MoE,
            args=args,
            **kwargs,
        )
        return model
    elif args.size == 'littlepro-MoE':
        model = WiFo2(
            embed_dim=128,
            depth=6,
            decoder_embed_dim=128,
            decoder_depth=4,  # 默认值为4
            num_heads=4,
            decoder_num_heads=4,
            mlp_ratio=4,
            t_patch_size=args.t_patch_size,
            patch_size=args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb=args.pos_emb,
            no_qkv_bias=bool(args.no_qkv_bias),
            MoE=args.MoE,
            args=args,
            **kwargs,
        )
        return model
    elif args.size == 'tinypro':
        model = WiFo2(
            embed_dim=64,
            depth=6,
            decoder_embed_dim=64,
            decoder_depth=4,  # 默认值为4
            num_heads=4,
            decoder_num_heads=4,
            mlp_ratio=1,
            t_patch_size=args.t_patch_size,
            patch_size=args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb=args.pos_emb,
            no_qkv_bias=bool(args.no_qkv_bias),
            MoE=args.MoE,
            args=args,
            **kwargs,
        )
        return model
    elif args.size == 'tinypro-MoE':
        model = WiFo2(
            embed_dim=64,
            depth=6,
            decoder_embed_dim=64,
            decoder_depth=4,  # 默认值为4
            num_heads=4,
            decoder_num_heads=4,
            mlp_ratio=4,
            t_patch_size=args.t_patch_size,
            patch_size=args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb=args.pos_emb,
            no_qkv_bias=bool(args.no_qkv_bias),
            MoE=args.MoE,
            args=args,
            **kwargs,
        )
        return model


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim, bias= qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def forward(self, x):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # FA
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        x = attn_out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x

class GateNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GateNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # Gating 网络的前向传播
        logits = F.softmax(self.fc(x),dim=-1)  # 使用 ReLU 激活函数
        return logits  # 返回原始得分


class GateNetworkSparse(nn.Module):
    def __init__(self, input_dim, num_experts, num_active_experts):

        super(GateNetworkSparse, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)  # 计算 gating logits
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts  # 每个 token 选择的专家数

    def forward(self, x):

        N, L, _ = x.shape  # 获取 batch 维度信息
        logits = F.softmax(self.fc(x),dim=-1)  # (N, L, num_experts) 计算 gating 得分


        # 选择 top-k 个专家
        topk_scores, topk_indices = torch.topk(logits, self.num_active_experts, dim=-1)  # (N, L, num_active_experts)

        # 构造 gate 权重矩阵，初始化为 0
        gate_weights = torch.zeros(N, L, self.num_experts, device=topk_indices.device)  # (N, L, num_experts)
        gate_weights.scatter_(-1, topk_indices, 1)  # 只激活 top-k 专家 (N, L, num_experts)

        # 计算专家的 token 分配比例 D (每个专家被选中的频率)
        D = gate_weights.sum(dim=(0, 1)) / (N * L * self.num_active_experts)  # (num_experts,)

        # 计算专家的 gating 概率 P (每个专家的平均分配概率)
        P = logits.sum(dim=(0, 1)) / (N * L)  # (num_experts,)

        # 计算负载均衡损失
        load_balance_loss = self.num_experts* (D * P).sum()  # 标量

        return topk_scores, topk_indices, load_balance_loss, D

class MLPWithExperts(nn.Module):
    """
    MLP layer with multiple experts, weighted by the gate network.
    """
    def __init__(self, in_features, hidden_features, num_experts, num_active_experts, drop):
        super().__init__()
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        expert_dim = hidden_features // num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, in_features),
                nn.Dropout(drop)
            )
            for _ in range(num_experts)
        ])

    def forward(self, hidden_states, topk_scores, topk_indices):
        orig_shape = hidden_states.shape  # 保存原始形状 [N, L, C]
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])  # 展平为 [NL, C]
        flat_topk_idx = topk_indices.view(-1)  # 展平专家索引为 [bsz * seq_len * top_k]
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_active_experts, dim=0)  # 重复输入以匹配专家数量
            y = torch.empty_like(hidden_states,dtype= hidden_states.dtype)  # 初始化输出张量[NLK,C]
            for i, expert in enumerate(self.experts):  # 遍历每个专家
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i]).to(y.dtype)  # 将输入分配给对应专家
            y = (y.view(*topk_scores.shape, -1) * topk_scores.unsqueeze(-1)).sum(dim=2)  # 加权求和?此处维度似乎不正确
            y = y.view(*orig_shape)  # 恢复原始形状
            # 推理模式
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_scores.view(-1, 1)).view(*orig_shape)  # 调用推理函数
        return y

    @torch.no_grad()  # 推理模式下禁用梯度计算
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # x: [NL,C]
        # flat_expert_indices: [NLk]
        # flat_expert_weights: [NLk,1]
        expert_cache = torch.zeros_like(x)  # 初始化输出缓存 [NL,C]
        idxs = flat_expert_indices.argsort()  # 对专家索引排序 [NLk]
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)  # 计算每个专家的 token 数 [K]
        token_idxs = idxs // self.num_active_experts  # 计算 token 对应的索引 [NLk]
        for i, end_idx in enumerate(tokens_per_expert):  # 遍历每个专家
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]  # 计算起始索引
            if start_idx == end_idx:  # 如果没有 token 分配给该专家，跳过
                continue
            expert = self.experts[i]  # 获取当前专家
            exp_token_idx = token_idxs[start_idx:end_idx]  # 获取分配给该专家的 token 索引
            expert_tokens = x[exp_token_idx]  # 获取对应的输入
            expert_out = expert(expert_tokens)  # 计算专家输出
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])  # 乘以权重
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out.to(expert_cache.dtype),
                                         reduce='sum')  # 将结果累加到缓存
        return expert_cache


class Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MoEBlock(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.num_experts = 8
        self.gate = nn.ModuleList([GateNetwork(dim, self.num_experts) for _ in range(4)])
        self.mlp = MLPWithExperts(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            num_experts=self.num_experts,
            drop=drop,
        )

    def forward(self, x, task_id):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # Gate network for MoE
        gate_weights = self.gate[task_id](x)  # (N, L, num_experts)

        # MLP block with weighted experts
        x = x + self.drop_path(self.mlp(self.norm2(x), gate_weights))
        return x

class SMoEBlock(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.num_experts = 8
        self.num_active_experts = 2
        self.gate = nn.ModuleList([GateNetworkSparse(dim, self.num_experts, self.num_active_experts) for _ in range(4)])
        self.mlp = MLPWithExperts(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            num_experts=self.num_experts,
            num_active_experts = self.num_active_experts,
            drop=drop,
        )

    def forward(self, x, task_id):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # Gate network for MoE
        topk_scores, topk_indices, load_balance_loss, Probb = self.gate[task_id](x)  # (N, L, num_experts)

        # MLP block with weighted experts
        x = x + self.drop_path(self.mlp(self.norm2(x), topk_scores, topk_indices))
        return x, load_balance_loss, Probb


class WiFo2(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, patch_size=1, in_chans=1,
                 embed_dim=512, decoder_embed_dim=512, depth=12, decoder_depth=8, num_heads=8,  decoder_num_heads=4,
                 mlp_ratio=2, norm_layer=nn.LayerNorm, t_patch_size=1,
                 no_qkv_bias=False, pos_emb = 'trivial', MoE = True, args=None,):
        super().__init__()

        self.args = args

        self.pos_emb = pos_emb

        # mask
        self.t_patch_size = t_patch_size
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.MoE = MoE

        self.embed_dim = embed_dim
        self.Embedding_encoder = nn.Linear(2*args.t_patch_size*args.patch_size**2, embed_dim)
        self.Embedding_encoder_CE = nn.Linear(2 * args.t_patch_size * args.patch_size ** 2, embed_dim)
        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 64, embed_dim)
        )
        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, 6, embed_dim)
        )
        self.decoder_pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 64, decoder_embed_dim)
        )
        self.decoder_pos_embed_temporal = nn.Parameter(
            torch.zeros(1, 6,  decoder_embed_dim)
        )



        if self.MoE:
            self.MoEblocks = nn.ModuleList(
                [
                    SMoEBlock(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        qkv_bias=not no_qkv_bias,
                        qk_scale=None,
                        norm_layer=norm_layer,
                    )
                    for i in range(depth)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        qkv_bias=not no_qkv_bias,
                        qk_scale=None,
                        norm_layer=norm_layer,
                    )
                    for i in range(depth)
                ]
            )

        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.no_qkv_bias = no_qkv_bias
        self.norm_layer = norm_layer


        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias= not self.args.no_qkv_bias)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if self.MoE:
            self.MoE_decoder_blocks = nn.ModuleList(
                [
                    SMoEBlock(
                        decoder_embed_dim,
                        decoder_num_heads,
                        mlp_ratio,
                        qkv_bias=not no_qkv_bias,
                        qk_scale=None,
                        norm_layer=norm_layer,
                    )
                    for i in range(decoder_depth)
                ]
            )
        else:
            self.decoder_blocks = nn.ModuleList(
                [
                    Block(
                        decoder_embed_dim,
                        decoder_num_heads,
                        mlp_ratio,
                        qkv_bias=not no_qkv_bias,
                        qk_scale=None,
                        norm_layer=norm_layer,
                    )
                    for i in range(decoder_depth)
                ]
            )
        self.decoder_depth = decoder_depth
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred_dual = nn.Sequential(
            nn.Linear(decoder_embed_dim, self.t_patch_size * patch_size ** 2 * 2, bias=True),
            nn.ReLU(),
            nn.Linear(self.t_patch_size * patch_size ** 2 * 2, self.t_patch_size * patch_size ** 2 * 2, bias=True)
        )

        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.t_patch_size * patch_size**2 * 2,
            bias=True,
        )

        self.decoder_pred_CE = nn.Linear(
            decoder_embed_dim,
            self.t_patch_size * patch_size ** 2 * 2,
            bias=True,
        )

        self.initialize_weights_trivial()

        self.task_to_id = {
            'random': 0,
            'temporal': 1,
            'fre': 2,
            'CE': 3,
        }

        print("model initialized!")


    def get_weights_sincos(self, num_t_patch, num_patch_1, num_patch_2):
        # initialize (and freeze) pos_embed by sin-cos embedding
        # 空域是2d sincos pos embed
        # 时域是1d sincos pos embed

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed_spatial.shape[-1],
            grid_size1 = num_patch_1,
            grid_size2 = num_patch_2
        )

        pos_embed_spatial = nn.Parameter(
                torch.zeros(1, num_patch_1 * num_patch_2, self.embed_dim)
            )
        pos_embed_temporal = nn.Parameter(
            torch.zeros(1, num_t_patch, self.embed_dim)
        )

        pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_temporal_emb = get_1d_sincos_pos_embed_from_grid(pos_embed_temporal.shape[-1], np.arange(num_t_patch, dtype=np.float32))

        pos_embed_temporal.data.copy_(torch.from_numpy(pos_temporal_emb).float().unsqueeze(0))

        pos_embed_spatial.requires_grad = False
        pos_embed_temporal.requires_grad = False

        return pos_embed_spatial, pos_embed_temporal, copy.deepcopy(pos_embed_spatial), copy.deepcopy(pos_embed_temporal)

    def initialize_weights_trivial(self):
        std_pre = 0.02 # 原本为0.02
        torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=std_pre)

        torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=std_pre)
        torch.nn.init.normal_(self.mask_token, std=std_pre)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)





    def patchify(self, imgs):
        """
        imgs: (N, 1, T, H, W)
        x: (N, L, patch_size**2 *1)
        # 输入为包含实部和虚部的imgs
        """
        N, _, T, H, W = imgs.shape  # N, 2, T, H, W
        p = self.args.patch_size
        u = self.args.t_patch_size
        assert H % p == 0 and W % p == 0 and T % u == 0
        h = H // p
        w = W // p
        t = T // u
        x = imgs.reshape(shape=(N, 2, t, u, h, p, w, p))  # 第2维代表实部/虚部
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2, 2))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        # x = x[:,:,:,0] + 1j*x[:,:,:,1]
        x = x.permute(0, 1, 3, 2)
        return x

    def unpatchify(self, imgs):
        """
        imgs: complex tensor, shape [N, L, Len]
        return: [N, 2, T, H, W]
        """
        N, T, H, W, p, u, t, h, w = self.patch_info

        # 现在最后一维顺序是 [real..., imag...]
        imgs = torch.cat([imgs.real, imgs.imag], dim=-1)  # [N, L, 2*Len]

        # 所以这里必须按 (o u p q) 还原，而不是 (u p q o)
        imgs = rearrange(
            imgs,
            'b (t h w) (o u p q) -> b o (t u) (h p) (w q)',
            t=t, h=h, w=w, o=2, u=u, p=p, q=p
        )
        return imgs

    def pos_embed_enc(self, ids_keep, batch, input_size):

        if self.pos_emb == 'trivial':  # [1,256,384] 后两维是token个数和emb_dim
            # pos_spatial = self.pos_embed_fre[:,:input_size[2]].repeat(
            #     1, input_size[1],1
            # ) + torch.repeat_interleave(
            #     self.pos_embed_antenna[:,:input_size[1]],
            #     input_size[2],
            #     dim=1
            # )  # [1,64,384]
            pos_embed = self.pos_embed_spatial[:,:input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal[:,:input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )

        elif self.pos_emb == 'SinCos':
            pos_embed_spatial, pos_embed_temporal, _, _ = self.get_weights_sincos(input_size[0], input_size[1], input_size[2])

            pos_embed = pos_embed_spatial[:,:input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                pos_embed_temporal[:,:input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )  # [1,256,256]

        pos_embed = pos_embed.to(ids_keep.device)

        pos_embed = pos_embed.expand(batch, -1, -1)

        pos_embed_sort = torch.gather(
            pos_embed,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
        )
        return pos_embed_sort

    def split_even(self, ED):
        assert ED % 2 == 0, "ED 必须是偶数"
        # b: 每份的“基准”偶数，k: 需要在前 k 份各 +2
        b = (ED // 3 // 2) * 2
        k = (ED - 3 * b) // 2
        return [b + 2 * (i < k) for i in range(3)]

    def pos_embed_enc_3d(self, ids_keep, batch, input_size, scale):

        T, H, W = input_size
        t = torch.arange(T)
        h = torch.arange(H)
        w = torch.arange(W)

        tt, hh, ww = torch.meshgrid(t, h, w, indexing='ij')
        ED1, ED2, ED3 = self.split_even(self.embed_dim)

        emb_t = get_1d_sincos_pos_embed_from_grid_with_resolution(ED1, tt.flatten(),scale[0])
        emb_h = get_1d_sincos_pos_embed_from_grid_with_resolution(ED2, hh.flatten(),scale[1])
        emb_w = get_1d_sincos_pos_embed_from_grid_with_resolution(ED3, ww.flatten(),scale[2])

        pos_embed = np.concatenate([emb_t, emb_h, emb_w], axis=1)
        pos_embed = torch.from_numpy(pos_embed).float()
        pos_embed.requires_grad = False

        pos_embed = pos_embed.to(ids_keep.device)  # [1,256,256]
        pos_embed = pos_embed.expand(batch, -1, -1)

        pos_embed_sort = torch.gather(
            pos_embed,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
        )
        return pos_embed_sort



    def pos_embed_dec(self, ids_keep, batch, input_size):

        if self.pos_emb == 'trivial':
            decoder_pos_embed = self.decoder_pos_embed_spatial[:,:input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal[:,:input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )

        elif self.pos_emb == 'SinCos':
            _, _, decoder_pos_embed_spatial, decoder_pos_embed_temporal  = self.get_weights_sincos(input_size[0], input_size[1], input_size[2])

            decoder_pos_embed = decoder_pos_embed_spatial[:,:input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                decoder_pos_embed_temporal[:,:input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )

        decoder_pos_embed = decoder_pos_embed.to(ids_keep.device)

        decoder_pos_embed = decoder_pos_embed.expand(batch, -1, -1)

        return decoder_pos_embed

    def pos_embed_dec_3d(self, ids_keep, batch, input_size, scale):

        T, H, W = input_size
        t = torch.arange(T)
        h = torch.arange(H)
        w = torch.arange(W)

        tt, hh, ww = torch.meshgrid(t, h, w, indexing='ij')

        # ED = self.embed_dim
        # ED1 = round(ED * 1.0 / 3)
        # ED2 = round(ED * 1.0 / 3)
        # ED3 = ED - ED1 - ED2
        ED1, ED2, ED3 = self.split_even(self.embed_dim)

        emb_t = get_1d_sincos_pos_embed_from_grid_with_resolution(ED1, tt.flatten(),scale[0])
        emb_h = get_1d_sincos_pos_embed_from_grid_with_resolution(ED2, hh.flatten(),scale[1])
        emb_w = get_1d_sincos_pos_embed_from_grid_with_resolution(ED3, ww.flatten(),scale[2])

        pos_embed = np.concatenate([emb_t, emb_h, emb_w], axis=1)
        decoder_pos_embed = torch.from_numpy(pos_embed).float()
        decoder_pos_embed.requires_grad = False

        decoder_pos_embed = decoder_pos_embed.to(ids_keep.device)

        decoder_pos_embed = decoder_pos_embed.expand(batch, -1, -1)

        return decoder_pos_embed

    def interpolate_scattered_3d_linear_extrap_with_fallback(self,
            values,  # [B, M] 或 [M]
            x_in, y_in, z_in,  # [B,M]
            x_out, y_out, z_out,  # 各 [Meshgrid]
            n_jobs=1,
            qhull_options="QJ"  # 缺省用 QJ 抖动以缓解退化/共面
    ): # 输出大小为[B, M]
        values = np.asarray(values)
        if values.ndim == 1:
            values = values[None, :]  # [1, M]
        if values.ndim != 2:
            raise ValueError("`values` 必须是 [B, M] 或 [M].")
        B, M = values.shape

        # x_in = np.asarray(x_in).ravel()
        # y_in = np.asarray(y_in).ravel()
        # z_in = np.asarray(z_in).ravel()
        if not (len(x_in) == len(y_in) == len(z_in) == M):
            raise ValueError("x_in/y_in/z_in 必须与 M 一致，且长度相等。")

        x_out = np.asarray(x_out).ravel()
        y_out = np.asarray(y_out).ravel()
        z_out = np.asarray(z_out).ravel()
        M2 = len(x_out)
        if not (len(y_out) == len(z_out) == M2):
            raise ValueError("x_out/y_out/z_out 必须与 M2 一致，且长度相等。")

        pts_in = np.column_stack([x_in, y_in, z_in])  # [M, 3]
        pts_q = np.column_stack([x_out, y_out, z_out])  # [M2, 3]

        # 最近邻查询树（用于外推回退 & 兜底）
        nn_tree = cKDTree(pts_in)

        # --- 先尝试构建 Delaunay（三角剖分） ---
        try:
            tri = Delaunay(pts_in, qhull_options=qhull_options)
            # 为外推选择“最近顶点所属的单纯形”
            vert_tree = cKDTree(pts_in)

            simplices = tri.find_simplex(pts_q)  # [M2]
            inside_mask = simplices >= 0
            outside_mask = ~inside_mask

            # 预备内部点的仿射->重心映射
            if np.any(inside_mask):
                s_inside = simplices[inside_mask]
                T_inside = tri.transform[s_inside]  # [Ni, 4, 3]
                A_in = T_inside[:, :3, :]  # [Ni, 3, 3]
                b_in = T_inside[:, 3, :]  # [Ni, 3]
                vtx_in = tri.simplices[s_inside]  # [Ni, 4]

            # 预备外部点：用最近顶点的一个临近单纯形做外推
            if np.any(outside_mask):
                _, nearest_v_idx = vert_tree.query(pts_q[outside_mask], k=1)  # [No]
                s_out = tri.vertex_to_simplex[nearest_v_idx]  # [No]
                # vertex_to_simplex 可能为 -1（没有内侧单纯形），这里兜底指向 0 号
                s_out = np.where(s_out < 0, 0, s_out)
                T_out = tri.transform[s_out]  # [No, 4, 3]
                A_out = T_out[:, :3, :]
                b_out = T_out[:, 3, :]
                vtx_out = tri.simplices[s_out]  # [No, 4]

            def _process_one_batch(vals_1d):
                # vals_1d: [M]
                out = np.empty(M2, dtype=float)

                # 内部：线性（重心）插值
                if np.any(inside_mask):
                    delta = pts_q[inside_mask] - b_in  # [Ni, 3]
                    bary3 = np.einsum('nij,nj->ni', A_in, delta)  # [Ni, 3]
                    bary4 = np.hstack([bary3, 1 - bary3.sum(axis=1, keepdims=True)])  # [Ni, 4]
                    vals4 = vals_1d[vtx_in]  # [Ni, 4]
                    out[inside_mask] = np.einsum('ni,ni->n', bary4, vals4)

                # 外部：线性外推 + 回退最近邻
                if np.any(outside_mask):
                    delta_o = pts_q[outside_mask] - b_out
                    bary3_o = np.einsum('nij,nj->ni', A_out, delta_o)
                    bary4_o = np.hstack([bary3_o, 1 - bary3_o.sum(axis=1, keepdims=True)])  # [No, 4]
                    vals4_o = vals_1d[vtx_out]  # [No, 4]
                    out_o = np.einsum('ni,ni->n', bary4_o, vals4_o)

                    # 非数回退：最近邻（用已算过的最近顶点）
                    bad = ~np.isfinite(out_o)
                    if np.any(bad):
                        out_o[bad] = vals_1d[nearest_v_idx[bad]]

                    out[outside_mask] = out_o

                return out

            # 并行按 batch 处理
            outs = Parallel(n_jobs=n_jobs)(
                delayed(_process_one_batch)(values[b]) for b in range(B)
            )
            out_arr = np.stack(outs, axis=0)  # [B, M2]
            return out_arr

        except Exception:
            # Delaunay 构建失败（点数少/共面/退化等），直接兜底为最近邻
            _, nn_idx = nn_tree.query(pts_q, k=1)  # [M2]

            def _nn_batch(vals_1d):
                return vals_1d[nn_idx]

            outs = Parallel(n_jobs=n_jobs)(
                delayed(_nn_batch)(values[b]) for b in range(B)
            )
            return np.stack(outs, axis=0)




    def CE_interpolation_new(self, x, mask_ratio, device_id):
        I_1, I_2, I_3 = mask_ratio
        N, C, T, H, W = x.shape

        # 1) 采样坐标（索引）
        t_idx = np.arange(0, T, I_1, dtype=int)
        h_idx = np.arange(0, H, I_2, dtype=int)
        w_idx = np.arange(0, W, I_3, dtype=int)

        # 2) 抽取样本值 —— 用切片步长，避免高级索引广播问题
        x_np = x.detach().contiguous().cpu().numpy()
        inputs_grid = x_np[:, :, ::I_1, ::I_2, ::I_3]  # (N,C,|t|,|h|,|w|)
        inputs = inputs_grid.reshape(N*C, -1)  # (N,C,K)

        # 3) 已知点坐标（与 reshape 顺序一致）
        Tg, Hg, Wg = np.meshgrid(t_idx, h_idx, w_idx, indexing="ij")
        x1 = Tg.ravel().astype(float)
        y1 = Hg.ravel().astype(float)
        z1 = Wg.ravel().astype(float)

        # 4) 输出网格（全体素）
        x_out = np.arange(T, dtype=float)
        y_out = np.arange(H, dtype=float)
        z_out = np.arange(W, dtype=float)
        x_out, y_out, z_out = np.meshgrid(x_out, y_out, z_out, indexing='ij')

        out_np = self.interpolate_scattered_3d_linear_extrap_with_fallback(
            inputs, x1, y1, z1,
            x_out, y_out, z_out,
            n_jobs=4
        )
        out_np = out_np.reshape(N,C,T,H,W)
        # 6) 回到 torch
        output = torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype)

        return output

    def forward_encoder(self, x, mask_ratio, mask_strategy, seed=None, data=None, scale=None):
        # embed patches
        N, _, T, H, W = x.shape

        T = T // self.args.t_patch_size  # patch_size之后的时间长度
        H = H // self.patch_size
        W = W // self.patch_size

        if mask_strategy == 'CE':
            x = self.CE_interpolation_new(x,mask_ratio,self.args.device_id)

        x = self.patchify(x)  # [N, L, P, 2]
        x = x.reshape(N, T*H*W, -1)

        # x = self.Embedding_encoder(x)

        if mask_strategy == 'CE':
            x = self.Embedding_encoder_CE(x)
        else:
            x = self.Embedding_encoder(x)


        _, L, C = x.shape

        if mask_strategy == 'random':
            x, mask, ids_restore, ids_keep = random_masking(x, mask_ratio)

        elif mask_strategy == 'tube':
            x, mask, ids_restore, ids_keep = tube_masking(x, mask_ratio, T=T)

        elif mask_strategy == 'block':
            x, mask, ids_restore, ids_keep = tube_block_masking(x, mask_ratio, T=T)

        elif mask_strategy == 'temporal':
            x, mask, ids_restore, ids_keep = causal_masking(x, mask_ratio, T=T)

        elif mask_strategy == 'fre':
            x, mask, ids_restore, ids_keep = fre_masking(x, mask_ratio, T=T, H=H, W=W)

        elif mask_strategy == 'antenna':
            x, mask, ids_restore, ids_keep = antenna_masking(x, mask_ratio, T=T, H=H, W=W)


        elif mask_strategy == 'CE':
            mask = torch.ones([N, L], device=x.device)
            ids_keep = torch.arange(L).unsqueeze(dim=0).repeat(N,1).to(x.device)  # [batch_size, L]
            ids_restore = ids_keep




        input_size = (T, H, W)
        if self.pos_emb == 'SinCos' or self.pos_emb == 'trivial':
            pos_embed_sort = self.pos_embed_enc(ids_keep, N, input_size)  # 位置编码
            assert x.shape == pos_embed_sort.shape
            x_attn = x + pos_embed_sort
        elif self.pos_emb == 'SinCos_3D':
            pos_embed_sort = self.pos_embed_enc_3d(ids_keep, N, input_size, scale=scale)
            assert x.shape == pos_embed_sort.shape
            x_attn = x + pos_embed_sort
        elif self.pos_emb == 'None':
            x_attn = x

        task_id = torch.tensor(self.task_to_id.get(mask_strategy), device=x.device)
        LB_loss_enc = 0
        if self.MoE:
            for index, blk in enumerate(self.MoEblocks):
                x_attn, load_balance_loss, Probb = blk(x_attn,task_id)
                LB_loss_enc = LB_loss_enc + load_balance_loss
        else:
            for index, blk in enumerate(self.blocks):
                x_attn = blk(x_attn)

        x_attn = self.norm(x_attn)

        return x_attn, mask, ids_restore, input_size, LB_loss_enc

    def forward_decoder(self, x, ids_restore, mask_strategy, input_size=None, data=None, scale=None, mask=None):
        N = x.shape[0]
        T, H, W = input_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        if mask_strategy == 'random':
            x = random_restore(x, ids_restore, N, T,  H, W, C, self.mask_token)

        elif mask_strategy in ['tube','block']:
            x = tube_restore(x, ids_restore, N, T, H,  W,  C, self.mask_token)

        elif mask_strategy == 'temporal':
            x = causal_restore(x, ids_restore, N, T, H,  W, C, self.mask_token)

        elif mask_strategy == 'antenna':
            x = antenna_restore(x, ids_restore, N, T, H,  W, C, self.mask_token)

        elif mask_strategy == 'fre':
            x = fre_restore(x, ids_restore, N, T, H,  W, C, self.mask_token)



        if self.pos_emb == 'SinCos' or self.pos_emb == 'trivial':
            decoder_pos_embed = self.pos_embed_dec(ids_restore, N, input_size)  # 位置编码
            assert x.shape == decoder_pos_embed.shape
            x_attn = x + decoder_pos_embed
        elif self.pos_emb == 'SinCos_3D':
            decoder_pos_embed = self.pos_embed_dec_3d(ids_restore, N, input_size, scale=scale)
            assert x.shape == decoder_pos_embed.shape
            x_attn = x + decoder_pos_embed
        elif self.pos_emb == 'None':
            x_attn = x
        task_id = torch.tensor(self.task_to_id.get(mask_strategy), device=x.device)
        LB_loss_dec = 0
        if self.MoE:
            for index, blk in enumerate(self.MoE_decoder_blocks):
                x_attn, load_balance_loss, Probb = blk(x_attn,task_id)
                # self.append_line(index+8, Probb)
                LB_loss_dec = LB_loss_dec + load_balance_loss
        else:
            for index, blk in enumerate(self.decoder_blocks):
                x_attn = blk(x_attn)

        x_attn = self.decoder_norm(x_attn)


        return x_attn, LB_loss_dec


    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 2, T, H, W] 1000*1*12*16*96
        pred: [N, t*h*w, u*p*p*1] 1000*576*32(2*4*4)
        mask: [N*t, h*w], 0 is keep, 1 is remove, 1000*576
        """

        target = self.patchify(imgs)


        assert pred.shape == target.shape  # 大小都是[B, L, 2, F]

        # loss = torch.abs(pred - target) ** 2
        real_diff = pred[:, :, 0, :] - target[:, :, 0, :]
        imag_diff = pred[:, :, 1, :] - target[:, :, 1, :]
        loss = real_diff ** 2 + imag_diff ** 2  # [B, L, F]
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)
        loss1 = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss2 = (loss * (1-mask)).sum() / (1-mask).sum()  #
        return loss1, loss2, target  # 原本是loss1


    def forward(self, imgs, mask_ratio=0.5, mask_strategy='random',seed=None, data='none'):
        # data的含义就是数据集的名字
        # imgs, imgs_mark, imgs_period = imgs

        # imgs_period = imgs_period[:,:,self.args.his_len:]
        # imgs = imgs[0]  # 16*2*16*16*64
        # imgs = torch.stack(imgs)

        # 考虑固定信噪比
        snr_db = 20
        # 考虑对数均匀信噪比
        # snr_db = torch.empty(1, device=imgs.device).uniform_(10.0, 25.0)  # 改这里就能换范围
        noise_power = torch.mean(imgs ** 2) * 10 ** (-snr_db / 10)

        noise = torch.randn_like(imgs) * torch.sqrt(noise_power)
        imgs_n = imgs + noise

        scale = [1,1,1]
        start_time = time.time()
        T, H, W = imgs_n.shape[2:] # imgs的维度为256*1*12*10*20(T*H*W)
        latent, mask, ids_restore, input_size, LB_loss_enc = self.forward_encoder(imgs_n, mask_ratio, mask_strategy, seed=seed, data=data, scale=scale)

        pred, LB_loss_dec = self.forward_decoder(latent, ids_restore, mask_strategy, input_size = input_size, data=data,scale=scale,mask=mask)  # [N, L, p*p*1]

        # predictor projection
        pred = self.decoder_pred(pred)  # 单头输出（采用）


        Len = self.t_patch_size * self.patch_size ** 2

        # pred_complex = pred[:, :, :Len] + 1j * pred[:, :, Len:]
        pred_complex = torch.stack([pred[:, :, :Len], pred[:, :, Len:]], dim=2) # 维度大小为[B,L,2,Len]

        # pred_oringin = self.unpatchify(pred[:, :, :Len] + 1j * pred[:, :, Len:])

        # 此处应当放end time
        end_time = time.time()
        original_time = end_time - start_time


        loss1, loss2, target = self.forward_loss(imgs, pred_complex, mask)  # target的大小为[B, L, 2, Len]

        pred0_all = self.unpatchify(pred[:, :, :Len] + 1j * pred[:, :, Len:])


        # 保存成 mat 文件
        # sio.savemat('x_pred.mat', {'x_pred': pred0_all.detach().cpu().numpy()})
        # sio.savemat('x_gt.mat', {'x_gt': imgs.detach().cpu().numpy()})
        # sio.savemat('x_n.mat', {'x_n': imgs_n.detach().cpu().numpy()})



        return loss1+0.03*(LB_loss_dec+LB_loss_enc)/(self.depth+self.decoder_depth), loss2, pred_complex, target, mask
