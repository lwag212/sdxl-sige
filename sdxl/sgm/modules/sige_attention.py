# from typing import Optional
# import logging
# import math

# import torch
# from einops import rearrange, repeat
# from torch import nn

# from sige.nn import Gather, Scatter, SIGEConv2d, SIGEModule
# from .attention import CrossAttention, MemoryEfficientCrossAttention, default, exists, FeedForward, Normalize, SpatialTransformer, zero_module
# from .diffusionmodules.util import my_group_norm
# from packaging import version

# logpy = logging.getLogger(__name__)
# if version.parse(torch.__version__) >= version.parse("2.0.0"):
#     SDP_IS_AVAILABLE = True
#     from torch.backends.cuda import SDPBackend, sdp_kernel

#     BACKEND_MAP = {
#         SDPBackend.MATH: {
#             "enable_math": True,
#             "enable_flash": False,
#             "enable_mem_efficient": False,
#         },
#         SDPBackend.FLASH_ATTENTION: {
#             "enable_math": False,
#             "enable_flash": True,
#             "enable_mem_efficient": False,
#         },
#         SDPBackend.EFFICIENT_ATTENTION: {
#             "enable_math": False,
#             "enable_flash": False,
#             "enable_mem_efficient": True,
#         },
#         None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
#     }
# else:
#     from contextlib import nullcontext

#     SDP_IS_AVAILABLE = False
#     sdp_kernel = nullcontext
#     BACKEND_MAP = {}
#     logpy.warn(
#         f"No SDP backend available, likely because you are running in pytorch "
#         f"versions < 2.0. In fact, you are using PyTorch {torch.__version__}. "
#         f"You might want to consider upgrading."
#     )

# try:
#     import xformers
#     import xformers.ops

#     XFORMERS_IS_AVAILABLE = True
# except:
#     XFORMERS_IS_AVAILABLE = False
#     logpy.warn("no module 'xformers'. Processing without...")


# class SIGECrossAttention(SIGEModule):
#     def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
#         super().__init__()
#         inner_dim = dim_head * heads
#         context_dim = default(context_dim, query_dim)

#         self.scale = dim_head ** (-0.5)
#         self.heads = heads

#         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

#         self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

#         self.cached_k = None
#         self.cached_v = None

#     def forward(self, x, context=None, mask=None):
#         h = self.heads

#         q = self.to_q(x)
#         context = default(context, x)
#         if self.mode == "full":
#             k = self.to_k(context)
#             v = self.to_v(context)
#             self.cached_k = k
#             self.cached_v = v
#         else:
#             k = self.cached_k
#             v = self.cached_v
#         q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

#         # sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
#         sim = torch.bmm(q, k.permute(0, 2, 1)) * self.scale

#         if exists(mask):
#             mask = rearrange(mask, "b ... -> b (...)")
#             max_neg_value = -torch.finfo(sim.dtype).max
#             mask = repeat(mask, "b j -> (b h) () j", h=h)
#             sim.masked_fill_(~mask, max_neg_value)

#         # attention, what we cannot get enough of
#         attn = sim.softmax(dim=-1)

#         # out = einsum("b i j, b j d -> b i d", attn, v)
#         out = torch.bmm(attn, v)

#         out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
#         return self.to_out(out)


# class SIGEMemoryEfficientCrossAttention(SIGEModule):
#     # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
#     def __init__(
#         self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
#     ):
#         super().__init__()
#         logpy.debug(
#             f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, "
#             f"context_dim is {context_dim} and using {heads} heads with a "
#             f"dimension of {dim_head}."
#         )
#         inner_dim = dim_head * heads
#         context_dim = default(context_dim, query_dim)

#         self.heads = heads
#         self.dim_head = dim_head

#         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
#         )
#         self.attention_op: Optional[Any] = None

#     def forward(
#         self,
#         x,
#         context=None,
#         mask=None,
#         additional_tokens=None,
#         n_times_crossframe_attn_in_self=0,
#     ):
#         if additional_tokens is not None:
#             # get the number of masked tokens at the beginning of the output sequence
#             n_tokens_to_mask = additional_tokens.shape[1]
#             # add additional token
#             x = torch.cat([additional_tokens, x], dim=1)
#         h = self.heads

#         q = self.to_q(x)
#         context = default(context, x)
#         if self.mode == "full":
#             k = self.to_k(context)
#             v = self.to_v(context)
#             self.cached_k = k
#             self.cached_v = v
#         else:
#             k = self.cached_k
#             v = self.cached_v
#         b, _, _ = q.shape
#         q, k, v = map(
#             lambda t: t.unsqueeze(3)
#             .reshape(b, t.shape[1], self.heads, self.dim_head)
#             .permute(0, 2, 1, 3)
#             .reshape(b * self.heads, t.shape[1], self.dim_head)
#             .contiguous(),
#             (q, k, v),
#         )

#         # actually compute the attention, what we cannot get enough of
#         if version.parse(xformers.__version__) >= version.parse("0.0.21"):
#             # NOTE: workaround for
#             # https://github.com/facebookresearch/xformers/issues/845
#             max_bs = 32768
#             N = q.shape[0]
#             n_batches = math.ceil(N / max_bs)
#             out = list()
#             for i_batch in range(n_batches):
#                 batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
#                 out.append(
#                     xformers.ops.memory_efficient_attention(
#                         q[batch],
#                         k[batch],
#                         v[batch],
#                         attn_bias=None,
#                         op=self.attention_op,
#                     )
#                 )
#             out = torch.cat(out, 0)
#         else:
#             out = xformers.ops.memory_efficient_attention(
#                 q, k, v, attn_bias=None, op=self.attention_op
#             )

#         # TODO: Use this directly in the attention operation, as a bias
#         if exists(mask):
#             raise NotImplementedError
#         out = (
#             out.unsqueeze(0)
#             .reshape(b, self.heads, out.shape[1], self.dim_head)
#             .permute(0, 2, 1, 3)
#             .reshape(b, out.shape[1], self.heads * self.dim_head)
#         )
#         if additional_tokens is not None:
#             # remove additional token
#             out = out[:, n_tokens_to_mask:]
#         return self.to_out(out)


# class SIGEBasicTransformerBlock(SIGEModule):
#     SIGE_ATTENTION_MODES = {
#         "softmax": SIGECrossAttention,  # vanilla attention
#         "softmax-xformers": SIGEMemoryEfficientCrossAttention,  # ampere
#     }
#     ATTENTION_MODES = {
#         "softmax": CrossAttention,  # vanilla attention
#         "softmax-xformers": MemoryEfficientCrossAttention,  # ampere
#     }
#     def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True, use_checkpoint=True, disable_self_attn=False, attn_mode="softmax", sdp_backend=None,):
#         super().__init__()
#         assert attn_mode in self.ATTENTION_MODES
#         if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
#             logpy.warn(
#                 f"Attention mode '{attn_mode}' is not available. Falling "
#                 f"back to native attention. This is not a problem in "
#                 f"Pytorch >= 2.0. FYI, you are running with PyTorch "
#                 f"version {torch.__version__}."
#             )
#             attn_mode = "softmax"
#         elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
#             logpy.warn(
#                 "We do not support vanilla attention anymore, as it is too "
#                 "expensive. Sorry."
#             )
#             if not XFORMERS_IS_AVAILABLE:
#                 assert (
#                     False
#                 ), "Please install xformers via e.g. 'pip install xformers==0.0.16'"
#             else:
#                 logpy.info("Falling back to xformers efficient attention.")
#                 attn_mode = "softmax-xformers"
#         attn_cls = self.ATTENTION_MODES[attn_mode]
#         sige_attn_cls = self.SIGE_ATTENTION_MODES[attn_mode]
#         self.attn1 = attn_cls(
#             query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
#         )  # is a self-attention
#         self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
#         self.attn2 = sige_attn_cls(
#             query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
#         )  # is self-attn if context is none
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.norm3 = nn.LayerNorm(dim)
#         self.use_checkpoint = use_checkpoint

#     def forward(self, x, full_x=None, context=None):
#         x = self.attn1(self.norm1(x), context=None if full_x is None else self.norm1(full_x)) + x
#         x = self.attn2(self.norm2(x), context=context) + x
#         x = self.ff(self.norm3(x)) + x
#         return x


# class SIGESpatialTransformer(SIGEModule, SpatialTransformer):
#     def __init__(
#         self,
#         in_channels,
#         n_heads,
#         d_head,
#         depth=1,
#         dropout=0.0,
#         context_dim=None,
#         use_checkpoint=True,
#         disable_self_attn=False,
#         use_linear=False,
#         attn_type="softmax",
#         block_size: Optional[int] = 4,
#         sdp_backend=None,
#     ):
#         super(SpatialTransformer, self).__init__()
#         SIGEModule.__init__(self, call_super=False)
#         self.in_channels = in_channels
#         inner_dim = n_heads * d_head
#         self.norm = Normalize(in_channels)

#         support_sparse = block_size is not None and not use_linear
#         Conv2d = SIGEConv2d if support_sparse else nn.Conv2d

#         self.support_sparse = support_sparse

#         if not use_linear:
#             self.proj_in = Conv2d(
#                 in_channels, inner_dim, kernel_size=1, stride=1, padding=0
#             )
#         else:
#             self.proj_in = nn.Linear(in_channels, inner_dim)

#         self.transformer_blocks = nn.ModuleList(
#             [
#                 SIGEBasicTransformerBlock(
#                     inner_dim,
#                     n_heads,
#                     d_head,
#                     dropout=dropout,
#                     context_dim=context_dim,
#                     disable_self_attn=disable_self_attn,
#                     attn_mode=attn_type,
#                     use_checkpoint=use_checkpoint,
#                     sdp_backend=sdp_backend,
#                 )
#                 for d in range(depth)
#             ]
#         )

#         if not use_linear:
#             self.proj_out = zero_module(
#                 Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
#             )
#         else:
#             # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
#             self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
#         self.use_linear = use_linear

#         if support_sparse:
#             self.gather = Gather(self.proj_in, block_size)
#             self.scatter1 = Scatter(self.gather)
#             self.scatter2 = Scatter(self.gather)
#         self.scale, self.shift = None, None

#     def forward(self, x, context=None):
#         # note: if no context is given, cross-attention defaults to self-attention
#         b, c, h, w = x.shape
#         x_in = x

#         if self.mode == "full":
#             if self.support_sparse:
#                 x = self.gather(x)
#             x, scale, shift = my_group_norm(x, self.norm)
#             self.scale, self.shift = scale, shift
#         elif self.mode in ("sparse", "profile"):
#             if self.support_sparse:
#                 x = self.gather(x, self.scale, self.shift)
#             else:
#                 x = x * self.scale + self.shift
#         else:
#             raise NotImplementedError("Unknown mode [%s]!!!" % self.mode)

#         if not self.use_linear:
#             x = self.proj_in(x).type(torch.float32)

#         if self.support_sparse:
#             full_x = self.scatter1(x)
#             full_x = rearrange(full_x, "b c h w -> b (h w) c")
#             if self.mode == "full":
#                 x = full_x
#             else:
#                 cc = x.size(1)
#                 x = x.view(b, -1, cc, x.size(2) * x.size(3))  # [b, nb, c, bh * bw]
#                 x = x.transpose(2, 3).reshape(b, -1, cc)
#         else:
#             full_x = None
#             x = rearrange(x, "b c h w -> b (h w) c")

#         if self.use_linear:
#             x = self.proj_in(x)

#         for block in self.transformer_blocks:
#             x = block(x, full_x=full_x, context=context).type(torch.float32)
        
#         if self.use_linear:
#             x = self.proj_out(x)

#         if self.support_sparse:
#             # [b, nb * bh * bw, c] -> [b * nb, c, bh, bw]
#             if self.mode == "full":
#                 x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
#             else:
#                 cc = x.size(-1)
#                 # [b, nb * bh * bw, c] -> [b, nb, bh * bw, c]
#                 x = x.view(b, -1, self.gather.block_size[0] * self.gather.block_size[1], cc)  # [b, nb, bh * bw, c]
#                 x = x.permute(0, 1, 3, 2).view(-1, cc, self.gather.block_size[0], self.gather.block_size[1])
#         else:
#             x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

#         if not self.use_linear:
#             x = self.proj_out(x).type(torch.float32)

#         if self.support_sparse:
#             x = self.scatter2(x, x_in).type(torch.float32)
#         else:
#             x = x + x_in
#         return x

from typing import Optional

import torch
from einops import rearrange, repeat
from torch import nn

from sige.nn import Gather, Scatter, SIGEConv2d, SIGEModule
from .attention import CrossAttention, default, exists, FeedForward, Normalize, SpatialTransformer, zero_module
from .diffusionmodules.sige_model import my_group_norm


class SIGECrossAttention(SIGEModule):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** (-0.5)
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

        self.cached_k = None
        self.cached_v = None

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        if self.mode == "full":
            k = self.to_k(context)
            v = self.to_v(context)
            self.cached_k = k
            self.cached_v = v
        else:
            k = self.cached_k
            v = self.cached_v
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        sim = torch.bmm(q, k.permute(0, 2, 1)) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        # out = einsum("b i j, b j d -> b i d", attn, v)
        out = torch.bmm(attn, v)

        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class SIGEBasicTransformerBlock(SIGEModule):
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True, use_checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = SIGECrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, full_x=None, context=None):
        x = self.attn1(self.norm1(x), context=None if full_x is None else self.norm1(full_x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SIGESpatialTransformer(SIGEModule, SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        use_checkpoint=True,
        block_size: Optional[int] = 4,
    ):
        super(SpatialTransformer, self).__init__()
        SIGEModule.__init__(self, call_super=False)
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        support_sparse = block_size is not None
        Conv2d = SIGEConv2d if support_sparse else nn.Conv2d

        self.support_sparse = support_sparse

        self.proj_in = Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                SIGEBasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    use_checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )

        self.proj_out = zero_module(Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))

        if support_sparse:
            self.gather = Gather(self.proj_in, block_size)
            self.scatter1 = Scatter(self.gather)
            self.scatter_blocks = nn.ModuleList([
              Scatter(self.gather) for _ in range(depth - 1)
            ])
            self.scatter2 = Scatter(self.gather)
        self.scale, self.shift = None, None
        self.depth = depth

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x

        if self.mode == "full":
            if self.support_sparse:
                x = self.gather(x)
            x, scale, shift = my_group_norm(x, self.norm)
            self.scale, self.shift = scale, shift
        elif self.mode in ("sparse", "profile"):
            if self.support_sparse:
                x = self.gather(x, self.scale, self.shift)
            else:
                x = x * self.scale + self.shift
        else:
            raise NotImplementedError("Unknown mode [%s]!!!" % self.mode)

        x = self.proj_in(x).type(torch.float32)

        if self.support_sparse:
            full_x = self.scatter1(x)
            full_x = rearrange(full_x, "b c h w -> b (h w) c")
            if self.mode == "full":
                x = full_x
            else:
                cc = x.size(1)
                x = x.view(b, -1, cc, x.size(2) * x.size(3))  # [b, nb, c, bh * bw]
                x = x.transpose(2, 3).reshape(b, -1, cc)
        else:
            full_x = None
            x = rearrange(x, "b c h w -> b (h w) c")

        for i,block in enumerate(self.transformer_blocks):
            x = block(x, full_x=full_x, context=context)
            # Update context for the next depth of the transformer block, no upadte on last iteration
            if i < self.depth - 1 and self.support_sparse:
              if self.mode == "full":
                  x_ = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
              else:
                  cc = x.size(-1)
                  # [b, nb * bh * bw, c] -> [b, nb, bh * bw, c]
                  x_ = x.view(b, -1, self.gather.block_size[0] * self.gather.block_size[1], cc)  # [b, nb, bh * bw, c]
                  x_ = x_.permute(0, 1, 3, 2).view(-1, cc, self.gather.block_size[0], self.gather.block_size[1])

              full_x = self.scatter_blocks[i](x_)
              full_x = rearrange(full_x, "b c h w -> b (h w) c")

        if self.support_sparse:
            # [b, nb * bh * bw, c] -> [b * nb, c, bh, bw]
            if self.mode == "full":
                x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            else:
                cc = x.size(-1)
                # [b, nb * bh * bw, c] -> [b, nb, bh * bw, c]
                x = x.view(b, -1, self.gather.block_size[0] * self.gather.block_size[1], cc)  # [b, nb, bh * bw, c]
                x = x.permute(0, 1, 3, 2).view(-1, cc, self.gather.block_size[0], self.gather.block_size[1])
        else:
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        x = self.proj_out(x).type(torch.float32)
        if self.support_sparse:
            x = self.scatter2(x, x_in)
        else:
            x = x + x_in
        return x
