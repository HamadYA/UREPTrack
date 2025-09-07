# vit_format_poolformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

# ---- bring these from your PoolFormer file (import them instead if in a module) ----
# PatchEmbed (conv2d version), PoolFormerBlock, GroupNorm, Mlp, Pooling
# -----------------------------------------------------------------------------------

class _PoolFormerBlockAsTokens(nn.Module):
    """
    Wrap a PoolFormerBlock (which expects [B,C,H,W]) so it can be used in a ViT-style
    token pipeline ([B,N,C]). We keep track of H,W to reshape back/forth.
    """
    def __init__(self, block_2d: nn.Module, embed_dim: int, hw: tuple[int, int]):
        super().__init__()
        self.block_2d = block_2d
        self.embed_dim = embed_dim
        self.H, self.W = hw

    def forward(self, x_tokens: torch.Tensor):
        # x_tokens: [B, N, C] with N = H*W
        B, N, C = x_tokens.shape
        assert C == self.embed_dim, "Channel mismatch"
        assert N == self.H * self.W, "Token/HW mismatch"

        x = x_tokens.transpose(1, 2).contiguous()        # [B,C,N]
        x = x.view(B, C, self.H, self.W)                 # [B,C,H,W]
        x = self.block_2d(x)                             # still [B,C,H,W]
        x = x.flatten(2).transpose(1, 2).contiguous()    # [B,N,C]
        return x


class PoolFormerViTAdapter(nn.Module):
    """
    Make a PoolFormer backbone present the same *public* interface as a ViT:
      - .patch_embed : module producing patch features
      - .pos_embed   : [1, N(+1), C] learnable
      - .cls_token   : [1,1,C] optional (enabled by add_cls_token)
      - .pos_drop
      - .blocks      : nn.Sequential of token-wise blocks (each returns [B,N,C])
      - .norm        : final norm over channels
      - .head        : classifier head (nn.Linear) or Identity

    forward_features(x) -> [B, N(+1), C]
    forward(x) -> logits (if head not Identity)
    """
    def __init__(
        self,
        poolformer,                 # an instantiated PoolFormer(...)
        img_size=224,
        in_chans=3,
        add_cls_token=True,
        num_classes=1000,
        norm_layer=None,
        drop_rate=0.0,
    ):
        super().__init__()
        self.backbone = poolformer
        self.add_cls_token = add_cls_token
        self.num_classes = num_classes

        # Use the first PatchEmbed from PoolFormer as "ViT patch_embed"
        self.patch_embed = self.backbone.patch_embed
        embed_dim = self.patch_embed.proj.out_channels
        self.embed_dim = embed_dim

        # Infer token grid size after the first embedding for a given img_size
        # NOTE: assumes square input and integer stride/padding as in PoolFormer
        stride = self.patch_embed.proj.stride
        padding = self.patch_embed.proj.padding
        kernel  = self.patch_embed.proj.kernel_size

        def _out_len(L, k, s, p):
            # standard conv formula with floor
            return (L + 2*p - k) // s + 1

        H = W = img_size
        H = _out_len(H, kernel[0], stride[0], padding[0])
        W = _out_len(W, kernel[1], stride[1], padding[1])
        self.grid_size = (H, W)
        self.num_patches = H * W

        # Positional embeddings and optional cls token
        pos_len = self.num_patches + (1 if add_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_len, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        # Wrap PoolFormer stages/blocks as token-wise "ViT blocks"
        # self.backbone.network is [stage0, down, stage1, down, stage2, down, stage3]
        # We only treat the *stages* (PoolFormerBlock sequences) as blocks over tokens.
        token_blocks = []

        # current HW after successive downsamplings
        cur_H, cur_W = H, W
        modules = list(self.backbone.network)

        i = 0
        while i < len(modules):
            m = modules[i]
            if isinstance(m, nn.Sequential):
                # m is a stage of PoolFormerBlock(s) operating at (cur_H, cur_W)
                for blk in m:
                    token_blocks.append(_PoolFormerBlockAsTokens(blk, embed_dim, (cur_H, cur_W)))
                i += 1
            else:
                # m is a downsampling PatchEmbed: updates (H,W) and embed_dim
                # We "inline" this into the token pipeline by reshaping tokens -> 2D -> conv -> tokens
                down = m
                def make_down_token(op, in_ch, out_ch, new_hw):
                    class _DownAsToken(nn.Module):
                        def __init__(self, op, in_ch, out_ch, old_hw, new_hw):
                            super().__init__()
                            self.op = op
                            self.in_ch = in_ch
                            self.out_ch = out_ch
                            self.old_H, self.old_W = old_hw
                            self.new_H, self.new_W = new_hw
                        def forward(self, x_tokens):
                            B, N, C = x_tokens.shape
                            assert C == self.in_ch and N == self.old_H*self.old_W
                            x = x_tokens.transpose(1,2).contiguous().view(B, C, self.old_H, self.old_W)
                            x = self.op(x)  # [B, out_ch, new_H, new_W]
                            x = x.flatten(2).transpose(1,2).contiguous()
                            return x
                    return _DownAsToken(op, in_ch, out_ch, (cur_H, cur_W), new_hw)

                # compute new HW after this conv downsample
                s = down.proj.stride
                p = down.proj.padding
                k = down.proj.kernel_size
                new_H = (cur_H + 2*p[0] - k[0]) // s[0] + 1
                new_W = (cur_W + 2*p[1] - k[1]) // s[1] + 1

                down_token = make_down_token(down, embed_dim, down.proj.out_channels, (new_H, new_W))
                token_blocks.append(down_token)

                # update trackers
                cur_H, cur_W = new_H, new_W
                embed_dim = down.proj.out_channels
                self.embed_dim = embed_dim  # keep current dim
                i += 1

        # After wrapping, unify dims for pos/norm/head: re-project if final dim != initial
        self.blocks = nn.Sequential(*token_blocks)

        # final norm & head in ViT style
        self.norm = getattr(self.backbone, 'norm', nn.LayerNorm(self.embed_dim))
        if isinstance(self.norm, (nn.Identity, )):
            # PoolFormer norm is GroupNorm on [B,C,H,W]; replace with LayerNorm over C (tokens)
            self.norm = nn.LayerNorm(self.embed_dim)

        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # init
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=0.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def _embed_to_tokens(self, x):
        # x: [B,3,H,W] -> conv embed -> [B,C,H',W'] -> tokens [B,N,C]
        x = self.patch_embed(x)                         # [B,C,H',W']
        H, W = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2).contiguous()   # [B,N,C]
        return x, (H, W)

    def forward_features(self, x):
        B = x.shape[0]
        x, (H, W) = self._embed_to_tokens(x)           # [B,N0,C0] at stage 0
        # cls token
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)     # [B,1,C]
            x = torch.cat([cls, x], dim=1)             # [B,1+N0,C]

        # pos embed
        # if dims changed later by downsampling, we keep using the **current seq length**
        # and slice or pad pos_embed on the fly
        if self.pos_embed is not None:
            if x.shape[1] != self.pos_embed.shape[1]:
                # resize positional embeddings via interpolation (2D grid assumption)
                # 1) split cls and grid
                if self.cls_token is not None:
                    pos_cls, pos_grid = self.pos_embed[:, :1], self.pos_embed[:, 1:]
                    gh = int(self.num_patches ** 0.5)
                    pos_grid = pos_grid.reshape(1, gh, gh, -1).permute(0, 3, 1, 2)  # [1,C,gh,gw]
                    # interpolate to current H,W
                    pos_grid = F.interpolate(pos_grid, size=(H, W), mode='bicubic', align_corners=False)
                    pos_grid = pos_grid.permute(0, 2, 3, 1).reshape(1, H*W, -1)
                    pos = torch.cat([pos_cls, pos_grid], dim=1)  # [1,1+HW,C]
                else:
                    gh = int(self.num_patches ** 0.5)
                    pos_grid = self.pos_embed.reshape(1, gh, gh, -1).permute(0, 3, 1, 2)
                    pos_grid = F.interpolate(pos_grid, size=(H, W), mode='bicubic', align_corners=False)
                    pos = pos_grid.permute(0, 2, 3, 1).reshape(1, H*W, -1)
            else:
                pos = self.pos_embed
            x = x + pos

        x = self.pos_drop(x)

        # run token pipeline (each block reshapes internally to 2D when needed)
        x = self.blocks(x)  # [B, N_final, C_final]
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)   # [B,N,C]
        if self.cls_token is not None:
            cls = x[:, 0]              # [B,C]
        else:
            cls = x.mean(dim=1)        # GAP over tokens if no CLS
        return self.head(cls)



# factories.py
from poolformer import poolformer_s12, poolformer_s24, poolformer_s36, poolformer_m36, poolformer_m48
from vit_format_poolformer import PoolFormerViTAdapter

def poolformer_s24_vitfmt(pretrained=False, img_size=224, num_classes=1000, **kwargs):
    pf = poolformer_s24(pretrained=pretrained, **kwargs)   # original backbone
    # Note: PoolFormer’s last stage changes channel dims; adapter tracks that.
    model = PoolFormerViTAdapter(
        pf, img_size=img_size, num_classes=num_classes, add_cls_token=True, drop_rate=0.0
    )
    return model
