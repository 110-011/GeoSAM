import torch
from torch import nn
from toolbox.models.Paper3.GeoSAM.mask_decoder import PromptMaskDecoder
from toolbox.models.Paper3.GeoSAM.image_encoder_Diff import ImageEncoderViT
from toolbox.models.Paper3.GeoSAM.transformer import TwoWayTransformer
from toolbox.models.Paper3.GeoSAM.prompt_encoder import PromptEncoder

from typing import Any, Optional, Tuple, Type

from reprlib import recursive_repr
import numpy as np
from torch.nn import functional as F
from toolbox.models.Paper3.GeoSAM.GeoDiffAdapter import ClassPriorGenerator
import random
from toolbox.models.Paper3.GeoSAM.prompt_diffusion import PromptGenerator, PromptDiffusionNet

class Geo_SAM(nn.Module):
    def __init__(self, img_size=512, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.pe_layer = PositionEmbeddingRandom(256 // 2)

        self.image_embedding_size = [img_size // 16, img_size // 16]
        self.img_size = img_size

        self.image_encoder = ImageEncoderViT(depth=12,
                                             embed_dim=768,
                                             img_size=img_size,
                                             mlp_ratio=4,
                                             norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                                             num_heads=12,
                                             patch_size=16,
                                             qkv_bias=True,
                                             use_rel_pos=True,
                                             global_attn_indexes=[2, 5, 8, 11],
                                             window_size=14,
                                             out_chans=256)

        self.class_prior = ClassPriorGenerator()

        self.prompt_encoder = PromptEncoder(embed_dim=256, image_embedding_size=(256, 256), mask_in_chans=6)

        self.mask_decoder = PromptMaskDecoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8
            ),
            transformer_dim=256,
            norm=norm,
            act=act
        )

        self.featuredecoder = FeatureDecoder()

        # === 注入 PromptDiffusion 模块与 PromptGenerator ===
        self.prompt_diffusion = PromptDiffusionNet(in_channels=6, cond_dim=256)
        self.prompt_generator = PromptGenerator(self.prompt_diffusion)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, img, ndsm, labels: Optional[torch.Tensor] = None, is_train=True):
        b = img.size(0)
        # get different layer's features of the image encoder
        class_prior = self.class_prior(ndsm, labels)  # [B, num_classes, 256, 256]
        features_list = self.image_encoder(img, ndsm, class_prior)# [2, 256, 16, 16]

        # === 提取条件特征向量 ===
        cond_feat = F.adaptive_avg_pool2d(features_list[-1], 1).view(b, -1)  # [B, 256]

        if is_train:
            t = torch.randint(0, 1000, (b,), device=img.device)
            gt_mask = (labels == 1).float().unsqueeze(1)  # [B, 1, H, W]
            gt_mask = torch.repeat_interleave(gt_mask, 6, dim=1)
            loss = self.prompt_generator.training_loss(gt_mask, t, cond_feat)

            # 训练阶段从 x_t - pred_noise 构造提示
            noise = torch.randn_like(gt_mask)
            beta_schedule = torch.linspace(1e-4, 0.02, 1000).to(img.device)
            alpha = 1. - beta_schedule
            alpha_bar = torch.cumprod(alpha, dim=0)[t].view(-1, 1, 1, 1)
            x_t = torch.sqrt(alpha_bar) * gt_mask + torch.sqrt(1 - alpha_bar) * noise
            pred_noise = self.prompt_diffusion(x_t, t, cond_feat)
            prompt_mask = x_t - pred_noise
        else:
            prompt_mask = self.prompt_generator.generate_prompt(cond_feat, shape=(b, 6, self.img_size, self.img_size))
            loss = None

        dense_prompt_embeddings = self.prompt_encoder(prompt_mask)
        # extract intermediate outputs for deep supervision to prevent model_01 overfitting on the detail enhancement module.
        img_pe = self.get_dense_pe()
        _, feature = self.mask_decoder(features_list[-1], dense_prompt_embeddings, img_pe)# [2, 32, 64, 64]

        mask = self.featuredecoder(features_list, feature)

        return mask, prompt_mask, loss

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
            self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

class partial:
    """New function with partial application of the given arguments
    and keywords.
    """

    __slots__ = "func", "args", "keywords", "__dict__", "__weakref__"

    def __new__(cls, func, *args, **keywords):
        if not callable(func):
            raise TypeError("the first argument must be callable")

        if hasattr(func, "func"):
            args = func.args + args
            keywords = {**func.keywords, **keywords}
            func = func.func

        self = super(partial, cls).__new__(cls)

        self.func = func
        self.args = args
        self.keywords = keywords
        return self

    def __call__(self, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        return self.func(*self.args, *args, **keywords)

    @recursive_repr()
    def __repr__(self):
        qualname = type(self).__qualname__
        args = [repr(self.func)]
        args.extend(repr(x) for x in self.args)
        args.extend(f"{k}={v!r}" for (k, v) in self.keywords.items())
        if type(self).__module__ == "functools":
            return f"functools.{qualname}({', '.join(args)})"
        return f"{qualname}({', '.join(args)})"

    def __reduce__(self):
        return type(self), (self.func,), (self.func, self.args,
                                          self.keywords or None, self.__dict__ or None)

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 4:
            raise TypeError(f"expected 4 items in state, got {len(state)}")
        func, args, kwds, namespace = state
        if (not callable(func) or not isinstance(args, tuple) or
                (kwds is not None and not isinstance(kwds, dict)) or
                (namespace is not None and not isinstance(namespace, dict))):
            raise TypeError("invalid partial state")

        args = tuple(args)  # just in case it's a subclass
        if kwds is None:
            kwds = {}
        elif type(kwds) is not dict:  # XXX does it need to be *exactly* dict?
            kwds = dict(kwds)
        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self.func = func
        self.args = args
        self.keywords = kwds

def reshapePos(pos_embed, img_size):
    token_size = int(img_size // 16)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
    return pos_embed

def reshapeRel(k, rel_pos_params, img_size):
    if not ('2' in k or '5' in k or '8' in k or '11' in k):
        return rel_pos_params

    token_size = int(img_size // 16)
    h, w = rel_pos_params.shape
    rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
    rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
    return rel_pos_params[0, 0, ...]

def load(net, ckpt, img_size):
    ckpt = torch.load(ckpt, map_location='cpu')
    from collections import OrderedDict
    dict = OrderedDict()
    for k, v in ckpt.items():
        # 把pe_layer改名
        if 'pe_layer' in k:
            dict[k[15:]] = v
            continue
        if 'pos_embed' in k:
            dict[k] = reshapePos(v, img_size)
            continue
        if 'rel_pos' in k:
            dict[k] = reshapeRel(k, v, img_size)
        elif "image_encoder" in k:
            if "neck" in k:
                # Add the original final neck layer to 3, 6, and 9, initialization is the same.
                for i in range(4):
                    new_key = "{}.{}{}".format(k[:18], i, k[18:])
                    dict[new_key] = v
            else:
                dict[k] = v
        if "mask_decoder.transformer" in k:
            dict[k] = v
        if "mask_decoder.iou_token" in k:
            dict[k] = v
        if "mask_decoder.output_upscaling" in k:
            dict[k] = v
    state = net.load_state_dict(dict, strict=False)
    return state

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
        )

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
        )

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels // 4, n_filters, 1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.conv3(x)
        return x

class SingleClassDecoder(nn.Module):
    def __init__(self, in_channels, mid_channels=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.block(x) # [B, 1, 64, 64]
        return x

'''
Task-specific Decoder:
采用多分支结构，例如为每一类语义设计专门的解码分支，然后 ensemble 融合：
'''
class ClassDecoder(nn.Module):
    def __init__(self, in_channels=32, num_classes=6, mid_channels=32):
        super().__init__()
        self.decoders = nn.ModuleList([
            SingleClassDecoder(in_channels, mid_channels) for _ in range(num_classes)
        ])

    def forward(self, x):
        outputs = [decoder(x) for decoder in self.decoders]  # List of [B, 1, 256, 256]
        return torch.cat(outputs, dim=1)                     # [B, 6, 256, 256]

class FeatureDecoder(nn.Module):
    def __init__(self, norm=nn.BatchNorm2d, act=nn.ReLU):
        super(FeatureDecoder, self).__init__()
        self.DB_12 = DecoderBlock(256, 16)
        self.DB_34 = DecoderBlock(256, 16)
        self.DB_1234 = DecoderBlock(32, 16)
        self.DB = DecoderBlock(32, 16)
        self.out_conv = nn.Sequential(
            DecoderBlock(48, 32),
            DecoderBlock(32, 32),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            norm(32),
            act(),
        )

        self.class_decoder = ClassDecoder(in_channels=32, num_classes=6)

    def forward(self, features_list, feature):
        mask_12 = self.DB_12(features_list[0] + features_list[1])  # [2, 16, 32, 32]
        mask_34 = self.DB_34(features_list[2] + features_list[3])  # [2, 16, 32, 32]
        mask_1234 = self.DB_1234(torch.cat([mask_12, mask_34], dim=1))  # [2, 16, 64, 64]
        mask = self.out_conv(torch.cat([mask_1234, feature], dim=1))  # [2, 16, 256, 256]

        mask = self.class_decoder(mask)
        return mask

if __name__ == '__main__':
    model = Geo_SAM(img_size=256).cuda()
    img = torch.randn(4, 3, 256, 256).cuda()
    ndsm = torch.randn(4, 1, 256, 256).cuda()
    label = torch.randint(0, 6, (4, 256, 256)).cuda()
    mask, prompt_mask, loss = model(img, ndsm, label, is_train=True)
    print(mask.shape, prompt_mask.shape, loss.item())
