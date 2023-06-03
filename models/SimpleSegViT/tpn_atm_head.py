import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import trunc_normal_, constant_

from .atm_single_head import ATMSingleHead
from .atm_head import trunc_normal_init, constant_init

import math

class TPNATMHead(nn.Module):
    def __init__(
            self,
            img_size,
            in_channels,
            embed_dims=768,
            num_layers=3,
            num_heads=8,
            use_stages=3,
            shrink_ratio=16,
            use_proj=True,
            CE_loss=False,
            crop_train=False,
            num_classes=60,
            training=True
    ):
        '''
        Args:
            embed_dims: embedding dimension of the class token
            num_layers: number of layers of a decoder block
        '''
        super().__init__()
        self.in_channels = in_channels
        dim = embed_dims
        self.use_stages = use_stages
        self.image_size = img_size

        proj_norm = []
        input_proj = []
        tpn_layers = []
        for i in range(self.use_stages):
            # FC layer to change ch
            if use_proj:
                proj = nn.Linear(self.in_channels, dim)
                trunc_normal_(proj.weight, std=.02)
            else:
                proj = nn.Identity()
            self.add_module("input_proj_{}".format(i + 1), proj)
            input_proj.append(proj)
            # norm layer
            if use_proj:
                norm = nn.LayerNorm(dim)
            else:
                norm = nn.Identity()
            self.add_module("proj_norm_{}".format(i + 1), norm)
            proj_norm.append(norm)
            # decoder layer
            decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim * 4, batch_first=True)
            decoder = nn.TransformerDecoder(decoder_layer, num_layers)
            self.add_module("decoder_{}".format(i + 1), decoder)
            tpn_layers.append(decoder)

        self.input_proj = input_proj
        self.proj_norm = proj_norm
        self.decoder = tpn_layers
        self.q = nn.Embedding((self.image_size // shrink_ratio)**2, dim)

        # atm
        self.atm = ATMSingleHead(img_size,
                           in_channels,
                           embed_dims,
                           num_layers,
                           num_heads,
                           use_stages=1,
                           use_proj=False,
                           CE_loss=CE_loss,
                           crop_train=crop_train,
                           shrink_ratio=shrink_ratio,
                           num_classes=num_classes,
                           training=training
                           )


    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)


    def forward(self, inputs):
        '''
        Args:
            inputs: a list of each layer's output from encoder; 
                expected shape (layers, B, L, C)
        Returns:
            out: if self.training is True then return a dictionary
                    pred_logits: class predictions; expected shape (B, N, N + 1)
                    pred_masks: mask; expected shape (B, N, h, w)
                    pred: semantic segmentation (B, N, h, w)
                    aux_outputs: a list of middle layer output
                        pred_logits: attention output (B, N, N + 1)
                        pred_masks: mask (B, N, h, w)
                 else return a semantic segmentation (B, N + 1, h, w)
        '''
        x = []
        for stage_ in inputs[:self.use_stages]:
            x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
        # do not reverse
        bs = x[0].size()[0] # batch_size, B
        laterals = []
        maps_size = []

        q = self.q.weight.repeat(bs, 1, 1) # (B, L, C)

        for idx, (x_, proj_, norm_, decoder_) in \
                enumerate(zip(x, self.input_proj, self.proj_norm, self.decoder)):
            lateral = norm_(proj_(x_)) # (B, L, C)
            q = decoder_(q, lateral) # (B, L, C)

        q = self.d3_to_d4(q) # (B, C, h, w)

        atm_out = self.atm([q])

        return atm_out


    def d3_to_d4(self, t: Tensor):
        '''
        Args:
            t: expected shape (B, L, N)
        Returns:
            expected shape: (B, N, h, w)
        '''
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)


    def d4_to_d3(self, t: Tensor):
        '''
        Args:
            t: expected shape (B, C, h, w)
        Returns:
            expected shape: (B, L, C)
        '''
        return t.flatten(-2).transpose(-1, -2)
