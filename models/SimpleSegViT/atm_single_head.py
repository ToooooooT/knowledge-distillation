import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn.init import trunc_normal_, constant_
from typing import Optional

import math

'''
dimension correspond to the paper
B: batch_size
L: HW/P^2
N: query number, in this paper is number of classes
C: embedding dimension of class token
'''

def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        constant_(module.bias, bias)  # type: ignore

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        constant_(module.bias, bias)

class TPN_Decoder(TransformerDecoder):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        '''
        Args: 
            tgt: expected shape (B, N, C)
            memory: expected shape (B, L, C)
        Returns:
            outputs: a list of class embedding; expected shape (num_layers, B, N, C)
            attns: a list of mask; expected shape (num_layers, B, N, L)
        '''
        output = tgt
        attns = []
        outputs = []
        for mod in self.layers:
            output, attn = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            attns.append(attn)
            outputs.append(output)
        if self.norm is not None:
            output = self.norm(output)

        return outputs, attns

class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        '''
        Args: 
            tgt: expected shape (B, N, C)
            memory: expected shape (B, L, C)
        Returns:
            tgt: expected shape (B, N, C)
            attn2: expected shape (B, N, L)
        '''
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn2 = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, 
                xq: Tensor, 
                xk: Tensor, 
                xv: Tensor):
        '''
        Args:
            xq: expected shape (B, N, C)
            xk: expected shape (B, L, C)
            xv: expected shape (B, L, C)
        Returns:
            x: expected shape (B, N, C)
            attn: expected shape (B, N, L)
        '''
        B, Nq, C = xq.size()
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        q = self.q(xq).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3) # (B, num_heads, N, C // num_heads)
        k = self.k(xk).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3) # (B, num_heads, L, C // num_heads)
        v = self.v(xv).reshape(B, Nv, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3) # (B, num_heads, L, C // num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, num_heads, N, L)
        attn_save = attn.clone() # (B, num_heads, N, L)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # (B, num_heads, N, L)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C) # (B, num_heads, N, C // num_heads) -> (B, N, C)
        x = self.proj(x) # (B, N, C)
        x = self.proj_drop(x) # (B, N, C)
        return x, attn_save.sum(dim=1) / self.num_heads # (B, N, C), (B, N, L)

class ATMSingleHead(nn.Module):
    def __init__(
            self,
            img_size,
            in_channels,
            embed_dims=768,
            num_layers=3,
            num_heads=8,
            use_stages=3,
            use_proj=True,
            CE_loss=False,
            crop_train=False,
            shrink_ratio=None,
            num_classes=60,
            training=True
    ):
        '''
        Args:
            embed_dims: embedding dimension of the class token
            num_layers: number of layers of a decoder block
        '''
        super().__init__()

        self.image_size = img_size
        self.in_channels = in_channels
        self.use_stages = use_stages
        self.crop_train = crop_train
        self.num_classes = num_classes
        self.training = training
        nhead = num_heads
        dim = embed_dims
        input_proj = []
        proj_norm = []
        atm_decoders = []
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
            decoder_layer = TPN_DecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4, batch_first=True)
            decoder = TPN_Decoder(decoder_layer, num_layers)
            self.add_module("decoder_{}".format(i + 1), decoder)
            atm_decoders.append(decoder)

        self.input_proj = input_proj
        self.proj_norm = proj_norm
        self.decoder = atm_decoders
        self.q = nn.Embedding(self.num_classes, dim)

        self.class_embed = nn.Linear(dim, self.num_classes + 1)
        self.CE_loss = CE_loss

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
    
    def forward(self, inputs: Optional[Tensor]):
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
        x.reverse()
        bs = x[0].shape[0] # batch_size, B

        laterals = []
        attns = [] # store mask of each layer
        maps_size = [] # store the size (h, w) of mask of each layer
        qs = [] # store output of each layer
        q = self.q.weight.repeat(bs, 1, 1) # (B, N, C)

        for idx, (x_, proj_, norm_, decoder_) in enumerate(zip(x, self.input_proj, self.proj_norm, self.decoder)):
            lateral = norm_(proj_(x_))
            laterals.append(lateral)

            q_, attn_ = decoder_(q, lateral) # a list of q (num_layers, B, N, C), a list of attn (num_layers, B, N, L)
            for q, attn in zip(q_, attn_):
                attn = attn.transpose(-1, -2) # attn (B, L, N)
                attn = self.d3_to_d4(attn) # (B, N, h, w)
                maps_size.append(attn.size()[-2:])
                qs.append(q)
                attns.append(attn)
        qs = torch.stack(qs, dim=0) # (self.use_stages, B, N, C)
        outputs_class = self.class_embed(qs)  # (self.use_stages, B, N, N + 1)
        out = {"pred_logits": outputs_class[-1]} # class predictions : (B, N, N + 1)

        outputs_seg_masks = [] # store the cumulative mask (self.use_stages, B, N, h, w)
        size = maps_size[-1] # last mask output size (h, w)

        for i_attn, attn in enumerate(attns):
            # TODO: why source code did not add all mask?
            if i_attn == 0:
                outputs_seg_masks.append(F.interpolate(attn, size=size, mode='bilinear', align_corners=False))
            else:
                outputs_seg_masks.append(outputs_seg_masks[i_attn - 1] +
                                         F.interpolate(attn, size=size, mode='bilinear', align_corners=False))

        out["pred_masks"] = F.interpolate(outputs_seg_masks[-1],
                                          size=(self.image_size, self.image_size),
                                          mode='bilinear', align_corners=False) # (B, N, h, w)

        out["pred"] = self.semantic_inference(out["pred_logits"], out["pred_masks"]) # (B, N + 1, h, w)

        if self.training:
            outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0) # (self.use_stages, B, N, h, w)
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_seg_masks
            )
        else:
            return out["pred"]

        return out


    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]
        

    def semantic_inference(self, mask_cls: Tensor, mask_pred: Tensor):
        '''
        Args:
            mask_cls: class predictions (B, N, N + 1)
            mask_pred: mask (B, N, h, w)
        Returns:
            semseg: semantic segmentation (B, N + 1, h, w)
        '''
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg


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