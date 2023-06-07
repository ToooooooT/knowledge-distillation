from .atm_head import ATMHead
from .tpn_atm_head import TPNATMHead
from .encoder import ViTEncoder

import torch
import torch.nn as nn
from torch import Tensor

class SimpleSegViT(nn.Module):
    def __init__(self,
                 shrink_idx=8,
                 img_size=224,
                 patch_size=16,
                 encoder_in_channels=3,
                 encoder_embed_dims=768,
                 encoder_num_layers=12,
                 encoder_num_heads=12,
                 mlp_ratio=4,
                 out_indices=-1,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 with_cls_token=True,
                 output_cls_token=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=False,
                 interpolate_mode='bicubic',
                 num_fcs=2,
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None, # --- encoder ---
                 single=False,
                 decoder_in_channels=3, # --- decoder ---
                 decoder_embed_dims=768,
                 decoder_num_layers=3,
                 decoder_num_heads=8,
                 use_stages=3,
                 use_proj=True,
                 CE_loss=False,
                 crop_train=False,
                 shrink_ratio=None,
                 num_classes=150,
                 training=True) -> None:
        super().__init__()

        if isinstance(out_indices, int):
            assert use_stages == 1
        elif isinstance(out_indices, list):
            assert len(out_indices) >= use_stages

        self.encoder = ViTEncoder(shrink_idx=shrink_idx,
                                img_size=img_size,
                                patch_size=patch_size,
                                in_channels=encoder_in_channels,
                                embed_dims=encoder_embed_dims,
                                num_layers=encoder_num_layers,
                                num_heads=encoder_num_heads,
                                mlp_ratio=mlp_ratio,
                                out_indices=out_indices,
                                qkv_bias=qkv_bias,
                                drop_rate=drop_rate,
                                attn_drop_rate=attn_drop_rate,
                                drop_path_rate=drop_path_rate,
                                with_cls_token=with_cls_token,
                                output_cls_token=output_cls_token,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg,
                                patch_norm=patch_norm,
                                final_norm=final_norm,
                                interpolate_mode=interpolate_mode,
                                num_fcs=num_fcs,
                                norm_eval=norm_eval,
                                with_cp=with_cp,
                                pretrained=pretrained,
                                init_cfg=init_cfg)
                            
        if single:
            self.decoder = TPNATMHead(img_size,
                                    in_channels=decoder_in_channels,
                                    embed_dims=decoder_embed_dims,
                                    num_layers=decoder_num_layers,
                                    num_heads=decoder_num_heads,
                                    use_stages=use_stages,
                                    shrink_ratio=shrink_ratio,
                                    use_proj=use_proj,
                                    CE_loss=CE_loss,
                                    crop_train=crop_train,
                                    num_classes=num_classes,
                                    training=training)
        else:
            self.decoder = ATMHead(img_size,
                                in_channels=decoder_in_channels,
                                embed_dims=decoder_embed_dims,
                                num_layers=decoder_num_layers,
                                num_heads=decoder_num_heads,
                                use_stages=use_stages,
                                use_proj=use_proj,
                                CE_loss=CE_loss,
                                crop_train=crop_train,
                                shrink_ratio=shrink_ratio,
                                num_classes=num_classes,
                                training=training)


    def forward(self, input: Tensor):
        '''
        Args:
            input: expected shape (B, C, H, W)
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

        out = self.encoder(input)
        out = self.decoder(out)
        return out