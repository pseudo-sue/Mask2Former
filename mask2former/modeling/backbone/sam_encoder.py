# file: d2_sam_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from segment_anything import sam_model_registry

@BACKBONE_REGISTRY.register()
class SAMBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        freeze = cfg.MODEL.SAM.FREEZE
        sam = sam_model_registry[cfg.MODEL.SAM.TYPE](checkpoint=cfg.MODEL.SAM.CKPT) # "vit_b" / "vit_l" / "vit_h" #checkpoint
        self.encoder = sam.image_encoder  # (B,256,64,64)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, groups=256)
        )
        self.downsampling = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, groups=256)
        )

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": 256,
            "res3": 256,
            "res4": 256,
            "res5": 256,
        }
        self._size_divisibility = 32

    def forward(self, x: torch.Tensor):
        feats16 = self.encoder(x)           # (B,256,64,64), stride=16
        feats8 = self.upsampling(feats16)
        feats4 = self.upsampling(feats8)
        feats32 = self.downsampling(feats16)

        return {"res2": feats4,"res3": feats8, "res4": feats16, "res5": feats32}

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name],
                            stride=self._out_feature_strides[name])
            for name in self._out_feature_channels
        }

    @property
    def size_divisibility(self):
        return self._size_divisibility

