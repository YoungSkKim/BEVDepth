# Copyright (c) Megvii Inc. All rights reserved.
from exps.base_cli import run_cli
from exps.radar.bev_depth_fusion_lss_r50_256x704_128x128_24e_radar import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel
from models.fusion_bev_depth_pts_bev import FusionBEVDepthPtsBEV


class BEVDepthLightningModel(BaseBEVDepthLightningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key_idxes = [-1]
        self.head_conf['bev_backbone_conf']['in_channels'] = 80 * (
            len(self.key_idxes) + 1)
        self.head_conf['bev_neck_conf']['in_channels'] = [
            80 * (len(self.key_idxes) + 1), 160, 320, 640
        ]
        self.head_conf['train_cfg']['code_weight'] = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
        self.model = FusionBEVDepthPtsBEV(self.backbone_conf,
                                          self.backbone_pts_conf,
                                          self.head_conf,
                                          is_train_depth=True)


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_fusion_lss_r50_256x704_128x128_24e_2key_radar')
