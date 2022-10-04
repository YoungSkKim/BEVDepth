# Copyright (c) Megvii Inc. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast

from mmcv.ops import Voxelization
from mmcv.runner import force_fp32

from mmdet3d.models import builder


class BackbonePoint(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 pts_voxel_layer,
                 pts_voxel_encoder,
                 pts_middle_encoder,
                 pts_backbone,
                 pts_neck
                 ):
        super(BackbonePoint, self).__init__()

        self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        self.pts_middle_encoder = builder.build_middle_encoder(pts_middle_encoder)
        self.pts_backbone = builder.build_backbone(pts_backbone)
        self.pts_neck = builder.build_neck(pts_neck)


    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        batch_size, _, _ = points.shape
        points_list = [points[i] for i in range(batch_size)]

        for res in points_list:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def _forward_single_sweep(self, ptss):
        voxels, num_points, coors = self.voxelize(ptss)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)

        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        return x[0]

    @autocast(False)
    def forward(self, pts):
        batch_size, num_sweeps, _, _ = pts.shape

        key_frame_res = self._forward_single_sweep(pts[:, 0, ...])
        if num_sweeps == 1:
            return key_frame_res

        ret_feature_list = [key_frame_res]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(pts[:, sweep_index, ...])
                ret_feature_list.append(feature_map)

        return torch.cat(ret_feature_list, 1)
