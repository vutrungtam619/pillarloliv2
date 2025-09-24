import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from configs.config import config
from packages import Voxelization, nms_cuda
from utils import points_lidar2image, bounding_bboxes, Anchors, anchor_target, anchors2bboxes, limit_period

class ImageStem(nn.Module):
    def __init__(self, mean, std, shape, out_channel, device=None):
        super(ImageStem, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device)
        self.shape = shape
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        ).to(self.device)

    def image2tensor(self, batch_image):
        batch_tensor = []
        for image in batch_image:
            image_tensor = torch.from_numpy(image).permute(2,0,1).to(dtype=torch.float32, device=self.device) / 255.0  # (C,H,W)
            image_tensor = (image_tensor - self.mean[:, None, None]) / self.std[:, None, None]
            if image.shape[:2] != self.shape:  # shape[:2] = H,W
                image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=self.shape, mode='bilinear', align_corners=False).squeeze(0)
            batch_tensor.append(image_tensor)
        batch_tensor = torch.stack(batch_tensor, dim=0)  # (B, C, H, W)
        return batch_tensor

    def forward(self, batch_image):
        batch_tensor = self.image2tensor(batch_image)
        stem = self.stem(batch_tensor)
        return stem
    
class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_points, max_voxels):
        super(PillarLayer, self).__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size, point_cloud_range=point_cloud_range, max_num_points=max_points, max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        """ Generate pillar from points
        Args:
            batched_pts [list torch.tensor float32, (N, 4)]: list of batch points, each batch have shape (N, 4)
                        
        Returns:
            pillars [torch.tensor float32, (p1 + p2 + ... + pb, N, c)]: c is number of features per point
            coors_batch [torch.tensor int64, (p1 + p2 + ... + pb, 1 + 3)]: coordinate of each pillar, 1 is the batch index
            num_points_per_pillar [torch.tensor int64, (p1 + p2 + ... + pb, )]: number of points in each pillar
        """
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 
            # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
        
        pillars = torch.cat(pillars, dim=0) # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0) # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0) # (p1 + p2 + ... + pb, 1 + 3)

        return pillars, coors_batch, npoints_per_pillar
        
class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, num_blocks, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.num_blocks = num_blocks
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_grid = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_grid = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
        
        self.point_encoder = nn.Sequential(
            nn.Conv1d(6, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True)
        )
        
        self.pillar_encoder = nn.Sequential(
            nn.Linear(num_blocks*2, out_channel, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True)
        )
        
        self.pillar_point_encoder = nn.Sequential(
            nn.Linear(out_channel * 2, out_channel, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True)               
        )

    def extract_boundary_points(self, pillars, npoints_per_pillars):
        '''
        Args:
            pillars [torch.tensor float32, (p1 + p2 + ... + pb, N, c)]: c is number of features per point
            num_points_per_pillar [torch.tensor int64, (p1 + p2 + ... + pb, )]: number of points in each pillar

        Returns:
            boundary_pts: [np.ndarray float32, (p1 + p2 + ... + pb, 4, 3)]: (z_min, y_min, z_max, y_max) points
        '''
        num_pillars, max_points, c = pillars.shape
        pts_xyz = pillars[..., :3]  

        mask = torch.arange(max_points, device=pillars.device)[None, :] < npoints_per_pillars[:, None]  
        # z_min
        z = pts_xyz[..., 2]  
        z_masked = z.masked_fill(~mask, float("inf"))
        idx_zmin = torch.argmin(z_masked, dim=1)
        z_min_pts = pts_xyz[torch.arange(num_pillars), idx_zmin]  
        # z_max
        z_masked = z.masked_fill(~mask, float("-inf"))
        idx_zmax = torch.argmax(z_masked, dim=1)
        z_max_pts = pts_xyz[torch.arange(num_pillars), idx_zmax]  
        # y_min
        y = pts_xyz[..., 1]
        y_masked = y.masked_fill(~mask, float("inf"))
        idx_ymin = torch.argmin(y_masked, dim=1)
        y_min_pts = pts_xyz[torch.arange(num_pillars), idx_ymin]
        # y_max
        y_masked = y.masked_fill(~mask, float("-inf"))
        idx_ymax = torch.argmax(y_masked, dim=1)
        y_max_pts = pts_xyz[torch.arange(num_pillars), idx_ymax]

        boundary_pts = torch.stack([z_min_pts, y_min_pts, z_max_pts, y_max_pts], dim=1)  # (p1 + p2 + ... + pb, 4, 3)
        return boundary_pts.cpu().numpy()

    def bboxes_to_rois(self, image_bboxes, coors_batch, batch_image_map, img_shape=(1280, 720)):
        '''
        '''
        device = batch_image_map.device
        num_pillars = image_bboxes.shape[0]

        scale_x = batch_image_map.shape[3] / img_shape[1]
        scale_y = batch_image_map.shape[2] / img_shape[0]

        rois = []
        for i in range(num_pillars):
            x1, y1 = image_bboxes[i][:,0].min(), image_bboxes[i][:,1].min()
            x2, y2 = image_bboxes[i][:,0].max(), image_bboxes[i][:,1].max()

            x1, y1 = x1 * scale_x, y1 * scale_y
            x2, y2 = x2 * scale_x, y2 * scale_y

            batch_idx = int(coors_batch[i,0].item())  
            rois.append([batch_idx, x1, y1, x2, y2])

        rois = torch.tensor(rois, dtype=torch.float32, device=device)  # (p1 + p2 + ... + pb, 5)
        return rois

    def scatter_features(self, features, coors_batch, bs, x_l, y_l, out_channel, device):
        batched_canvas = []
        for i in range(bs):
            cur_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_idx, :]
            cur_features = features[cur_idx]

            canvas = torch.zeros((x_l, y_l, out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()  # (C, H, W)
            batched_canvas.append(canvas)
        return torch.stack(batched_canvas, dim=0)  # (B, C, H, W)

    def forward(self, pillars, coors_batch, npoints_per_pillar, batch_image_map, batch_calibs, batch_size):
        ''' Encode pillars into BEV feature map of lidar and image
        Args:
            pillars [torch.tensor float32, (p1 + p2 + ... + pb, N, c)]: c is number of features per point
            coors_batch [torch.tensor int64, (p1 + p2 + ... + pb, 1 + 3)]: coordinate of each pillar, 1 is the batch index
            num_points_per_pillar [torch.tensor int64, (p1 + p2 + ... + pb, )]: number of points in each pillar
            
        Returns: 
            bev_lidar [torch.tensor float32, (B, C, H, W)]: bev map of lidar features
            bev_image [torch.tensor float32, (B, C, H, W)]: bev map of image features
        '''
        device = pillars.device
        num_pillars = pillars.shape[0]

        # Point feature encoding
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None] # (p1 + p2 + ... + pb, num_points, 3)

        x_offset_pi_center = pillars[:, :, 0:1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset) # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset) # (p1 + p2 + ... + pb, num_points, 1)

        features = torch.cat([pillars[:, :, 2:3], offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1) # (p1 + p2 + ... + pb, num_points, 6) which x, y, intensity is remove

        # find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(device) # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :] # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]

        point_features = features.permute(0, 2, 1).contiguous() # (p1 + p2 + ... + pb, 6, num_points)
        point_features = self.point_encoder(point_features) # (p1 + p2 + ... + pb, out_channel, num_points)
        point_features = torch.max(point_features, dim=-1)[0] # (p1 + p2 + ... + pb, out_channel)
        
        # Pillar feature encoding
        block_size = (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.num_blocks
        pillar_features = []
        for b in range(self.num_blocks):
            z_lower = self.point_cloud_range[2] + b * block_size
            z_upper = z_lower + block_size           
            point_in_blocks = (pillars[:, :, 2] >= z_lower) & (pillars[:, :, 2] < z_upper) & mask
            num_pts_block = point_in_blocks.sum(dim=1).float() / npoints_per_pillar.float()           
            z_points = torch.where(point_in_blocks, pillars[:, :, 2], float('nan'))
            z_points_masked = torch.where(torch.isnan(z_points), torch.full_like(z_points, -float('inf')), z_points)
            z_max = torch.max(z_points_masked, dim=1)[0]
            z_points_masked = torch.where(torch.isnan(z_points), torch.full_like(z_points, float('inf')), z_points)
            z_min = torch.min(z_points_masked, dim=1)[0]
            delta_z = torch.where(torch.isnan(z_max - z_min), torch.zeros_like(z_max), z_max - z_min)         
            pillar_features.append(torch.stack([num_pts_block, delta_z], dim=1)) 
            
        pillar_features = torch.cat(pillar_features, dim=1) # (p1 + p2 + ... + pb, 2 * num_blocks)
        pillar_features = self.pillar_encoder(pillar_features) # (p1 + p2 + ... + pb, out_channel)
        
        # Fusion two features
        lidar_features = torch.cat([point_features, pillar_features], dim=1) # (p1 + p2 + ... + pb, 2 * out_channel)
        lidar_features = self.pillar_point_encoder(lidar_features) # (p1 + p2 + ... + pb, out_channel)
        
        # Image feature encoding
        boundary_points = self.extract_boundary_points(pillars, npoints_per_pillar) # (p1 + p2 + ... + pb, 4, 3) np array

        points2image = torch.zeros((num_pillars, 4, 2), device=device)
        batch_idx = coors_batch[:, 0]
        for b in torch.unique(batch_idx):
            mask = batch_idx == b
            pts = boundary_points[mask.cpu().numpy()]  # lấy numpy array cho points_lidar2image
            calib = batch_calibs[b.item()]
            img_pts = points_lidar2image(pts, tr_velo_to_cam=calib['T'], r0_rect=calib['R'], P2=calib['P'])
            points2image[mask] = torch.from_numpy(img_pts).to(device)

        image_bboxes = bounding_bboxes(points2image.cpu().numpy())
        
        rois = self.bboxes_to_rois(image_bboxes, coors_batch, batch_image_map)
        roi_feats = torchvision.ops.roi_align(batch_image_map, rois, output_size=(3,3), spatial_scale=1, aligned=True)  # (p1 + p2 + ... + pb, out_channel, 3, 3)
        image_features = F.adaptive_avg_pool2d(roi_feats, (1,1)).squeeze(-1).squeeze(-1)  # (p1 + p2 + ... + pb, out_channel)
        
        # 1. scatter LiDAR BEV
        bev_lidar = self.scatter_features(
            lidar_features, coors_batch, batch_size,
            self.x_grid, self.y_grid, self.out_channel, device
        )

        # 2. scatter Image BEV
        bev_image = self.scatter_features(
            image_features, coors_batch, batch_size,
            self.x_grid, self.y_grid, self.out_channel, device
        )
        
        return bev_lidar, bev_image
    
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv (Expansion)
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # 3x3 depthwise conv
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # 1x1 pointwise conv (Projection)
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Backbone(nn.Module):
    def __init__(self, in_channel=64, out_channels=[64, 128, 256], layer_nums=[3, 3, 3], layer_strides=[1, 2, 2], expand_ratio=3):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)

        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(InvertedResidualBlock(in_channel, out_channels[i], stride=layer_strides[i], expand_ratio=expand_ratio))
            for _ in range(layer_nums[i] - 1):
                blocks.append(InvertedResidualBlock(out_channels[i], out_channels[i], stride=1, expand_ratio=expand_ratio))
            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))
            
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * out_channels[i], out_channels[i], kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels[i]),
                nn.ReLU(inplace=True)
            )
            for i in range(len(out_channels))
        ])
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, bev_lidar, bev_image):
        '''
        Args:
            bev_lidar [torch.tensor float32, (B, C, H, W)]: bev map of lidar features
            bev_image [torch.tensor float32, (B, C, H, W)]: bev map of image features
            
        Middle:
            lidar_conv_blocks [list of torch.tensor]: (B, C, H, W), (B, 2C, H/2, W/2), (B, 4C, H/4, W/4)
            image_conv_blocks [list of torch.tensor]: (B, C, H, W), (B, 2C, H/2, W/2), (B, 4C, H/4, W/4)
        
        Returns:
            fusion_conv_blocks [list of torch.tensor]: (B, C, H, W), (B, 2C, H/2, W/2), (B, 4C, H/4, W/4)         
        '''
        lidar_conv_blocks, image_conv_blocks, fusion_conv_blocks = [], [], []
        for i in range(len(self.multi_blocks)):
            bev_lidar = self.multi_blocks[i](bev_lidar)
            bev_image = self.multi_blocks[i](bev_image)
            
            lidar_conv_blocks.append(bev_lidar)
            image_conv_blocks.append(bev_image)
            
            fused = torch.cat([bev_lidar, bev_image], dim=1)
            fused = self.fusion_convs[i](fused)
            fusion_conv_blocks.append(fused)
                        
        return fusion_conv_blocks
    
class Neck(nn.Module):
    def __init__(self, in_channels=[64, 128, 256], out_channels=[128, 128, 128], upsample_strides=[1, 2, 4]):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(nn.ConvTranspose2d(in_channels[i], out_channels[i], upsample_strides[i], stride=upsample_strides[i], bias=False))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, conv_blocks):
        '''
        Args:
            conv_blocks [list of torch.tensor float32]: (B, C, H, W), (B, 2C, H/2, W/2), (B, 4C, H/4, W/4)    
        
        Returns:
            feature_map [torch.tensor float32, (B, 6C, H, W)]: Neck feature map
        '''
        feature_map = []
        for i in range(len(self.decoder_blocks)):
            deconv_blocks = self.decoder_blocks[i](conv_blocks[i]) 
            feature_map.append(deconv_blocks)
        feature_map = torch.cat(feature_map, dim=1)
        return feature_map
    
class Head(nn.Module):
    def __init__(self, in_channel, n_anchors, n_classes):
        super().__init__()
        
        self.conv_cls = nn.Conv2d(in_channel, n_anchors*n_classes, kernel_size=1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors*7, kernel_size=1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors*2, kernel_size=1)

        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, feature_map):
        '''
        Args:
            feature_map [torch.tensor float32, (B, 6C, H, W)]
        
        Returns:
            bbox_cls_pred [torch.tensor float32, (B, n_anchor*n_classes, H, W)]
            bbox_pred [torch.tensor float32, (B, n_anchor*7, H, W)]
            bbox_dir_cls_pred: [torch.tensor float32, (B, n_anchor*2, H, W)]
        '''
        bbox_cls_pred = self.conv_cls(feature_map)
        bbox_pred = self.conv_reg(feature_map)
        bbox_dir_cls_pred = self.conv_dir_cls(feature_map)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred
    
class Pillarloli(nn.Module):
    def __init__(
        self,
        nclasses=config['num_classes'], 
        voxel_size=config['voxel_size'],
        point_cloud_range=config['point_cloud_range'],
        max_points=config['max_points'],
        max_voxels=config['max_voxels'],
        shape=config['shape'],
        mean=config['mean'],
        std=config['std']
    ):
        super(Pillarloli, self).__init__()
        
        self.nclasses = nclasses
        
        self.image_stem = ImageStem(
            mean=mean,
            std=std,
            shape=shape,
            out_channel=64,
        )
        
        self.pillar_layer = PillarLayer(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_points=max_points,
            max_voxels=max_voxels,
        )
        
        self.pillar_encoder = PillarEncoder(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            num_blocks=config['num_blocks'],
            out_channel=64,
        )
        
        self.backbone = Backbone(
            in_channel=64,
            out_channels=[64, 128, 256],
            layer_nums=[3, 3, 3],
            layer_strides=[1, 2, 2],
            expand_ratio=3,
        )
        
        self.neck = Neck(
            in_channels=[64, 128, 256],
            out_channels=[128, 128, 128],
            upsample_strides=[1, 2, 4],
        )
        
        self.head = Head(
            in_channel=384,
            n_anchors=2,
            n_classes=nclasses
        )
        
        ranges = [[0, -20.48, -4, 30.72, 20.48, 1]]
        sizes = [[0.6, 0.6, 1.7]]
        rotations = [0, 1.57]
        
        self.anchors_generator = Anchors(ranges=ranges, sizes=sizes, rotations=rotations)

        self.assigners = [{'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35}]

        self.nms_pre = 500
        self.nms_thr = 0.1
        self.score_thr = 0.3
        self.max_num = 30

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        '''
        bbox_cls_pred: (n_anchors*3, 248, 216) 
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return: 
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, ) 
        '''
        # 0. pre-process 
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)
        
        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        # 1. obtain self.nms_pre bboxes based on scores
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        # 2. decode predicted offsets to bboxes
        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        # 3. nms
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                 bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                 bbox_pred[:, 6:]], dim=-1) # (n_anchors, 5)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            # 3.1 filter bboxes with scores below self.score_thr
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]
            
            # 3.2 nms core
            keep_inds = nms_cuda(boxes=cur_bbox_pred2d, 
                                 scores=cur_bbox_cls_pred, 
                                 thresh=self.nms_thr, 
                                 pre_maxsize=None, 
                                 post_max_size=None)

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(cur_bbox_pred) # [-pi, 0]
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * torch.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        # 4. filter some bboxes if bboxes number is above self.max_num
        if len(ret_bboxes) == 0:
            result = {
                'lidar_bboxes': np.zeros((0, 7), dtype=np.float32),
                'labels': np.zeros((0,), dtype=np.int64),
                'scores': np.zeros((0,), dtype=np.float32)
            }
            return result
        
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        result = {
            'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
            'labels': ret_labels.detach().cpu().numpy(),
            'scores': ret_scores.detach().cpu().numpy()
        }
        return result


    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        '''
        bbox_cls_pred: (bs, n_anchors*n_classes, 248, 216) 
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return: 
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ] 
        '''
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i],
                                                      bbox_pred=bbox_pred[i], 
                                                      bbox_dir_cls_pred=bbox_dir_cls_pred[i], 
                                                      anchors=batched_anchors[i])
            results.append(result)
        return results
    
    def forward(
        self, 
        batched_pts, 
        mode='test', 
        batched_gt_bboxes=None, 
        batched_gt_labels=None, 
        batched_images=None, 
        batched_calibs=None
    ):
        batch_size = len(batched_pts)
        
        image_stem = self.image_stem(batched_images)
        
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        
        bev_lidar, bev_image = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar, image_stem, batched_calibs, batch_size)
        
        conv_blocks = self.backbone(bev_lidar, bev_image)
        
        feature_map = self.neck(conv_blocks)
        
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(feature_map)
        
        # anchors
        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]
        
        if mode == 'train':
            anchor_target_dict = anchor_target(batched_anchors=batched_anchors, 
                                               batched_gt_bboxes=batched_gt_bboxes, 
                                               batched_gt_labels=batched_gt_labels, 
                                               assigners=self.assigners,
                                               nclasses=self.nclasses)
            
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        elif mode == 'val':
            anchor_target_dict = anchor_target(batched_anchors=batched_anchors, 
                                               batched_gt_bboxes=batched_gt_bboxes, 
                                               batched_gt_labels=batched_gt_labels, 
                                               assigners=self.assigners,
                                               nclasses=self.nclasses)
            return results

        elif mode == 'test':
            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred, 
                                                bbox_pred=bbox_pred, 
                                                bbox_dir_cls_pred=bbox_dir_cls_pred, 
                                                batched_anchors=batched_anchors)
            return results
        else:
            raise ValueError   
        
        