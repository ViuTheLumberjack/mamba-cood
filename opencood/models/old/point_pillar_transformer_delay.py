import torch
import torch.nn as nn
import os

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.fuse_utils import regroup
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.v2xvit_basic import V2XTransformer

from opencood.models.sub_modules.cnn_delay import DelayModule

# U-Net Architecture for Feature Maps [B, 256, 48, 176]
class UNet(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.up1 = self.upconv(1024, 512)
        self.dec1 = self.conv_block(1024, 512)

        self.up2 = self.upconv(512, 256)
        self.dec2 = self.conv_block(512, 256)

        self.up3 = self.upconv(256, 128)
        self.dec3 = self.conv_block(256, 128)

        self.up4 = self.upconv(128, 64)
        self.dec4 = self.conv_block(128, 64)

        # Output
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """Basic Conv Block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        """Upsampling Layer"""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, feat_delay):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder with Skip Connections
        up1 = self.up1(bottleneck)
        dec1 = self.dec1(torch.cat([up1, enc4], dim=1))

        up2 = self.up2(dec1)
        dec2 = self.dec2(torch.cat([up2, enc3], dim=1))

        up3 = self.up3(dec2)
        dec3 = self.dec3(torch.cat([up3, enc2], dim=1))

        up4 = self.up4(dec3)
        dec4 = self.dec4(torch.cat([up4, enc1], dim=1))

        return self.final(dec4)


class PointPillarTransformer(nn.Module):
    def __init__(self, args):
        super(PointPillarTransformer, self).__init__()
        self.args = args

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = V2XTransformer(args['transformer'])

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)
        
        #todo: new thing
        # self.module_delay()
        # print('ok')
        # self.cnn_delay = DelayModule(259, 256)
        self.unet_delay = UNet(256, 256)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward_feature(self, data_dict):
        id_data = data_dict['id_data'][0]

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        # B, max_cav, 3(dt dv infra), 1, 1
        prior_encoding = data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features = batch_dict['spatial_features']
        spatial_features_2d = batch_dict['spatial_features_2d']
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        
        #NEW
      

        # N, C, H, W -> B,  L, C, H, W
        regroup_feature, mask = regroup(spatial_features_2d, record_len, self.max_cav)
        sp_base_group = regroup_feature.clone()
        # prior encoding added
        prior_encoding = prior_encoding.repeat(1, 1, 1, regroup_feature.shape[3], regroup_feature.shape[4])
        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)

        #new
        feature_residual = self.cnn_delay(regroup_feature.view(-1, 259, 48, 176)).view(-1, self.max_cav, 256, 48, 176)

        feature_total = feature_residual
        # feature_total = sp_base_group + feature_residual

        # b l c h w -> b l h w c
        # regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        # # transformer fusion
        # fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        # # b h w c -> b c h w
        # fused_feature = fused_feature.permute(0, 3, 1, 2)

        # psm = self.cls_head(fused_feature)
        # rm = self.reg_head(fused_feature)

        output_dict = {'feature_output': feature_total, 'mask': mask, 'feature_base': sp_base_group}

        return output_dict
    
    def forward(self, data_dict):
        id_data = data_dict['id_data'][0]

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        # B, max_cav, 3(dt dv infra), 1, 1
        prior_encoding = data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features = batch_dict['spatial_features']
        spatial_features_2d = batch_dict['spatial_features_2d']
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d, spatial_features_2d_end = self.naive_compressor.forward_extract(spatial_features_2d)

        #regroup_feature: [1, 5, 259, 48, 176]
        if self.args['cnn_delay']:
            # regroup_feature = self.cnn_delay(regroup_feature.view(-1, 259, 48, 176)).view(-1, self.max_cav, 256, 48, 176)
            # regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)
            regroup_feature = self.unet_delay(regroup_feature.view(-1, 256, 48, 176)).view(-1, self.max_cav, 256, 48, 176)

        # N, C, H, W -> B,  L, C, H, W
        regroup_feature, mask = regroup(spatial_features_2d, record_len, self.max_cav)
        # prior encoding added
        prior_encoding = prior_encoding.repeat(1, 1, 1, regroup_feature.shape[3], regroup_feature.shape[4])
        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)



        # b l c h w -> b l h w c 
        #[1, 5, 48, 176, 259]
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        # transformer fusion
        fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        # b h w c -> b c h w
        fused_feature = fused_feature.permute(0, 3, 1, 2)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict

    def extract_feature(self, data_dict):
        id_data = data_dict['id_data'][0]

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        # B, max_cav, 3(dt dv infra), 1, 1
        prior_encoding = data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features = batch_dict['spatial_features']
        spatial_features_2d = batch_dict['spatial_features_2d']
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d, x_enc = self.naive_compressor.forward_extract(spatial_features_2d)

        # output_dict = {'spatial_features_2d': spatial_features_2d, 'id_data': id_data}
        output_dict = {'spatial_features_2d': x_enc, 'id_data': id_data}



        # # N, C, H, W -> B,  L, C, H, W
        # regroup_feature, mask = regroup(spatial_features_2d, record_len, self.max_cav)
        # # prior encoding added
        # prior_encoding = prior_encoding.repeat(1, 1, 1, regroup_feature.shape[3], regroup_feature.shape[4])
        # regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)
        #
        # # b l c h w -> b l h w c
        # regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        # # transformer fusion
        # fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        # # b h w c -> b c h w
        # fused_feature = fused_feature.permute(0, 3, 1, 2)
        #
        # psm = self.cls_head(fused_feature)
        # rm = self.reg_head(fused_feature)
        #
        # output_dict = {'psm': psm,
        #                'rm': rm}

        return output_dict
