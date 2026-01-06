import torch
import torch.nn as nn
import os

from torch.utils.checkpoint import checkpoint

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.fuse_utils import regroup
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.v2xvit_basic import V2XTransformer

# from opencood.models.delay.cnn_delay import DelayModule
# from opencood.models.delay.delay_film import FeatureModifier
# from opencood.models.delay.delay_local_att import AttentionBasedModifier
# from opencood.models.delay.delay_uTransformer import UTransformerModifier
# from opencood.models.delay.delay_convlstm import FutureFramePredictor
# from opencood.models.delay.delay_3dcnn import FutureFramePredictor
# from opencood.models.delay.delay_mamba import FutureFramePredictor
# from opencood.models.delay.delay_f2f import FutureFramePredictor
# from opencood.models.delay.delay_timesformer_style import FutureFramePredictor
# from opencood.models.delay.delay_transformer import FutureFramePredictor
import opencood.models.delay

#module F
import torch.nn.functional as F

class PointPillarTransformer(nn.Module):
    def __init__(self, args):
        super(PointPillarTransformer, self).__init__()
        self.args = args
        # args['lidar_range'] = [-200.8, -88.4, -6, 200.8, 88.4, 3]

        self.freeze_heads = args['freeze_heads']
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

        self.module_delay_flag = args['module_delay']
        if self.module_delay_flag:
            self.module_delay = opencood.models.delay.build_delay_module(
                args['delay']
            )

        # not enough memory
        self.all_preds = True #len(args['delay']['args']['future_delay_list']) > 1

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

        # todo, put finetuning
        if self.freeze_heads:
            for p in self.cls_head.parameters():
                p.requires_grad = False
            for p in self.reg_head.parameters():
                p.requires_grad = False

            #freeze fusion_net
            for p in self.fusion_net.parameters():
                p.requires_grad = False
                
        print('loaded')

    def forward_feature_wo_backbone(self, data_dict, inference=False):
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        # B, max_cav, 3(dt dv infra), 1, 1
        prior_encoding = data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)

        # read from saved feature
        feature_saved = data_dict['current_features']

        if self.module_delay_flag:
            past_feature = data_dict['past_features']    
            # concatenate along time dimension        
            total_feature = torch.cat([past_feature, feature_saved.unsqueeze(1)], dim=1)
            
            # Use the future frame predictor
            feature_encoded, predictions = self.module_delay(total_feature)

            # IMP2: residuals
            predictions = predictions + feature_saved
            feature_encoded = feature_encoded + feature_saved
        else:
            feature_encoded = feature_saved

        if not self.all_preds:
            spatial_features_2d = self.naive_compressor.decoder(feature_encoded)
            regroup_feature, mask = regroup(spatial_features_2d, record_len, self.max_cav)
            
            # prior encoding added
            prior_encoding_ = prior_encoding.repeat(1, 1, 1, regroup_feature.shape[3], regroup_feature.shape[4])
            regroup_feature = torch.cat([regroup_feature, prior_encoding_], dim=2)

            # b l c h w -> b l h w c 
            #[1, 5, 48, 176, 259]
            regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
            # transformer fusion
            fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
            # b h w c -> b c h w
            fused_feature = fused_feature.permute(0, 3, 1, 2)

            psm = self.cls_head(fused_feature).unsqueeze(1)
            rm = self.reg_head(fused_feature).unsqueeze(1)
        else:
            num_preds = predictions.shape[0]
            
            psm = []
            rm = []
            for i in range(num_preds):
                # Process one prediction at a time
                pred_i = predictions[i]  # (N, C, H, W)
                
                # Use checkpointing to save memory during backward
                spatial_features_2d = checkpoint(
                    self.naive_compressor.decoder, 
                    pred_i, 
                    use_reentrant=False
                )
                
                # Checkpoint the heavy fusion processing
                psm_i, rm_i = checkpoint(
                    self._process_single_prediction,
                    spatial_features_2d,
                    record_len,
                    prior_encoding,
                    spatial_correction_matrix,
                    use_reentrant=False
                )
                
                psm.append(psm_i)
                rm.append(rm_i)

        output_dict = {'psm': psm,
                    'rm': rm,
                    'feature_output': feature_encoded,
                    'predictions': predictions}

        return output_dict

    def _process_single_prediction(self, spatial_features_2d, record_len, prior_encoding, spatial_correction_matrix):
        """Helper function for checkpointing - processes a single prediction."""
        regroup_feature, mask = regroup(spatial_features_2d, record_len, self.max_cav)
        
        # prior encoding added
        prior_encoding_ = prior_encoding.repeat(1, 1, 1, regroup_feature.shape[3], regroup_feature.shape[4])
        regroup_feature = torch.cat([regroup_feature, prior_encoding_], dim=2)

        # b l c h w -> b l h w c
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        # transformer fusion
        fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        # b h w c -> b c h w
        fused_feature = fused_feature.permute(0, 3, 1, 2)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        return psm, rm
    
    def extract_feature(self, data_dict):
        id_data = data_dict['id_data'][0]

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        # B, max_cav, 3(dt dv infra), 1, 1
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
            spatial_features_2d, spatial_features_2d_enc = self.naive_compressor.forward_extract(spatial_features_2d)
            # print('ok')
            #todo: new thing
            # spatial_features_2d[spatial_features_2d < 0.1] = 0.0

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

