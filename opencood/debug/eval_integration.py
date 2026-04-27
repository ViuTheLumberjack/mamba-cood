# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from tqdm import tqdm
import sys

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt

def show_pred_gt(output_dict, batch_data, global_iteration, save_path=None):
    # print('ok')
    choose_ex = 0
    # print(output_dict['ego'].keys())
    record_len = batch_data['ego']['record_len'][choose_ex]
    id_data =  batch_data['ego']['id_data'][choose_ex]
    feature_base = batch_data['ego']['current_features'][:record_len]
    # feature_residual = output_dict['feature_residual'][choose_ex][:record_len]
    preds = output_dict['ego']['predictions'][:record_len]
    gts = batch_data['ego']['gt_features'][:record_len]
    # mask = output_dict['mask'][choose_ex][:record_len]
    time_delay = batch_data['ego']['time_delay'][choose_ex][:record_len]
    infra = batch_data['ego']['infra'][choose_ex][:record_len]
    velocity = batch_data['ego']['velocity'][choose_ex][:record_len]

    label_ag = ['ego']
    for inf in infra[1:]:
        if inf == 1:
            label_ag.append('inf')
        elif inf == 0:
            label_ag.append('av')

    for t in range(preds.shape[0]):
        pred = preds[t]
        gt = gts[t]
        B, C, H, W = pred.shape

        title_primary = f'example: {id_data}, {label_ag}, time step: {t}'
        titles_secondary = []
        for i in range(record_len):
            #round velocity in .3
            titles_secondary.append(f'{label_ag[i]}, delay: {time_delay[i].item()}, vel: {round(velocity[i].item(),4)}')
        #pad with 'padding' until 5
        titles_secondary += ['padding'] * (5 - record_len)

        #create image, sum in channels
        feature_base_image = feature_base.sum(1).detach().cpu()
        # feature_residual_image = feature_residual.sum(1).detach().cpu()
        pred_image = pred.sum(1).detach().cpu()
        gt_image = gt.sum(1).detach().cpu()
        
        #padding
        feature_base_image = torch.cat([feature_base_image, torch.zeros(5 - record_len, feature_base_image.shape[1], feature_base_image.shape[2])], dim=0)
        # feature_residual_image = torch.cat([feature_residual_image, torch.zeros(5 - record_len, feature_residual_image.shape[1], feature_residual_image.shape[2])], dim=0)
        pred_image = torch.cat([pred_image, torch.zeros(5 - record_len, H, W)], dim=0)
        gt_image = torch.cat([gt_image, torch.zeros(5 - record_len, gt_image.shape[1], gt_image.shape[2])], dim=0)

        # Constants
        num_rows = 5  # Number of rows
        H, W = feature_base_image.shape[1], feature_base_image.shape[2]  # Dimensions of each subplot (just for demo)
        # titles = ["base", "residual", "pred", "gt"]
        titles = ["base", "pred", "gt"]

        # images = [feature_base_image, feature_residual_image, pred_image, gt_image]
        images = [feature_base_image, pred_image, gt_image]


        # Create figure
        fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(10, 10))

        # Set main title
        fig.suptitle(title_primary, fontsize=16, fontweight='bold')

        for row in range(num_rows):

            for col in range(3):
                ax = axes[row, col]
                ax.imshow(images[col][row], cmap='viridis')

                min_value = images[col][row].min().item()
                max_value = images[col][row].max().item()

                title = titles[col] # Dummy image
                title = f'{title} min:{min_value:.2f} max:{max_value:.2f}'
                if title == 'gt':
                    title = f'{title}_{titles_secondary[row]}'
                ax.set_title(title, fontsize=10)
                ax.axis('off')  # Hide axes for clarity

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit titles

        #save in wandb
        plt.savefig(os.path.join(save_path, f'global_{global_iteration}_example_{id_data}_t{t}.png'))
        plt.close()

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, default='MODEL_v2xset/v2x-vit',
                        help='Continued training path')
    parser.add_argument('--name_yaml', default=None,
                help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    
    parser.add_argument('--mode', type=str, default='feature')
    parser.add_argument('--specific_path', type=str, default=None) #if None, it is loaded the original model     #'MODEL_v2xset/v2x-vit/delay400ms_3dcnn/net_epoch67.pth'
    parser.add_argument('--specific_epoch', type=int, default=None)   #70
    parser.add_argument('--split_dataset', type=str, default='validate')  #validate, test
    parser.add_argument('--freeze_heads', type=bool, default=False) 
    parser.add_argument('--name_output_result', type=str, default='prova')  #validate, test
    parser.add_argument('--len_past', type=int, default=2)  #validate, test
    parser.add_argument('--delay', type=int, default=None)
    
    parser.add_argument('--module_delay', action='store_true', default=False)
    parser.add_argument('--baseline', type=str, default='MODEL_v2xset/v2x-vit/delay400ms_3dcnn/net_epoch67.pth')
    #wo_backbone: jump the backbone part that generate the feature map and read it from the disk, saved previously
    #classic: use the backbone to generate the feature map
    parser.add_argument('--forward_type', type=str, default='wo_backbone')  #wo_backbone, classic

    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)

    print(opt)

    #add mode in hypes
    hypes['mode'] = opt.mode
    hypes['module_delay'] = opt.module_delay
    hypes['split_dataset'] = opt.split_dataset
    hypes['freeze_heads'] = opt.freeze_heads
    hypes['len_past'] = opt.len_past

    if opt.delay is not None:
        hypes['module_delay'] = True
        hypes['wild_setting']['async_overhead'] = int(opt.delay) * 100
        hypes['delay']['future_delay'] = opt.delay
        hypes['model']['args']['delay']['args']['future_delay'] = int(opt.delay) * 100

    print('module delay:', hypes['module_delay'])
    print('split dataset:', hypes['split_dataset'])

    if hypes['split_dataset'] == 'validate':
        hypes['validate_dir'] = 'v2xset/validate'
    elif hypes['split_dataset'] == 'test':
        hypes['validate_dir'] = 'v2xset/test'
    elif hypes['split_dataset'] == 'training':
        hypes['validate_dir'] = 'v2xset/train'

    # check if the code is runing with debug mode
    if sys.gettrace() is not None:
        num_workers = 0
    else:
        num_workers = 16

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # opencood_dataset = build_dataset(hypes, visualize=True, train=True)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=num_workers,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False, 
                             pin_memory=False,
                             drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes, module_delay=opt.module_delay)
    
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Loading Model from checkpoint: specific_path {opt.specific_path} and specific_epoch {opt.specific_epoch}')
    saved_path = opt.model_dir
    # _, model = train_utils.load_saved_model(saved_path, model, specific_path=opt.specific_path, specific_epoch=opt.specific_epoch)
    delay_model = torch.load(opt.specific_path, weights_only=True)

    # change all the delaymodel keys to module_delay.*
    delay_model = {f"module_delay.{k}": v for k, v in delay_model.items()}

    model.load_state_dict(delay_model, strict=False)
    
    # load standalone prediction module
    if opt.baseline is not None:
        baseline = torch.load(opt.baseline, weights_only=True)
        missing, unexpected = model.load_state_dict(baseline, strict=False)
        
    model.eval()

    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    for i, batch_data in tqdm(enumerate(data_loader)):
        # print(i)
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset,
                                                           forward_type=opt.forward_type)
            elif opt.fusion_method == 'intermediate':
                output_dict, pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset,
                                                                  forward_type=opt.forward_type,
                                                                  )
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            
            if i % 100 == 0:
                print(f'Processed {i} samples')
                img_path = os.path.join(opt.model_dir, 'wandb_visualization', str(opt.specific_epoch))
                os.makedirs(img_path, exist_ok=True)
                show_pred_gt(output_dict, batch_data, global_iteration=i, save_path=img_path)

            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego'][
                                                       'origin_lidar'][0],
                                                   i,
                                                   npy_save_path)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'],
                        vis_pcd,
                        mode='constant'
                        )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  opt.global_sort_detections,
                                  name_file = opt.name_output_result,
                                  opt=opt)

    model_timings = model.timings 
    if len(model_timings) >0:
        print(f'Average inference time per sample: {sum(model_timings)/len(model_timings):.2f} ms')

    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
