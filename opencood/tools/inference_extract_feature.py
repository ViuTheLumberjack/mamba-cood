# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from tqdm import tqdm
import sys
from itertools import islice

import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, default='MODEL_v2xset/v2x-vit',
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='late, early or intermediate')
    parser.add_argument('--name_yaml', default='config_extract.yaml',
                    help='Continued training path')
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
    parser.add_argument('--mode', type=str, default='no_feature')
    parser.add_argument('--split_dataset', type=str, default='train')

    opt = parser.parse_args()
    return opt

def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'
    
    if sys.gettrace() is not None:
        num_workers = 0
    else:
        num_workers = 8

    hypes = yaml_utils.load_yaml(None, opt)
    #add mode in hypes
    hypes['mode'] = opt.mode
    print('Dataset Building')

    if opt.split_dataset == 'train':
        split_dataset = 'train'
        print("split dataset: Train")
        opt.split_dataset = True
    elif opt.split_dataset == 'validate':
        split_dataset = 'validate'
        hypes['validate_dir'] = '/equilibrium/datasets/V2X/v2xset/validate'
        print("split dataset: validate")
        opt.split_dataset = False
    elif opt.split_dataset == 'test':
        split_dataset = 'test'
        hypes['validate_dir'] = '/equilibrium/datasets/V2X/v2xset/test'
        print("split dataset: Test")
        opt.split_dataset = False

    opencood_dataset = build_dataset(hypes, visualize=True, train=opt.split_dataset)
    print(f"{len(opencood_dataset)} samples found.")

    #todo
    start_index = 5032  # start from the middle
    subset = Subset(opencood_dataset, list(range(start_index, len(opencood_dataset))))
    # dataloader = DataLoader(subset, batch_size=your_batch_size, shuffle=False)
    data_loader = DataLoader(subset,
                            batch_size=1,
                            num_workers=num_workers,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)


    # data_loader = DataLoader(opencood_dataset,
    #                          batch_size=1,
    #                          num_workers=num_workers,
    #                          collate_fn=opencood_dataset.collate_batch_test,
    #                          shuffle=False,
    #                          pin_memory=False,
    #                          drop_last=False)
    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    for i, batch_data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            output_dict = inference_utils.inference_intermediate_fusion_extract(batch_data, model, opencood_dataset, split_dataset)


if __name__ == '__main__':
    main()
