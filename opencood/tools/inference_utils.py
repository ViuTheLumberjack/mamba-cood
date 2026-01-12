# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy
import h5py

def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_early_fusion(batch_data, model, dataset, forward_type='classic'):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']

    if forward_type == 'classic':
        output_dict['ego'] = model(cav_content)
    elif forward_type == 'wo_backbone':
        output_dict['ego'] = model.forward_feature_wo_backbone(cav_content, inference=False)
    
    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)

    return output_dict, pred_box_tensor, pred_score, gt_box_tensor

def inference_early_fusion_extract(batch_data, model, dataset, split_dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']

    output_dict = model.extract_feature(cav_content)
    id_data = output_dict['id_data']
    feature = output_dict['spatial_features_2d']
    data_example = batch_data['ego']['data_example'][0]
    scenario_index = data_example['scenario_index']
    timestep = data_example['timestamp_key']
    cav_total = batch_data['ego']['cav_total'][0]

    for i_cav, cav in enumerate(cav_total):
        folder_path = f'FEATURE_SAVED2/feature_saved_f2f_scenarioTS_agents_folders_2/{split_dataset}/{scenario_index}/{timestep}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        try:
            np.savez_compressed(f'{folder_path}/{cav}', features=feature.half().cpu().numpy()[i_cav])
        except:
            print(f'Error saving {folder_path}/{cav}')
            # continue
        # temp = feature.half().cpu().numpy()[i_cav]


    # if scenario_index == 0 and timestep == '000188':
    #     print('ok')

    # np.savez_compressed(f'FEATURE_SAVED/feature_saved_f2f_scenarioTS/{split_dataset}/features_sc{scenario_index}_ts{timestep}', features=feature.half().cpu().numpy())

    #save cat_total in the npy file

    # #save for each scenario and for each cav
    # feature = feature.half().cpu().numpy()
    # for i_cav, cav in enumerate(cav_total):
    #     feature_single = feature[i_cav]
    #     #folder for each scenario
    #     if not os.path.exists(f'FEATURE_SAVED/feature_saved_f2f/{split_dataset}/{scenario_index}'):
    #         os.makedirs(f'FEATURE_SAVED/feature_saved_f2f/{split_dataset}/{scenario_index}')
    #     #folder for each cav in each scenario
    #     if not os.path.exists(f'FEATURE_SAVED/feature_saved_f2f/{split_dataset}/{scenario_index}/{cav}'):
    #         os.makedirs(f'FEATURE_SAVED/feature_saved_f2f/{split_dataset}/{scenario_index}/{cav}')
    #     folder_final = f'FEATURE_SAVED/feature_saved_f2f/{split_dataset}/{scenario_index}/{cav}'
    #     np.savez_compressed(f'{folder_final}/features_{timestep}', features=feature_single)



    #
    #
    # with h5py.File(f'feature_saved/train/features_{id_data}_tris', "w") as f:
    #     f.create_dataset("features", data=feature.cpu().numpy(), compression="gzip")


    return output_dict


def inference_intermediate_fusion(batch_data, model, dataset, forward_type='classic'):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return inference_early_fusion(batch_data, model, dataset, forward_type)

def inference_intermediate_fusion_extract(batch_data, model, dataset, split_dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return inference_early_fusion_extract(batch_data, model, dataset, split_dataset)


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy_test' % timestamp), gt_np)





    #save feature
    # torch.save(feature.half(), f'feature_saved/train/features_{id_data}_first.pt', _use_new_zipfile_serialization=True)
    # torch.save(feature.half(), f'feature_saved/train/features_{id_data}_second.pt', pickle_protocol=4)
    # print('ok')
    # np.savez_compressed(f'feature_saved/train/features_{id_data}_bis', features=feature.half().cpu().numpy())


    # np.savez_compressed(f'feature_saved/{split_dataset}/features_{id_data}', features=feature.half().cpu().numpy())
    # np.savez_compressed(f'feature_saved_compressed/{split_dataset}/features_{id_data}', features=feature.half().cpu().numpy())


    # np.savez_compressed(f'FEATURE_SAVED/feature_saved_f2f/{split_dataset}/features_{id_data}', features=feature.half().cpu().numpy())