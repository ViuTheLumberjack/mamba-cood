# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import statistics
import yaml

import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from opencood.tools import train_utils, inference_utils
from opencood.utils import eval_utils

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils

import wandb
import sys

import matplotlib.pyplot as plt
import numpy as np


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, default='opencood/hypes_yaml/point_pillar_v2xvit_delay.yaml', help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='MODEL_v2xset/v2x-vit', help='Continued training path')
    parser.add_argument('--name_yaml', default='config_training.yaml', help='name of yaml with parameters to train')
    parser.add_argument("--half", action='store_true', default=True, help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--info', type=str, default='prova123545', help='name of the experiment')
    parser.add_argument('--mode', type=str, default='feature')
    parser.add_argument('--split_dataset', type=str, default='validate')  #validate, test
    parser.add_argument('--module_delay', type=bool, default=True)  #False: no delay, True: delay
    parser.add_argument('--forward_type', type=str, default='wo_backbone')  #wo_backbone: the feature is given by disk saved previously, classic: as the original
    parser.add_argument('--freeze_heads', type=bool, default=False) #False: train the heads, True: freeze the heads
    parser.add_argument('--len_past', type=str, default=2)  #validate, test

    parser.add_argument('--global_sort_detections', action='store_true',
                    help='whether to globally sort detections by confidence score.'
                            'If set to True, it is the mainstream AP computing method,'
                            'but would increase the tolerance for FP (False Positives).')
    opt = parser.parse_args()
    return opt


def show_pred_gt(output_dict, batch_data, global_iteration):
    # print('ok')
    choose_ex = 0
    record_len = batch_data['ego']['record_len'][choose_ex]
    id_data =  batch_data['ego']['id_data'][choose_ex]
    feature_base = batch_data['ego']['current_features'][:record_len]
    # feature_residual = output_dict['feature_residual'][choose_ex][:record_len]
    pred = output_dict['feature_output'][:record_len]
    gt = batch_data['ego']['gt_features'][:record_len]
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

    title_primary = f'example: {id_data}, {label_ag}'
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
    pred_image = torch.cat([pred_image, torch.zeros(5 - record_len, pred_image.shape[1], pred_image.shape[2])], dim=0)
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
            # ax.imshow(np.random.rand(H, W), cmap='viridis') 
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
    wandb.log({"train/images": [wandb.Image(plt)], "it": global_iteration})
    plt.close()
    

def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    hypes['mode'] = opt.mode
    hypes['split_dataset'] = opt.split_dataset
    hypes['validate_dir'] = '/equilibrium/datasets/V2X/v2xset/validate'
    hypes['module_delay'] = opt.module_delay
    hypes['freeze_heads'] = opt.freeze_heads
    hypes['len_past'] = opt.len_past
    info = opt.info

    if sys.gettrace() is not None:
        mode_wandb = 'disabled'
        num_workers = 0
    else:
        num_workers = 8
        mode_wandb = 'online'

    # #create folder in os.path.join(saved_path, 'prova'
    # if not os.path.exists(os.path.join(opt.model_dir, info)):
    #     os.makedirs(os.path.join(opt.model_dir, info))

    # multi_gpu_utils.init_distributed_mode(opt)
    opt.distributed = False

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=num_workers,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=num_workers,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=num_workers,   #8
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=1, #hypes['train_params']['batch_size'],
                                num_workers=num_workers,    #
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes, module_delay=opt.module_delay)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)


    info_name = opt.info
    # dict_args = vars(self.args)
    wandb.init(project='opencood', notes="", name=info_name, save_code=True, mode=mode_wandb, config=hypes)
    wandb.define_metric("epoch")
    wandb.define_metric("it")
    wandb.define_metric("train/*", step_metric="it")
    wandb.define_metric("val/*", step_metric="epoch")


    ###########################################

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    global_iteration = 0
    ap_70_best = 0.0
    first_epoch = True
    #save opt parameters in local disk
    saved_path_dir = os.path.join(saved_path, 'TRAININGS')
    #create folder
    if not os.path.exists(os.path.join(saved_path_dir, info)):
        os.makedirs(os.path.join(saved_path_dir, info))
    with open(os.path.join(saved_path_dir, info, 'config_training_hypes.yaml'), 'w') as outfile:
        yaml.dump(hypes, outfile, default_flow_style=False)
    #save also opt
    with open(os.path.join(saved_path_dir, info, 'config_training_opt.yaml'), 'w') as outfile:
        yaml.dump(vars(opt), outfile, default_flow_style=False)
    

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        print('epoch %d' % epoch)
        if first_epoch == False:
            # todo: put the scheduler or not
            # if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            #     scheduler.step(epoch)
            # if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            #     scheduler.step_update(epoch * num_steps + 0)
            for param_group in optimizer.param_groups:
                print('learning rate %.7f' % param_group["lr"])

            #wandb for lr
            wandb.log({"lr": param_group["lr"], "epoch": epoch})

            if opt.distributed:
                sampler_train.set_epoch(epoch)

            pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

            model.train()
            for i, batch_data in tqdm.tqdm(enumerate(train_loader)):
                #to try the training pipeline
                # if i == 20:
                #     break

                print('step training %d' % i)
                # the model will be evaluation mode during validation

                model.zero_grad()
                optimizer.zero_grad()

                batch_data = train_utils.to_device(batch_data, device)

                # case1 : late fusion train --> only ego needed,
                # and ego is random selected
                # case2 : early fusion train --> all data projected to ego
                # case3 : intermediate fusion --> ['ego']['processed_lidar']
                # becomes a list, which containing all data from other cavs
                # as well
                if not opt.half:
                    ouput_dict = model.forward_feature_wo_backbone(batch_data['ego'])
                    # first argument is always your output dictionary,
                    # second argument is always your label dictionary.
                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
                    # final_loss = criterion(ouput_dict, batch_data) 
                else:
                    with torch.cuda.amp.autocast():
                        ouput_dict = model.forward_feature_wo_backbone(batch_data['ego'])
                        final_loss = criterion(ouput_dict, batch_data) 
                
                #set in wandb
                loss_feature = criterion.loss_dict['loss_feature']
                conf_loss = criterion.loss_dict['conf_loss']
                reg_loss = criterion.loss_dict['reg_loss']
                wandb.log({"train/loss": final_loss.item(), "it": global_iteration})
                wandb.log({"train/loss_feature": loss_feature.item(), "it": global_iteration})
                wandb.log({"train/conf_loss": conf_loss.item(), "it": global_iteration})
                wandb.log({"train/reg_loss": reg_loss.item(), "it": global_iteration})

                #show in wandb the 2d feature maps: current -> pred - gt
                if global_iteration % 20 == 0:
                    show_pred_gt(ouput_dict, batch_data, global_iteration)

                criterion.logging(epoch, i, len(train_loader), pbar=pbar2)
                pbar2.update(1)

                if not opt.half:
                    final_loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(final_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                # todo: put the scheduler or not
                # if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                #     scheduler.step_update(epoch * num_steps + i)

                global_iteration += 1

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                # os.path.join(saved_path, info, 'net_epoch%d.pth' % (epoch + 1)))
                os.path.join(saved_path_dir, info, 'net_epoch%d.pth' % (epoch)))
            #save config files in local

                

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            # Create the dictionary for evaluation.
            # also store the confidence score for each prediction
            result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                        0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}


            with torch.no_grad():
                for i, batch_data in tqdm.tqdm(enumerate(val_loader)):
                    # if i == 20:
                    #     break
                    print('validation step %d' % i)
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    # ouput_dict, pred_box_tensor, pred_score, gt_box_tensor = model.forward_feature_wo_backbone(batch_data['ego'])

                    ouput_dict, pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_intermediate_fusion(batch_data,
                                                                                        model,
                                                                                        opencood_validate_dataset,
                                                                                        forward_type=opt.forward_type)


                    final_loss = criterion(ouput_dict['ego'], batch_data) 
                    valid_ave_loss.append(final_loss.item())

                    #new
                    eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3)
                    eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5)
                    eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7)
                    
                ap_30, ap_50, ap_70 = eval_utils.eval_final_results(result_stat,opt.model_dir, opt.global_sort_detections, get_result=True)
                print('The Average Precision at IOU 0.3 is %.3f, '
                'The Average Precision at IOU 0.5 is %.3f, '
                'The Average Precision at IOU 0.7 is %.3f' % (ap_30, ap_50, ap_70))

                #log in wandb
                wandb.log({"val/ap_30": ap_30, "epoch": epoch})
                wandb.log({"val/ap_50": ap_50, "epoch": epoch})
                wandb.log({"val/ap_70": ap_70, "epoch": epoch})

                if ap_70 > ap_70_best:
                    ap_70_best = ap_70
                    print('The best Average Precision at IOU 0.7 is %.3f' % (ap_70_best))
                    # save the model
                    torch.save(model_without_ddp.state_dict(),
                        os.path.join(saved_path_dir, info, 'best_net.pth'))
                    # wandb.save(os.path.join(saved_path, info, 'best_net.pth'))

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch, valid_ave_loss))
            # writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)
            wandb.log({"val/loss": valid_ave_loss, "epoch": epoch})
            first_epoch = False

    print('Training Finished, checkpoints saved to %s' % saved_path_dir)


if __name__ == '__main__':
    main()
