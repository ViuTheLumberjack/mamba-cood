#! /bin/bash
GPU_ID=$1

if [ -z "$GPU_ID" ]; then
    GPU_ID=0
fi

echo "Using GPU ID: $GPU_ID"

# my baseline: v2x-vit, intermediate fusion, classic delay compensation, 400ms delay (1-4 past frames)
if false; then
echo "Evaluating v2x-vit with classic delay compensation (400ms delay) 1 len past"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'training_classic_delay400ms' --split_dataset 'training'
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'validate_classic_delay400ms' --split_dataset 'validate'
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'test_classic_delay400ms' --split_dataset 'test'

# MAMBA TRAININGS WITH 400MS DELAY (Not so fair because of no residual connections in the delay module)
echo "Evaluating v2x-vit with 2 MAMBA block delay compensation with residual connections (400ms delay) 2 len past"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/mamba2-encoding-cls-2-residual --fusion_method 'intermediate' --name_output_result 'training_mamba2_encoding_cls_2_res_400ms' --split_dataset 'training' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/TRAININGS/mamba2-encoding-cls-2-residual/net_epoch70.pth'  --specific_epoch 70 --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/mamba2-encoding-cls-2-residual --fusion_method 'intermediate' --name_output_result 'validate_mamba2_encoding_cls_2_res_400ms' --split_dataset 'validate' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/TRAININGS/mamba2-encoding-cls-2-residual/net_epoch70.pth'  --specific_epoch 70 --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/mamba2-encoding-cls-2-residual --fusion_method 'intermediate' --name_output_result 'test_mamba2_encoding_cls_2_res_400ms' --split_dataset 'test' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/TRAININGS/mamba2-encoding-cls-2-residual/net_epoch70.pth'  --specific_epoch 70 --name_yaml config_training_hypes.yaml

fi

echo "3dCNN delay compensation evaluations"
# 3dCNN 
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'validate_CNN3D_delay400ms_89' --split_dataset 'validate' --module_delay --specific_path 'MODEL_v2xset/v2x-vit/TRAININGS/delay400ms_3dcnn/net_epoch89.pth' --specific_epoch 89
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'test_CNN3D_delay400ms_89' --split_dataset 'test' --module_delay --specific_path 'MODEL_v2xset/v2x-vit/TRAININGS/delay400ms_3dcnn/net_epoch89.pth' --specific_epoch 89
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'training_CNN3D_delay400ms_89' --split_dataset 'training' --module_delay --specific_path 'MODEL_v2xset/v2x-vit/TRAININGS/delay400ms_3dcnn/net_epoch89.pth' --specific_epoch 89
