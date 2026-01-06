#! /bin/bash
GPU_ID=$1

if [ -z "$GPU_ID" ]; then
    GPU_ID=0
fi

echo "Using GPU ID: $GPU_ID"

EPOCH=65
TITLE="mamba2-encoding-cls-3-200ms"
echo "Evaluating ${TITLE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_$TITLE_$EPOCH" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_$TITLE_$EPOCH" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_$TITLE_$EPOCH" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
