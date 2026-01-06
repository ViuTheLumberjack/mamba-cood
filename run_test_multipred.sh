#! /bin/bash
GPU_ID=$1

if [ -z "$GPU_ID" ]; then
    GPU_ID=0
fi

echo "Using GPU ID: $GPU_ID"

EPOCH=70
TITLE="bimamba2-512-multipred-smoothl1-encoding-cls-3-400ms-2drop-noRTE"
echo "Evaluating ${TITLE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_$TITLE_$EPOCH" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_$TITLE_$EPOCH" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_$TITLE_$EPOCH" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml

EPOCH=80
TITLE="mamba2-256-multipred-smoothl1-encoding-cls-3-400ms-2drop-noRTE"
echo "Evaluating ${TITLE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_$TITLE_$EPOCH" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_$TITLE_$EPOCH" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_$TITLE_$EPOCH" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml

EPOCH=80
TITLE="bimamba2-256-multipred-charb-encoding-cls-1-400ms-2drop-noRTE"
echo "Evaluating ${TITLE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_$TITLE_$EPOCH" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_$TITLE_$EPOCH" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_$TITLE_$EPOCH" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml

EPOCH=66
TITLE="bimamba2-512-multipred-charb-encoding-cls-1-400ms-2drop-noRTE"
echo "Evaluating ${TITLE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_$TITLE_$EPOCH" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_$TITLE_$EPOCH" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_$TITLE_$EPOCH" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml

EPOCH=70
TITLE="bimamba2-768-multipred-charb-encoding-cls-1-400ms-2drop-noRTE"
echo "Evaluating ${TITLE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_$TITLE_$EPOCH" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_$TITLE_$EPOCH" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_$TITLE_$EPOCH" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml

EPOCH=72 
TITLE="bimamba2-256-multipred-charb-encoding-cls-1-400ms-2drop-4p-noRTE"
echo "Evaluating ${TITLE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_$TITLE_$EPOCH" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_$TITLE_$EPOCH" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_$TITLE_$EPOCH" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml

EPOCH=86 
TITLE="bimamba2-768-multipred-charb-encoding-cls-1-400ms-2drop-4p-noRTE"
echo "Evaluating ${TITLE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_$TITLE_$EPOCH" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_$TITLE_$EPOCH" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_$TITLE_$EPOCH" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml

EPOCH=70
TITLE="bimamba2-512-multipred-charb-encoding-cls-1-400ms-2drop-4p-noRTE"
echo "Evaluating ${TITLE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_$TITLE_$EPOCH" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_$TITLE_$EPOCH" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_$TITLE_$EPOCH" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
