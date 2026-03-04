#! /bin/bash
GPU_ID=$1

if [ -z "$GPU_ID" ]; then
    GPU_ID=0
fi

echo "Using GPU ID: $GPU_ID"

if false; then
EPOCH=65
TITLE="mamaba2-encoding-cls-3-100ms"
echo "Evaluating ${TITLE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_$TITLE_$EPOCH" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_$TITLE_$EPOCH" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_$TITLE_$EPOCH" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml

EPOCH=86
TITLE="mamba2-encoding-cls-3-02drop-0ms_noRTE2"
echo "Evaluating ${TITLE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_$TITLE_$EPOCH" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml

EPOCH=82
TITLE="mamba2-encoding-cls-3-02drop-100ms_noRTE2"
echo "Evaluating ${TITLE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_$TITLE_$EPOCH" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
fi

for EPOCH in 76 78 80 82 84 86; do
    TITLE="mamba2-encoding-cls-3-02drop-100ms_noRTE2" ## Great
    echo "Evaluating ${TITLE}"
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred100ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
done

if false; then
EPOCH=88
TITLE="mamba2-encoding-cls-3-02drop-200ms_noRTE2"
echo "Evaluating ${TITLE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_$TITLE_$EPOCH" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml

EPOCH=88
TITLE="mamba2-encoding-cls-3-02drop-300ms_noRTE2"
echo "Evaluating ${TITLE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_$TITLE_$EPOCH" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
fi