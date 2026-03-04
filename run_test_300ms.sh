#! /bin/bash
GPU_ID=$1

if [ -z "$GPU_ID" ]; then
    GPU_ID=0
fi

echo "Using GPU ID: $GPU_ID"

EPOCH=86
TITLE="mamba2-encoding-cls-3-02drop-300ms" ## Great
echo "Evaluating ${TITLE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml


EPOCH=82
TITLE="bimamba2-encoding-cls-02drop-200ms-noRTE" ## Great
echo "Evaluating ${TITLE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml

EPOCH=86
TITLE="bimamba2-encoding-cls-02drop-100ms-noRTE" ## Great
echo "Evaluating ${TITLE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml
