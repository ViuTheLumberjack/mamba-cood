#! /bin/bash
GPU_ID=$1

if [ -z "$GPU_ID" ]; then
    GPU_ID=0
fi

predictions=(0 1 2 3)

echo "Using GPU ID: $GPU_ID"

EPOCH=80
TITLE="bimamba2-multpred2-512-smoothl1-encoding-cls-3-400ms-2drop-4past-noRTE"
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done

EPOCH=82
TITLE="bimamba2-multpred-512-encoding-cls-3-400ms-2drop-4past"
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done
