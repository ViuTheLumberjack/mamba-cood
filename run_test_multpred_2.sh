#! /bin/bash
GPU_ID=$1

if [ -z "$GPU_ID" ]; then
    GPU_ID=0
fi

predictions=(1 2 3 4)

echo "Using GPU ID: $GPU_ID"

# ignore the following models for now
if false; then
EPOCH=90
TITLE="bimamba2-512-3-encoding-cls-02drop-400ms-multpred4-singleres-noRTE"
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done

EPOCH=80
TITLE="bimamba2-512-3-encoding-cls-02drop-400ms-multpred-singleres-RTE"
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done

EPOCH=95
TITLE="bimamba2-512-3-encoding-cls-02drop-4past-400ms_noRTE"
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done

EPOCH=70
TITLE="bimamba2-512-1-encoding-cls-02drop-4past-400ms"
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    #CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
    #CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done

predictions=(4)

EPOCH=88
TITLE="bimamba2-512-1-encoding-cls-02drop-400ms-maskedl1-nolin-noRTE" ## Something wrong
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    #CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
    #CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done

predictions=(0 1 2 3 4)

EPOCH=88
TITLE="bimamba2-512-1-encoding-cls-02drop-400ms-maskedl1-nolin-RTE" ## Something wrong
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    #CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "validate_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'validate' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
    #CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "training_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'training' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth  --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done

EPOCH=84
TITLE="bimamba2-encoding-cls-2-02drop-400ms-multpred5" ## SGreat
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done

EPOCH=99
TITLE="bimamba2-encoding-cls-2-02drop-400ms-multpred5-noRTE" ## Great
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done

EPOCH=65
TITLE="bimamba2-encoding-cls-1-02drop-400ms-noego-noRTE" # very very bad
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done

EPOCH=66
TITLE="bimamba2-encoding-cls-1-02drop-400ms-noego" # vwery  bad
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done

EPOCH=76
TITLE="bimamba2-encoding-cls-3-nolin-02drop-400ms" # good
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done

EPOCH=88
TITLE="bimamba2-encoding-cls-3-nolin-02drop-400ms-noRTE" #interesting
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done

EPOCH=70
TITLE="mamba2-encoding-cls-3-02drop-400ms-multpred-noRTE" 
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done

EPOCH=68
TITLE="bimamba2-encoding-cls-1-nolin-02drop-400ms-multpred-RTE"
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done

EPOCH=68
TITLE="bimamba2-3-encoding-cls-02drop-400ms-multpred-noRTE"
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --len_past 4 --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done
fi

predictions=(0 1 2 3 4)

EPOCH=62
TITLE="bimamba2-encoding-cls-1-nolin-02drop-400ms-multpred-RTE"
echo "Evaluating ${TITLE}"
for PRED in "${predictions[@]}"; do
    echo "  with prediction horizon: ${PRED}ms"
    CUDA_VISIBLE_DEVICES=$GPU_ID python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit/TRAININGS/$TITLE --fusion_method 'intermediate' --name_output_result "test_${TITLE}_epoch${EPOCH}_pred${PRED}ms" --split_dataset 'test' --module_delay  --specific_path MODEL_v2xset/v2x-vit/TRAININGS/$TITLE/net_epoch${EPOCH}.pth --specific_epoch $EPOCH --name_yaml config_training_hypes.yaml --delay $PRED
done
