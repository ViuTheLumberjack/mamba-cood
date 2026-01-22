#script to generate features for v2xset dataset


CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_extract_feature.py --model_dir MODEL_v2xreal/v2x-vit --fusion_method 'intermediate' --split_dataset 'train' --name_yaml config.yaml
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_extract_feature.py --model_dir MODEL_v2xreal/v2x-vit --fusion_method 'intermediate' --split_dataset 'validate' --name_yaml config.yaml
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_extract_feature.py --model_dir MODEL_v2xreal/v2x-vit --fusion_method 'intermediate' --split_dataset 'test' --name_yaml config.yaml