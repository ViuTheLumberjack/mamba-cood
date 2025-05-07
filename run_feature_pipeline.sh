#script to generate features for v2xset dataset


CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_extract_feature.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --split_dataset 'train'
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_extract_feature.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --split_dataset 'validate'
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_extract_feature.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --split_dataset 'test'