#choose delay time and other parameters in config file ('MODEL_v2xset/v2x-vit/config_training.yaml')
# [--model_dir  ${CHECKPOINT_FOLDER}] add for finetuning!

CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_feature.py --info 'add_info_of_the_exp' --hypes_yaml 'opencood/hypes_yaml/point_pillar_v2xvit.yaml' --half --model_dir 'MODEL_v2xset/v2x-vit'


