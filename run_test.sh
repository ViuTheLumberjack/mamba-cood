
#choose delay time in config file ('MODEL_v2xset/v2x-vit/config_evaluation.yaml')

# CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit \
#                                                           --fusion_method 'intermediate'  \
#                                                           --name_output_result 'prova_validate_delay' \
#                                                           --split_dataset 'validate' \
#                                                           --module_delay \
#                                                           --specific_path 'MODEL_v2xset/v2x-vit/new_model/delay400ms_3dcnn/net_epoch89.pth' \
#                                                           --specific_epoch 89 \


#100ms TRAININGS
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'training_classic_delay100ms' --split_dataset 'training'
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'training_CNN3D_delay100ms' --split_dataset 'training' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/new_model/3dcnn_noFreezeheads_lr4_len3_deccomFT/net_epoch68.pth'  --specific_epoch 68 

