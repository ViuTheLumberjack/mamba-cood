
#choose delay time in config file ('MODEL_v2xset/v2x-vit/config.yaml')
#'MODEL_v2xset/v2x-vit/delay400ms_3dcnn/net_epoch89.pth'

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



CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'validate_classic_delay400ms' --split_dataset 'validate'
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'validate_CNN3D_delay400ms' --split_dataset 'validate' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/new_model/delay400ms_3dcnn/net_epoch89.pth'  --specific_epoch 89 

CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'test_classic_delay400ms' --split_dataset 'test'
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'test_CNN3D_delay400ms' --split_dataset 'test' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/new_model/delay400ms_3dcnn/net_epoch89.pth'  --specific_epoch 89 

#noRTE, 400ms
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'test_CNN3D_delay400ms_noRTE' --split_dataset 'test' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/TRAININGS/delay400ms_noRTE/best_net.pth'  --specific_epoch 89 
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'validate_CNN3D_delay400ms_noRTE' --split_dataset 'validate' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/TRAININGS/delay400ms_noRTE/best_net.pth'  --specific_epoch 89 

#delay 200ms with module delay
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'test_CNN3D_delay200ms' --split_dataset 'test' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/delay200ms/net_epoch72.pth'  --specific_epoch 72
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'validate_CNN3D_delay200ms' --split_dataset 'validate' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/delay200ms/net_epoch72.pth'  --specific_epoch 72



######################  400ms  ######################################## ok
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'test_classic_delay400ms' --split_dataset 'test'
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'validate_classic_delay400ms' --split_dataset 'validate'

CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'test_CNN3D_delay400ms_89' --split_dataset 'test' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/new_model/delay400ms_3dcnn/net_epoch89.pth'  --specific_epoch 89 
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'validate_CNN3D_delay400ms_89' --split_dataset 'validate' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/new_model/delay400ms_3dcnn/net_epoch89.pth'  --specific_epoch 89 

CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'test_CNN3D_delay400ms_73' --split_dataset 'test' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/new_model/delay400ms_3dcnn/net_epoch73.pth'  --specific_epoch 73
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'validate_CNN3D_delay400ms_73' --split_dataset 'validate' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/new_model/delay400ms_3dcnn/net_epoch73.pth'  --specific_epoch 73

CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'test_CNN3D_delay400ms_72' --split_dataset 'test' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/new_model/delay400ms_3dcnn/net_epoch72.pth'  --specific_epoch 72
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'validate_CNN3D_delay400ms_72' --split_dataset 'validate' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/new_model/delay400ms_3dcnn/net_epoch72.pth'  --specific_epoch 72

CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'test_CNN3D_delay400ms_89' --split_dataset 'test' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/TRAININGS/delay400ms_retry/best_net.pth'  --specific_epoch 89 
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'validate_CNN3D_delay400ms_89' --split_dataset 'validate' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/TRAININGS/delay400ms_retry/best_net.pth'  --specific_epoch 89 


#noRTE, 400ms
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'test_CNN3D_delay400ms_noRTE' --split_dataset 'test' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/TRAININGS/delay400ms_noRTE/best_net.pth'  --specific_epoch 89 
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'validate_CNN3D_delay400ms_noRTE' --split_dataset 'validate' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/TRAININGS/delay400ms_noRTE/best_net.pth'  --specific_epoch 89 


#past 1
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'test_CNN3D_delay400ms_past1' --split_dataset 'test' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/TRAININGS/delay400ms_len1/best_net.pth'  --specific_epoch 89 --len_past 1
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'validate_CNN3D_delay400ms_past1' --split_dataset 'validate' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/TRAININGS/delay400ms_len1/best_net.pth'  --specific_epoch 89  --len_past 1

#past 0
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'test_CNN3D_delay400ms_past0' --split_dataset 'test' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/TRAININGS/delay400ms_len0/best_net.pth'  --specific_epoch 89 
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'validate_CNN3D_delay400ms_past0' --split_dataset 'validate' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/TRAININGS/delay400ms_len0/best_net.pth'  --specific_epoch 89

######################  200ms  ######################################## ok
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'test_classic_delay200ms' --split_dataset 'test'
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'validate_classic_delay200ms' --split_dataset 'validate'

#with module delay
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'test_CNN3D_delay200ms' --split_dataset 'test' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/delay200ms/net_epoch72.pth'  --specific_epoch 72
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir MODEL_v2xset/v2x-vit --fusion_method 'intermediate' --name_output_result 'validate_CNN3D_delay200ms' --split_dataset 'validate' --module_delay  --specific_path 'MODEL_v2xset/v2x-vit/delay200ms/net_epoch72.pth'  --specific_epoch 72