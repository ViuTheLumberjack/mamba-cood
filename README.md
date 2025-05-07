# v2x_feature_forecasting

#modified repository of Opencood: https://github.com/DerrickXuNu/OpenCOOD
For the details, follow that repo!

modified files (that this repo is using):
opencood/models/point_pillar_transformer.py
opencood/models/sub_models/delay_*.py  (modules to forecast the features for delay solving)
opencood/models/sub_models/loss/point_pillar_loss.py
opencood/data_utils/datasets/data_utils/intermediate_fusion_dataset.py
opencood/data_utils/datasets/data_utils/basedataset.py

opencood/tools/inference_utils.py
opencood/tools/train_utils.py


0. Install the environment using the guide in installation.sh (work in )

1. Dataset Preparation
following the guide in https://github.com/DerrickXuNu/v2x-vit?tab=readme-ov-file, section Data
url to download data: https://ucla.app.box.com/v/UCLA-MobilityLab-V2XVIT

2. Extract the feature to avoid to compute each time during the experiment using run_feature_pipeline.py

2_bis. Get the feature from disk in ('/equilibrium/fmarchetti/v2x/OpenCOOD/FEATURE_SAVED/feature_saved_f2f_scenarioTS_agents_folders_2')

3. Set your w&b account to log the experiment

4. Train the model with delay module using run_train.sh (tha train script is opencood/tools/train_feature.py)
change "root_dir" and "validate_dir" in MODEL_v2xset/v2x-vit/config_training.yaml, specify where is your dataset

5. Get results of specific split (train/val/test) using run_test.py (the evaluation script is opencood/tools/inference.py)
change root_dir in MODEL_v2xset/v2x-vit/config_evaluation.yaml, specify where is your dataset



 - the pretrained model, config and results are in MODEL_v2xset/v2x-vit (those that i've used)
 - the training of the model are saved in MODEL_v2xset/v2x-vit/TRAININGS