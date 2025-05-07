

#conda env create -f environment_38.yml
conda env create -f environment.yml
source activate opencood

python setup.py develop "(give the error error: Couldnt find a setup script in /tmp/easy_install-sxzz3370/scikit_image-0.25.2.tar.gz but its ok)"


conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install spconv-cu113
python opencood/utils/setup.py build_ext --inplace

#it give error, but it is ok, don't run it
#python opencood/pcdet_utils/setup.py build_ext --inplace

pip install h5py
conda install mkl==2024.0
pip install wandb

