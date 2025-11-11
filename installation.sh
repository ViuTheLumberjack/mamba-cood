

#conda env create -f environment_38.yml
conda env create -f environment.yml
source activate opencood_test

python setup.py develop # "(give the error error: Couldnt find a setup script in /tmp/easy_install-sxzz3370/scikit_image-0.25.2.tar.gz but its ok)"

# conda install pytorch==2.8.0 torchvision torchaudio cuda-toolkit=12.8 -c pytorch
#pip install spconv-cu113
python opencood/utils/setup.py build_ext --inplace

#it give error, but it is ok, don't run it
#python opencood/pcdet_utils/setup.py build_ext --inplace

#pip install h5py
# conda install mkl==2024.0
#pip install wandb

