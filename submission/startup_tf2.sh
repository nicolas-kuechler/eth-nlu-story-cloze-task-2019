#only when running for the first time

module load python_gpu/3.6.4 cuda/9.0.176 cudnn/7.5

pip install --user pipenv
virtualenv --system-site-packages -p python3 ./venv_tf2
source ./venv_tf2/bin/activate
pip3 install sklearn
pip3 install tf-nightly-gpu-2.0-preview
pip3 install gensim
deactivate
