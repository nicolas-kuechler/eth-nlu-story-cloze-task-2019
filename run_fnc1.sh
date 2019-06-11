#load python and cuda version
module load python_gpu/3.6.4 cuda/10.0.130 cudnn/7.5

#enter virtual env
source ./venv_tf2/bin/activate

python3 code/model/fnc-1/pred.py

#cleanup
deactivate

