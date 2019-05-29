# NLU Project 2

## Usage Instructions

Download, unzip and place data folder in root of the project: https://polybox.ethz.ch/index.php/s/vGgC6Dv3jofwR1x

### Running on Leonhard Cluster
`module load gcc/4.8.5 python_gpu/3.6.4 hdf5 eth_proxy`
`module load cudnn/7.2`

use a virtual environment with the requirements from the root directory:
`pip install -r path/to/requirements.txt`


#### Run Training (Fine-Tuning with BERT)
`bsub -n 8 -W 12:00 -R "rusage[mem=16000, ngpus_excl_p=1]" python code/training.py --epochs=4 --output_dir=/cluster/scratch/kunicola`

(from the root of the project)

#### Run Prediction
`bsub -n 2 -W 4:00 -R "rusage[mem=16000, ngpus_excl_p=1]" python code/prediction.py --epoch_dir=/cluster/scratch/kunicola/runs/1559038542/epoch3`

(from the root of the project)