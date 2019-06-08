# NLU Project 2

## Usage Instructions

Download, unzip and place data folder in root of the project: https://polybox.ethz.ch/index.php/s/VsYxJUwCMMxpeoQ
e.g `curl -o data.zip https://polybox.ethz.ch/index.php/s/VsYxJUwCMMxpeoQ/download` and `unzip data.zip`

### Running on Leonhard Cluster
`module load gcc/4.8.5 python_gpu/3.6.4 hdf5 eth_proxy`
`module load cudnn/7.2`

use a virtual environment with the requirements from the root directory:
`pip install -r path/to/requirements.txt`


#### Run Training on Training Set (Fine-Tuning with BERT)
`bsub -n 8 -W 12:00 -R "rusage[mem=16000, ngpus_excl_p=1]" python code/training.py --epochs 3 --output_dir $SCRATCH`

(from the root of the project)

#### Run Training on Validation Set (Fine-Tuning with BERT)
`bsub -n 8 -W 4:00 -R "rusage[mem=16000, ngpus_excl_p=1]" python code/training_valid.py --epochs 3 --init_ckpt_dir $SCRATCH/runs/1559038542/epoch3 --output_dir $SCRATCH`

(from the root of the project)

#### Run Prediction
`bsub -n 2 -W 4:00 -R "rusage[mem=16000, ngpus_excl_p=1]" python code/prediction.py --epoch_dir $SCRATCH/runs/1559038542/epoch3`

(from the root of the project)

#### Run Ablation Study
`bsub -n 8 -W 12:00 -R "rusage[mem=16000, ngpus_excl_p=1]" python code/ablation.py --epochs 3 --output_dir $SCRATCH --ablations 1 2 3 4`

(from the root of the project)


#### Run FNC-1 Model
once run `bash startup_tf2.sh` in the root folder and to run model run `bsub -R "rusage[mem=16000, ngpus_excl_p=1]" <run_fnc1.sh`
