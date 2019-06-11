# NLU Project 2 - Group 4 - ETHZ 2019

- Brunner Lucas	    brunnelu@student.ethz.ch
- Blatter Philippe	pblatter@student.ethz.ch
- Küchler Nicolas	kunicola@student.ethz.ch

## Usage Instructions
Download, unzip and place data folder in root of the project: https://polybox.ethz.ch/index.php/s/VsYxJUwCMMxpeoQ
(e.g. `curl -o data.zip https://polybox.ethz.ch/index.php/s/VsYxJUwCMMxpeoQ/download` and `unzip data.zip -d data`)


### Running on Leonhard Cluster
```
module load gcc/4.8.5 python_gpu/3.6.4 hdf5 eth_proxy
module load cudnn/7.2
```

use a virtual environment with the requirements from the root directory:
```
pip install -r path/to/requirements.txt
```

#### Reproduce Results Quick Version
1. Download and unzip data folder from polybox (see above)
2. Download epoch directory of fine-tuned BERT checkpoints (TODO: download link)
3. Run Predictions with adjusted EPOCH_DIR param: `bsub -n 2 -W 4:00 -R "rusage[mem=16000, ngpus_excl_p=1]" python code/prediction.py --epoch_dir $SCRATCH/runs/{EPOCH_DIR}`

#### Run Training on Training Set (Fine-Tuning with BERT)
```
bsub -n 8 -W 12:00 -R "rusage[mem=16000, ngpus_excl_p=1]" python code/training.py --epochs 3 --output_dir $SCRATCH
```
(from the root of the project)

This runs the fine-tuning of BERT with the training set. Requires that either the complete data folder was downloaded or the required files are generated (see dataset section below).

#### Run Prediction
```
bsub -n 2 -W 4:00 -R "rusage[mem=16000, ngpus_excl_p=1]" python code/prediction.py --epoch_dir $SCRATCH/runs/1559038542/epoch3
```
(from the root of the project)

After fine-tuning BERT on the training set find the epoch directory and calculate the predictions with this command.

#### Run Training on Validation Set (Fine-Tuning with BERT)

```
bsub -n 8 -W 4:00 -R "rusage[mem=16000, ngpus_excl_p=1]" python code/training_valid.py --epochs 3 --init_ckpt_dir $SCRATCH/runs/1559038542/epoch3 --output_dir $SCRATCH
```
(from the root of the project)

This takes the BERT model fine-tuned on the training set and then fine-tunes it further on the validation set.
To run the command find the epoch directory from the training on the training set (e.g. $SCRATCH/runs/1559038542/epoch3) and use this as --init_ckpt_dir in the following command:


#### Run Ablation Study
```
bsub -n 8 -W 12:00 -R "rusage[mem=16000, ngpus_excl_p=1]" python code/ablation.py --epochs 3 --output_dir $SCRATCH --ablations 1 2 3 4
```
(from the root of the project)

This runs an ablation study highlighting the impact of different parts of the training set generation.
For every selected ablation study, the dataset is generated and then the fine-tuning of BERT is run on the dataset.
At the end the predictions are run to calculate validation and test accuracy. 

#### Run FNC-1 Model
make sure you you get all datasets ready. If you don't have them follow the instructions in the section 'Create Dataset from ROC Corpus'
once run `bash startup_tf2.sh` in the root folder and to run model run `bsub -R "rusage[mem=16000, ngpus_excl_p=1]" <run_fnc1.sh`

#### Dataset

Instead of newly creating the dataset, one can also download the complete data folder in the proper format from the polybox link above.
**When working with the complete data folder from polybox, running the instructions below is not necessary**.


##### Create Dataset from ROC Corpus

The provided data files must be placed in the data folder:
```
data/train_stories.csv
data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv
data/test_for_report-stories_labels.csv
data/test-stories.csv
```

The index file created by training embeddings on story titles and selecting the 20 most similars per sample (see train embeddings on story titles)
```
data/train_stories_top_20_most_similar_titles.csv
```

The pre-trained checkpoints for the uncased BERT-Base from: https://github.com/google-research/bert
```
data/init/bert_model.ckpt.data-00000-of-00001
data/init/bert_model.ckpt.index
data/init/bert_model.ckpt.meta
```

With all these files in place the dataset can be created with:
```
python code/dataset.py
```

##### Train Embeddings on Story Titles

```
bsub -n 2 -W 4:00 -R "rusage[mem=16000, ngpus_excl_p=1]" python code/embedding/title_similarities.py
bsub -n 2 -W 4:00 -R "rusage[mem=16000, ngpus_excl_p=1]" python code/embedding/title_top20.py
```
(from the root of the project)

The first command trains embeddings based on the similarity of story titles. 
The second command generates an index table with the 20 most similar samples per sample.

##### Train Embeddings on Entire Stories
```
bsub -n 2 -W 4:00 -R "rusage[mem=16000, ngpus_excl_p=1]" python code/embedding/story_similarities.py
bsub -n 2 -W 4:00 -R "rusage[mem=16000, ngpus_excl_p=1]" python code/embedding/stories_top1.py
```
(from the root of the project)

The first command trains embeddings based on the similarity of stories. 
The second command generates an index table with the most similar sample.
