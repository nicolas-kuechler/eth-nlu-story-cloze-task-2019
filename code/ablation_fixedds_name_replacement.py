import pandas as pd
import time, os, subprocess, argparse
import tensorflow as tf

import dataset


parser = argparse.ArgumentParser(description='')
parser.add_argument('--epochs', help='number of epochs', type=int, default=4)
parser.add_argument('--output_dir', help='output directory (suggested to use scratch dir on cluster)')
args = parser.parse_args()

N_EPOCH = args.epochs

OUTPUT_DIR = args.output_dir

print(f"Run Training for {N_EPOCH} epochs")
print(f"Output Directory: {OUTPUT_DIR}")

BERT_BASE_DIR = "./data/uncased_L-12_H-768_A-12"
DATA_DIR = "./data"

MAX_SEQ_LENGTH = 128
TRAIN_BATCH_SIZE = 32
PREDICT_BATCH_SIZE = 32
LEARNING_RATE = "2e-5"

RUN_ID = str(int(time.time())) # use current timestamp as runid
prev_epoch_dir = f"{DATA_DIR}/init"

# set running directory
cur_epoch_dir = f"{OUTPUT_DIR}/runs/{RUN_ID}"
os.makedirs(cur_epoch_dir)
print(f"CURRENT EPOCH DIRECTORY: {cur_epoch_dir}")

print(f"CREATING NEW TRAIN DATASET...")
ds_train_true = dataset.get_train_dataset_true(path_train=f"{DATA_DIR}/train_stories.csv", shuffle=False)
ds_train_false = dataset.get_fixed_dataset(path_train=f'{DATA_DIR}/train_stories.csv', name_replacement=True, shuffle=False, path_names='./data/first_names.csv')
# combine samples with highest prediction and false label with the true samples
ds_train = ds_train_true.append(ds_train_false, ignore_index=True)

# shuffle new training dataset
print(f"SHUFFLE NEW TRAIN DATASET...")
ds_train = ds_train.sample(frac=1).reset_index(drop=True)

# writing new training dataset
print(f"WRITE NEW TRAIN DATASET AS TSV TO: {cur_epoch_dir}")
dataset.save_dataset_as_tsv(ds_train, path=f"{cur_epoch_dir}/ds_train.tsv")

checkpoint = f"{DATA_DIR}/init/bert_model.ckpt"

# finetune BERT classifier with current train dataset
print(f"START FINE-TUNING OF BERT FOR ONE EPOCH WITH NEW DATASET...")
start = time.time()

bert_out = subprocess.check_output(["python", "code/model/bert/run_classifier.py", "--task_name=SCT",
                                            "--do_train=true",
                                            "--do_eval=true",
                                            "--do_predict_cross=true",
                                            "--do_predict_valid=false",
                                            "--do_predict_test=false",
                                            f"--train_data_dir={cur_epoch_dir}",
                                            f"--data_dir={DATA_DIR}",
                                            f"--vocab_file={BERT_BASE_DIR}/vocab.txt",
                                            f"--bert_config_file={BERT_BASE_DIR}/bert_config.json",
                                            f"--init_checkpoint={checkpoint}",
                                            f"--max_seq_length={MAX_SEQ_LENGTH}",
                                            f"--train_batch_size={TRAIN_BATCH_SIZE}",
                                            f"--predict_batch_size={PREDICT_BATCH_SIZE}",
                                            f"--learning_rate={LEARNING_RATE}",
                                            f"--num_train_epochs={N_EPOCH}",
                                            f"--output_dir={cur_epoch_dir}"])



print(f"FINISHED FINE-TUNING EPOCH OF BERT: elapsed time = {time.time()-start}")

