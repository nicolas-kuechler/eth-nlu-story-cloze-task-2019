import pandas as pd
import subprocess, argparse
import tensorflow as tf
import os, time

import dataset, prediction

# TODO [nku] needs to be run

parser = argparse.ArgumentParser(description='')
parser.add_argument('--init_ckpt_dir', help='initial checkpoint directory')
parser.add_argument('--epochs', help='number of epochs', type=int, default=3)
parser.add_argument('--output_dir', help='output directory (suggested to use scratch dir on cluster)')

args = parser.parse_args()


RUN_ID = str(int(time.time())) # use current timestamp as runid
output_dir = f"{args.output_dir}/runs/train_on_valid_{RUN_ID}"
os.makedirs(output_dir)

n_epoch = args.epochs
init_ckpt_dir = args.init_ckpt_dir
checkpoint = tf.train.latest_checkpoint(checkpoint_dir=init_ckpt_dir)

print("Running BERT Training on Validation Set: ")
print(f"   for {n_epoch} epochs")
print(f"   using checkpoint: {checkpoint}")
print(f"   using output dir: {output_dir}")


# Create Training Set From Validation Set
ds_valid = dataset.get_valid_dataset(path_valid='./data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv', shuffle=True)
dataset.save_dataset_as_tsv(ds_valid, path=f"{output_dir}/ds_train.tsv")

# BERT params
BERT_BASE_DIR = "./data/uncased_L-12_H-768_A-12"
DATA_DIR = "./data"

MAX_SEQ_LENGTH = 128
TRAIN_BATCH_SIZE = 32
PREDICT_BATCH_SIZE = 32
LEARNING_RATE = "2e-5"

# Run Ablation Training + Prediction
subprocess.check_output(["python", "code/model/bert/run_classifier.py", "--task_name=SCT",
                                                        "--do_train=true",
                                                        "--do_eval=true",
                                                        "--do_predict_valid=true",
                                                        "--do_predict_test=true",
                                                        "--do_predict_eth_test=true",
                                                        f"--train_data_dir={output_dir}",
                                                        f"--data_dir={DATA_DIR}",
                                                        f"--vocab_file={BERT_BASE_DIR}/vocab.txt",
                                                        f"--bert_config_file={BERT_BASE_DIR}/bert_config.json",
                                                        f"--init_checkpoint={checkpoint}",
                                                        f"--max_seq_length={MAX_SEQ_LENGTH}",
                                                        f"--train_batch_size={TRAIN_BATCH_SIZE}",
                                                        f"--predict_batch_size={PREDICT_BATCH_SIZE}",
                                                        f"--learning_rate={LEARNING_RATE}",
                                                        f"--num_train_epochs={n_epoch}",
                                                        f"--output_dir={output_dir}"])

# Convert the BERT results
prediction.convert_bert_results(output_dir)