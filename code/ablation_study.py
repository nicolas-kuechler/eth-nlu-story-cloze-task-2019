import pandas as pd
import time, os, subprocess, argparse
import tensorflow as tf
import random

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

print(f"CREATING NEW TRAIN DATASET...")
ds_train_true = dataset.get_train_dataset_true(path_train=f"{DATA_DIR}/train_stories.csv", shuffle=False)
ds_train_false = dataset.get_fixed_ds(path_train_true=f'{DATA_DIR}/train_stories.csv', path_train_false=f'{DATA_DIR}/false_dataset.csv', name_replacement=False, path_names='data/first_names.csv',shuffle=False)
ds_train = ds_train_true + ds_train_false
print(f'length of ds_train: {len(ds_train)}')
print(f'sample: {ds_train[88159:88163]}')

# shuffle new training dataset
print(f"SHUFFLE NEW TRAIN DATASET...")
#ds_train = ds_train.sample(frac=1).reset_index(drop=True)
random.shuffle(ds_train)

#top1 ablation study 
ablation_1 = True
if ablation_1:
    ds_train = df.from_csv("./data/ds_train.tsv", sep='\t', index=False, columns=['story_start_id', 'story_end_id', 'story_start', 'story_end', 'label'])



checkpoint = f"{DATA_DIR}/init/bert_model.ckpt"

for epoch in range(N_EPOCH):

    print(f"RUNNING FINE-TUNING EPOCH: {epoch+1}...")

    # set running directory
    cur_epoch_dir = f"{OUTPUT_DIR}/runs/{RUN_ID}/epoch{epoch+1}"
    os.makedirs(cur_epoch_dir)
    print(f"CURRENT EPOCH DIRECTORY: {cur_epoch_dir}")

    # writing new training dataset
    print(f"WRITE NEW TRAIN DATASET AS TSV TO: {cur_epoch_dir}")

    if ablation_1:
        dataset.save_dataset_as_tsv(ds_train, path=f"{cur_epoch_dir}/ds_train.tsv")
    else:
        dataset.save_dataset_as_tsv_list(ds_train, path=f"{cur_epoch_dir}/ds_train.tsv")

    # finetune BERT classifier with current train dataset
    print(f"START FINE-TUNING OF BERT FOR ONE EPOCH WITH NEW DATASET...")
    start = time.time()

    bert_out = subprocess.check_output(["python", "code/model/bert/run_classifier.py", "--task_name=SCT",
                                                "--do_train=true",
                                                "--do_eval=true",
                                                "--do_predict_cross=false",
                                                "--do_predict_valid=true",
                                                "--do_predict_test=true",
                                                f"--train_data_dir={cur_epoch_dir}",
                                                f"--data_dir={DATA_DIR}",
                                                f"--vocab_file={BERT_BASE_DIR}/vocab.txt",
                                                f"--bert_config_file={BERT_BASE_DIR}/bert_config.json",
                                                f"--init_checkpoint={checkpoint}",
                                                f"--max_seq_length={MAX_SEQ_LENGTH}",
                                                f"--train_batch_size={TRAIN_BATCH_SIZE}",
                                                f"--predict_batch_size={PREDICT_BATCH_SIZE}",
                                                f"--learning_rate={LEARNING_RATE}",
                                                f"--num_train_epochs=1",
                                                f"--output_dir={cur_epoch_dir}"])



    print(f"FINISHED FINE-TUNING EPOCH OF BERT: elapsed time = {time.time()-start}")

    checkpoint = tf.train.latest_checkpoint(checkpoint_dir=cur_epoch_dir)

    # adjust prev epoch dir
    prev_epoch_dir = cur_epoch_dir