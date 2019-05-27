import pandas as pd
import time, os, subprocess

from dataset_generation import dataset


N_EPOCH = 5

OUTPUT_DIR = "/cluster/scratch/kunicola"
BERT_BASE_DIR = "./data/uncased_L-12_H-768_A-12"
DATA_DIR = "./data"

MAX_SEQ_LENGTH = 128
TRAIN_BATCH_SIZE = 32
PREDICT_BATCH_SIZE = 32
LEARNING_RATE = "2e-5"

RUN_ID = str(int(time.time())) # use current timestamp as runid
prev_epoch_dir = f"{DATA_DIR}/init"

ds_train_true = dataset.get_train_dataset_true(path_train=f"{DATA_DIR}/train_stories.csv", shuffle=False)
df_cross = pd.read_csv(f"{DATA_DIR}/ds_cross_product_false.tsv", sep='\t')


for epoch in range(N_EPOCH):
     print(f"RUNNING FINE-TUNING EPOCH: {epoch+1}...")

     # set running directory
     cur_epoch_dir = f"{OUTPUT_DIR}/runs/{RUN_ID}/epoch{epoch+1}"
     os.makedirs(cur_epoch_dir)
     print(f"CURRENT EPOCH DIRECTORY: {cur_epoch_dir}")

     # read BERT prediction results from prev epoch and combine with cross product dataset to generate new dataset for this epoch
     print(f"READING PREDICTIONS FROM: {prev_epoch_dir}")
     df_preds = pd.read_csv(f"{prev_epoch_dir}/cross_results.tsv", sep='\t', names=['prob0', 'prob1'])
     print(f"COMBINE PREDICTIONS WITH CROSS PROD DATASET...")
     df = pd.concat([df_cross, df_preds], axis = 1)

     # find most likely prediction per story start
     print(f"FIND MOST LIKELY WRONG PREDICTION PER STORY START...")
     start = time.time()
     ds_train_max_false = df.sort_values("prob1", ascending=False).drop_duplicates(["story_start_id"])
     print(f"finished: elapsed time = {time.time()-start}")

     # combine samples with highest prediction and false label with the true samples
     print(f"CREATING NEW TRAIN DATASET...")
     ds_train = ds_train_max_false.append(ds_train_true, ignore_index=True)

     # shuffle new training dataset
     print(f"SHUFFLE NEW TRAIN DATASET...")
     ds_train = ds_train.sample(frac=1).reset_index(drop=True)

     # writing new training dataset
     print(f"WRITE NEW TRAIN DATASET AS TSV TO: {cur_epoch_dir}")
     dataset.save_dataset_as_tsv(ds_train, path=f"{cur_epoch_dir}/ds_train.tsv")

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
                                                    f"--init_checkpoint={prev_epoch_dir}/bert_model.ckpt",
                                                    f"--max_seq_length={MAX_SEQ_LENGTH}",
                                                    f"--train_batch_size={TRAIN_BATCH_SIZE}",
                                                    f"--predict_batch_size={PREDICT_BATCH_SIZE}",
                                                    f"--learning_rate={LEARNING_RATE}",
                                                    f"--num_train_epochs=1",
                                                    f"--output_dir={cur_epoch_dir}"])



     print(f"FINISHED FINE-TUNING EPOCH OF BERT: elapsed time = {time.time()-start}")

     # adjust prev epoch dir
     prev_epoch_dir = cur_epoch_dir
