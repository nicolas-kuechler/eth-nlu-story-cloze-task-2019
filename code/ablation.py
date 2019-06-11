import pandas as pd
import time, os, subprocess, argparse
import tensorflow as tf

import dataset, prediction


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epochs', help='number of epochs', type=int, default=4)
    parser.add_argument('--output_dir', help='output directory (suggested to use scratch dir on cluster)')
    parser.add_argument('--ablations', nargs='+', type=int, default=[1, 2, 3, 4], help='specify ablation studies to train (1-4)')

    args = parser.parse_args()

    N_EPOCH = args.epochs
    OUTPUT_DIR = args.output_dir
    ABLATIONS = args.ablations

    print(f"Run Ablation Training for {N_EPOCH} epochs")
    print(f"Output Directory: {OUTPUT_DIR}")

    RUN_ID = str(int(time.time())) # use current timestamp as runid
    if 1 in ABLATIONS:
        ablation_dir = f"{OUTPUT_DIR}/runs/{RUN_ID}/ablation1"
        os.makedirs(ablation_dir)

        print("Running Ablation 1 Training: With Fixed Dataset and Random Ending")
        ds_train_ablation1 = dataset.get_ablation1_dataset()
        dataset.save_dataset_as_tsv(ds_train_ablation1, path=f"{ablation_dir}/ds_train.tsv")
        run_ablation_training(ablation_dir=ablation_dir, n_epoch=N_EPOCH)
        prediction.convert_bert_results(ablation_dir)

    if 2 in ABLATIONS:
        ablation_dir = f"{OUTPUT_DIR}/runs/{RUN_ID}/ablation2"
        os.makedirs(ablation_dir)

        print("Running Ablation 2 Training: With Fixed Dataset, Random Ending and Name Replacement")
        ds_train_ablation2 = dataset.get_ablation2_dataset()
        dataset.save_dataset_as_tsv(ds_train_ablation2, path=f"{ablation_dir}/ds_train.tsv")
        run_ablation_training(ablation_dir=ablation_dir, n_epoch=N_EPOCH)
        prediction.convert_bert_results(ablation_dir)

    if 3 in ABLATIONS:
        ablation_dir = f"{OUTPUT_DIR}/runs/{RUN_ID}/ablation3"
        os.makedirs(ablation_dir)

        print("Running Ablation 3 Training: With Fixed Dataset, Top1 Title Similarity Ending and Name Replacement")
        ds_train_ablation3 = dataset.get_ablation3_dataset()
        dataset.save_dataset_as_tsv(ds_train_ablation3, path=f"{ablation_dir}/ds_train.tsv")
        run_ablation_training(ablation_dir=ablation_dir, n_epoch=N_EPOCH)
        prediction.convert_bert_results(ablation_dir)
    
    if 4 in ABLATIONS:
        ablation_dir = f"{OUTPUT_DIR}/runs/{RUN_ID}/ablation4"
        os.makedirs(ablation_dir)

        print("Running Ablation 4 Training: With Fixed Dataset, Top1 Stories Similarity Ending and Name Replacement")
        ds_train_ablation4 = dataset.get_ablation4_dataset()
        dataset.save_dataset_as_tsv(ds_train_ablation4, path=f"{ablation_dir}/ds_train.tsv")
        run_ablation_training(ablation_dir=ablation_dir, n_epoch=N_EPOCH)
        prediction.convert_bert_results(ablation_dir)

def run_ablation_training(ablation_dir, n_epoch):

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
                                                        "--do_predict_cross=false",
                                                        "--do_predict_valid=true",
                                                        "--do_predict_test=true",
                                                        "--do_predict_eth_test=true",
                                                        f"--train_data_dir={ablation_dir}",
                                                        f"--data_dir={DATA_DIR}",
                                                        f"--vocab_file={BERT_BASE_DIR}/vocab.txt",
                                                        f"--bert_config_file={BERT_BASE_DIR}/bert_config.json",
                                                        f"--init_checkpoint={DATA_DIR}/init/bert_model.ckpt",
                                                        f"--max_seq_length={MAX_SEQ_LENGTH}",
                                                        f"--train_batch_size={TRAIN_BATCH_SIZE}",
                                                        f"--predict_batch_size={PREDICT_BATCH_SIZE}",
                                                        f"--learning_rate={LEARNING_RATE}",
                                                        f"--num_train_epochs={n_epoch}",
                                                        f"--output_dir={ablation_dir}"])

if __name__ == '__main__':
    main()