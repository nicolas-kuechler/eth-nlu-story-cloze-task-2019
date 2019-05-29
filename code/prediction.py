import pandas as pd
import subprocess, argparse
import tensorflow as tf




def run_bert_pred(epoch_dir):
    BERT_BASE_DIR = "./data/uncased_L-12_H-768_A-12"
    DATA_DIR = "./data"
    MAX_SEQ_LENGTH = 128

    print("Running BERT Prediction: ")
    print(f"   using epoch dir: {epoch_dir}")

    checkpoint = tf.train.latest_checkpoint(checkpoint_dir=epoch_dir)
    print(f"   using checkpoint: {checkpoint}")

    subprocess.check_output(["python", "code/model/bert/run_classifier.py", "--task_name=SCT",
                                                        "--do_train=false",
                                                        "--do_eval=false",
                                                        "--do_predict_cross=false",
                                                        "--do_predict_valid=true",
                                                        "--do_predict_test=true",
                                                        f"--train_data_dir={epoch_dir}",
                                                        f"--data_dir={DATA_DIR}",
                                                        f"--vocab_file={BERT_BASE_DIR}/vocab.txt",
                                                        f"--bert_config_file={BERT_BASE_DIR}/bert_config.json",
                                                        f"--init_checkpoint={checkpoint}",
                                                        f"--max_seq_length={MAX_SEQ_LENGTH}",
                                                        f"--output_dir={epoch_dir}"])


def convert_bert_pred(ds_path, ds_path_flat, ds_path_flat_valid_results, ds_output_path):
    """
    ds_path: path of the original dataset (csv)
    ds_path_flat: path of the tsv dataset given to BERT for predictions (tsv)
    ds_path_flat_valid_results: path of the predictions produced by BERT (tsv)
    ds_output_path: where to write output
    """

    df = pd.read_csv(ds_path)
    df_flat = pd.read_csv(ds_path_flat, sep='\t')
    df_flat_preds = pd.read_csv(ds_path_flat_valid_results, sep='\t', names=['prob0', 'prob1'])

    assert(df_flat.shape[0]==df_flat_preds.shape[0])

    # join predictions to df_flat
    df_flat = pd.concat([df_flat, df_flat_preds], axis = 1)

    # remove _1 from story start to get original story start id
    df_flat["story_start_id"] = df_flat["story_start_id"].str.split("_", n = 1, expand = True)[0]


    def _convert(x):
        # find probability of both endings
        prob_ending1 = df_flat[(df_flat['story_start_id']==x['InputStoryid']) & (df_flat['story_end']==x['RandomFifthSentenceQuiz1'])]['prob1'].values
        prob_ending2 =  df_flat[(df_flat['story_start_id']==x['InputStoryid']) & (df_flat['story_end']==x['RandomFifthSentenceQuiz2'])]['prob1'].values
        assert(len(prob_ending1)==1)
        assert(len(prob_ending2)==1)
        x['ProbQuiz1'] = prob_ending1[0]
        x['ProbQuiz2'] = prob_ending2[0]
            
        # predict ending with higher probability
        x['PredRightEnding'] = 1 if x['ProbQuiz1'] > x['ProbQuiz2'] else 2
            
        # check if prediction is correct
        x['CorrectPred'] = x['PredRightEnding']==x['AnswerRightEnding']
        return x

    # add predictions of the two endings to the original dataset
    df = df.apply(_convert,axis=1)

    accuracy = df['CorrectPred'].sum()/df.shape[0]

    df.to_csv(ds_output_path, index=False)

    return accuracy


parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch_dir', help='running directory of epoch')
args = parser.parse_args()

epoch_dir = args.epoch_dir

#epoch_dir = "./runs/1559038542/epoch3"
print(f"Using Epoch Dir: {epoch_dir}")

run_bert_pred(epoch_dir=epoch_dir)

# convert the predictions for the validation set
print("Converting Validation Results...")
valid_accuracy = convert_bert_pred(ds_path="./data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv", 
                    ds_path_flat="./data/ds_valid.tsv", 
                    ds_path_flat_valid_results=f"{epoch_dir}/valid_results.tsv",
                    ds_output_path=f"{epoch_dir}/valid_results_converted.csv")

print("Converting Test Results...")
test_accuracy = convert_bert_pred(ds_path="./data/cloze_test_test__spring2016 - cloze_test_ALL_test.csv",
                    ds_path_flat="./data/ds_test.tsv",
                    ds_path_flat_valid_results=f"{epoch_dir}/test_results.tsv",
                    ds_output_path=f"{epoch_dir}/test_results_converted.csv")

with open(f"{epoch_dir}/prediction_results_converted.txt", "w") as text_file:
    text_file.write(f"Validation Accuracy: {valid_accuracy}")
    text_file.write(f"Test Accuracy: {test_accuracy}")



