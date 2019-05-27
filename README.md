NLU Project 2

download pretrained model and unzip (base model):
cd data
curl -o bert.zip https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip bert.zip
rm bert.zip

for large model:
curl -o bert.zip https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip

to run:
#bert NLI
bsub -R "rusage[mem=16000,ngpus_excl_p=1]" < run_bert_sct.sh
Bert SCT
bsub -R "rusage[mem=16000,ngpus_excl_p=1]" < run.sh