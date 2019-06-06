# Copyright 2017 Benjamin Riedel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##Atttention For generating the data a huge amount of memory is neccesary

#if no scratch set one by export SCRATCH=$PWD 

# Import relevant packages and modules
from util_data import * # pylint: disable=unused-wildcard-import
import random
import numpy as np
import tensorflow as tf
import os
import sys
from metric import f1_score


# Prompt for mode
mode ='train' #input('mode (load / train)? ')

# Set file names
file_train = "./data/ds_train.tsv"
file_val = './data/ds_valid.tsv'
#TODO remove for actual training
#file_train = file_val
file_predictions = os.environ['SCRATCH']+'/data/ucl/pred.tsv'

dir_save_np = os.environ['SCRATCH']+'/data/ucl/data/'
checkpoint_directory = os.environ['SCRATCH']+'/data/ucl/'


# Initialise hyperparameters
r = random.Random()
lim_unigram = 1000
target_size = 2
hidden_size = 100
train_keep_prob = 0.6
l2_alpha = 0.00001
learn_rate = 0.01
clip_ratio = 5
batch_size_train = 500
epochs = 20

load_processed_data=False
save_model_when_training=True

# In comand line run and visit localhost:6006
# tensorboard --logdir=ucl_tf
callback = tf.keras.callbacks.TensorBoard(log_dir=checkpoint_directory)


#generate Data
if load_processed_data==False:
    save_new_body_body(file_train, file_val,dir_save_np, lim_unigram = lim_unigram)

train_set, train_stances,test_set, n_train, feature_size, val_stance = load_body_body(dir_save_np)


##Model description##
input_shape = (train_set.shape[1],)
print(f"train_set[0].shape= {input_shape}")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=hidden_size,input_shape=input_shape, \
        kernel_regularizer=tf.keras.regularizers.l2(l=l2_alpha)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(rate=1 - (train_keep_prob)),
    tf.keras.layers.Dense(units=target_size, \
        kernel_regularizer=tf.keras.regularizers.l2(l=l2_alpha)),
    tf.keras.layers.Dropout(rate=1 - (train_keep_prob)),
    #tf.keras.layers.Reshape([batch_size_train, target_size]), # was 1 instead of batch_size
    tf.keras.layers.Softmax(),
])

optimizer=tf.keras.optimizers.Adam(lr=learn_rate)


print("model compile")
model.compile(optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])#,f1_score()

#print model
model.summary()
print()

print('create checkpoint for saving and restoring')
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

if mode == 'train':

    Y= tf.keras.utils.to_categorical(train_stances,num_classes=2)
    #print(f"train_stances shape: {train_stances.shape} and cothegorical shape: {Y.shape}")

    print("model fit")
    '''
    model.fit(x=train_set, y= train_stances,\
        verbose=0, \
        epochs=epochs, \
        batch_size= batch_size_train, \
        validation_data = (test_set,val_stance), \
        callbacks=[callback])
    '''
    ds_train = tf.data.Dataset.from_generator(lambda: batch_generator(train_set,train_stances,batch_size_train),(tf.float32,tf.int32)).prefetch(16)

    ds_val = tf.data.Dataset.from_generator(lambda: batch_generator(test_set,val_stance,batch_size_train),(tf.float32,tf.int32)).prefetch(16)
    

    model.fit_generator(generator = ds_train,\
        verbose=2, \
        epochs=epochs, \
        validation_data = ds_val, \
        callbacks=[callback])
    #predict train and test set for scoring
    y_train_pred = tf.argmax(model.predict(train_set),axis=1).numpy()
    y_val_pred = tf.argmax(model.predict(test_set),axis=1).numpy()

    calculate_and_print_scores(train_stances,y_train_pred,val_stance,y_val_pred)

    if save_model_when_training:
        print("save model")
        print(os.path.join(checkpoint_directory, "ckpt"))
        checkpoint.save(os.path.join(checkpoint_directory, "ckpt"))
 
elif mode=='load':
    print("load model")
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
else:
    print('specify if load or train!')
    exit(0)

predict = model.predict(test_set)
print(f"predicted: {predict}")
test_pred =tf.argmax(predict,axis=1).numpy()
print(f"predicted after argmax: {test_pred}")

# Save predictions
print("Save predictions")
save_predictions(predict, file_predictions)

