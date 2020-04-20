import pandas as pd
import os
import numpy as np
import nltk
import tensorflow as tf
from gensim.models import Word2Vec 
from nltk.tokenize import word_tokenize

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, LSTM, Masking, TimeDistributed, Dense, Embedding
from tensorflow.keras.layers import Bidirectional


# From clean-data.py, we know there are 179 tags and 56057 words


# Architectures:
#   1: single-directional
#   2: bi-directional
architecture = 2


def numToVec(num, size=179):
    vec = np.zeros((size))
    vec[num] = 1
    return vec

def preprocess(line):
    defaults = [tf.constant([], dtype=tf.string)]*6
    pieces = tf.io.decode_csv(line, defaults)
    words = tf.strings.to_number(tf.strings.split(pieces[4]), out_type=tf.dtypes.int32)
    tags = tf.strings.to_number(tf.strings.split(pieces[5]), out_type=tf.dtypes.int32)
    y = tf.one_hot(tags, 179)   
    return words, y
    

def csv_reader_dataset(filepaths, repeat=1, n_readers=5, n_read_threads=None,
                       shuffle_buffer_size=10000, n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths)
    dataset = dataset.interleave(
                lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
                cycle_length=n_readers, num_parallel_calls=n_read_threads
            )
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.shuffle(shuffle_buffer_size).repeat(repeat)
    return dataset.batch(batch_size).prefetch(1)


if __name__ == '__main__':
    
#    train_filepaths = ['../data/processed/train/train-{}.csv'.format(i) for i in range(10)]
#    filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)
#    
#    n_readers = 5
#    dataset = filepath_dataset.interleave(
#            lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
#            cycle_length = n_readers
#        )
#    
#    for line in dataset.take(5):
#        #preprocess(line.numpy())
#        print(preprocess(line.numpy()))
    
    trainDir = '../data/processed/train'
    trainFiles = os.listdir(trainDir)
    trainPaths = []
    for file in trainFiles:
        trainPaths.append(os.path.join(trainDir, file))
    
    valDir = '../data/processed/validation'
    valFiles = os.listdir(valDir)
    valPaths = []
    for file in valFiles:
        valPaths.append(os.path.join(valDir, file))
        
    testDir = '../data/processed/test'
    testFiles = os.listdir(testDir)
    testPaths = []
    for file in testFiles:
        testPaths.append(os.path.join(testDir, file))
        
    trainSet = csv_reader_dataset(trainPaths)
    valSet = csv_reader_dataset(valPaths)
    testSet = csv_reader_dataset(testPaths)

    print('Training model...')
    if architecture == 1:
        model = Sequential()
        model.add(Embedding(56058, 50, input_length=180))
        model.add(LSTM(256, return_sequences=True))
        model.add(TimeDistributed(Dense(179)))
        model.add(Activation('softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', 
                      metrics=['categorical_accuracy'])
        model.fit(trainSet, epochs=7, validation_data=valSet, verbose=2)
        model.evaluate(testSet)
    elif architecture == 2:
        model = Sequential()
        model.add(Embedding(56058, 50, input_length=180))
        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(TimeDistributed(Dense(179)))
        model.add(Activation('softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', 
                      metrics=['categorical_accuracy'])
        model.fit(trainSet, epochs=10, validation_data=valSet, verbose=2)
        model.evaluate(testSet)