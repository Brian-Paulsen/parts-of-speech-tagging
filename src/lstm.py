import pandas as pd
import numpy as np
import nltk
from gensim.models import Word2Vec 
from nltk.tokenize import word_tokenize

from keras.models import Sequential
from keras.layers import Activation, LSTM


if __name__ == '__main__':
    df = pd.read_csv('../data/brown-cleaned.csv')
    print(df.head())
        
    word2VecModel = Word2Vec.load('../models/word2Vec.model')
    
    uniqueTagsSet = set()
    for sentence in df['tokenized_pos']:
        uniqueTagsSet |= set(sentence.split(' '))
    uniqueTags = list(uniqueTagsSet)
    uniqueTagCount = len(uniqueTags)
    
    tagDict = {}
    for i in range(uniqueTagCount):
        tagDict[uniqueTags[i]] = i
    
    inputLists = []
    outputLists = []
    for i in range(df.shape[0]):
        sentence = df.loc[i, 'tokenized_text']
        tagsStr = df.loc[i, 'tokenized_pos']
        words = sentence.split(' ')
        tags = tagsStr.split(' ')
        wordCount = len(words)
        inputList = np.zeros((50, wordCount))
        outputList = np.zeros((uniqueTagCount, wordCount))
        for i in range(wordCount):
            inputList[:,i] = word2VecModel.wv[words[i]]
            outputList[tagDict[tags[i]], i] = 1
        inputLists.append(inputList)
        outputLists.append(outputList)
        
model = Sequential()
model.add(LSTM(30, return_sequences=True))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['categorical_accuracy'])