import pandas as pd
import numpy as np
import nltk
from gensim.models import Word2Vec 
from nltk.tokenize import word_tokenize


if __name__ == '__main__':
    df = pd.read_csv('../data/brown-cleaned.csv')
    print(df.head())
        
    inputLists = []
    for sentence in df['tokenized_text']:
        words = word_tokenize(sentence)
        wordCount = len(words)
        newList = np.zeros((50, wordCount))
        for word in words:
            newList[,i] = 
        inputLists.append(newList)