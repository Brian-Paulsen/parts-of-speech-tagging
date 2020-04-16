import pandas as pd
import numpy as np
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize 


if __name__ == '__main__':
    df = pd.read_csv('../data/brown.csv')
    print(df.head())
    tokenizedText = df.loc[0, 'tokenized_text']
    print(word_tokenize(tokenizedText))