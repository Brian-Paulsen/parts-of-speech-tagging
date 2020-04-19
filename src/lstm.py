import pandas as pd
import numpy as np
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize 


if __name__ == '__main__':
    df = pd.read_csv('../data/brown.csv')
    print(df.head())
    
    df1 = pd.DataFrame({
        'a' : ['as er ad', 'red dr as', 're dr efe'] 
    })
    checks = set(['er', 're'])
    print(['as', 'er', 'ad'] in ['er', 're'])
    print(list(bool(set(d.split()) & checks) for d in df1['a']))