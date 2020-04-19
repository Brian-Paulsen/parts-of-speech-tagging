import time
import gensim
import pickle
from math import floor
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize


df = pd.read_csv('../data/brown-cleaned.csv')
print(df.head())

data = []


for sentence in df['tokenized_text']:
    data.append(word_tokenize(sentence))
    
print('Training Word2Vec...')
start = time.time()
word2VecModel = Word2Vec(data, min_count=1, size=50, workers=3, window=2, sg=1)
finish = time.time()
interval = floor(finish - start)
print('Trained in {} seconds'.format(interval))

print('Dumping...')
with open('word2vec.pickle', 'wb') as f:
    pickle.dump(word2VecModel, f)