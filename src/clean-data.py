import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize, word_tokenize

pd.options.mode.chained_assignment = None 


def clean_tokens(str):
    return str.replace('fw-', '').replace('-tl', '').replace('-hl', '').replace('-nc', '')


if __name__ == '__main__':
    df = pd.read_csv('../data/brown.csv')
    documents = df['filename'].unique()
    trainDocs, testDocs = train_test_split(documents, test_size = 0.2)
    trainDf = df[list(filename in trainDocs for filename in df['filename'])]
    testDf = df[list(filename in testDocs for filename in df['filename'])]

    print(df.shape)
    print(trainDf.shape)
    print(testDf.shape)

    trainDf['tokenized_pos'] = trainDf['tokenized_pos'].apply(clean_tokens)
    
    distinctTags = set()
    for l in trainDf['tokenized_pos']:
        tags = set(word_tokenize(l))
        distinctTags |= tags
    print(sorted(distinctTags))


