import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize, word_tokenize

pd.options.mode.chained_assignment = None 


def clean_tokens(str):
    return str.replace('fw-', '').replace('-tl', '').replace('-hl', '').replace('-nc', '')

def clean_sentence(sentence: str):
    parts = sentence.split(' ')
    words = []
    tokens = []
    for part in parts:
        pieces = part.split('/')
        wordPiece = pieces[:-1]
        word = '/'.join(wordPiece)
        token = pieces[-1]
        words.append(word)
        tokens.append(token)
    return ' '.join(words), ' '.join(tokens)
        

if __name__ == '__main__':
    df = pd.read_csv('../data/brown.csv')
    wordStrs = []
    tokenStrs = []
    for sentence in df['raw_text']:
        wordStr, tokenStr = clean_sentence(sentence)
        wordStrs.append(wordStr)
        tokenStrs.append(tokenStr)
        
    cleanDf = pd.DataFrame({
        'filename' : df['filename'],
        'para_id' : df['para_id'],
        'sent_id' : df['sent_id'],
        'raw_text' : df['raw_text'],
        'tokenized_text' : wordStrs,
        'tokenized_pos' : tokenStrs,
        'label' : df['label']
    })

    cleanDf['tokenized_pos'] = cleanDf['tokenized_pos'].apply(clean_tokens)

    distinctTags = set()
    for l in cleanDf['tokenized_pos']:
        tags = set(word_tokenize(l))
        distinctTags |= tags
    print(sorted(distinctTags))
    
    cleanDf.to_csv('../data/brown-cleaned.csv')


