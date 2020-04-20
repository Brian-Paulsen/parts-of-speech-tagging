import pandas as pd
import numpy as np
from math import ceil

pd.options.mode.chained_assignment = None 


def clean_tokens(str):
    return str.replace('fw-', '').replace('-tl', '').replace('-hl', '').replace('-nc', '').replace('$', '')

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
        
def arrayToString(array):
    string = ''
    for element in array:
        string += str(int(element)) + ' '
    return string.strip()


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
        tags = set(l.split())
        distinctTags |= tags
    print(sorted(distinctTags))
    
    cleanDf.to_csv('../data/brown-cleaned.csv')


    uniqueWords = set()
    uniqueTags = set()
    for i in range(cleanDf.shape[0]):
        words = set(cleanDf.loc[i, 'tokenized_text'].split(' '))
        tags = set(cleanDf.loc[i, 'tokenized_pos'].split(' '))
        uniqueWords |= words
        uniqueTags |= tags
        
    wordDict = {}
    uniqueWordsList = list(uniqueWords)
    for i in range(len(uniqueWordsList)):
        wordDict[uniqueWordsList[i]] = i + 1
    
    tagDict = {}
    uniqueTagList = list(uniqueTags)
    for i in range(len(uniqueTags)):
        tagDict[uniqueTagList[i]] = i + 1
        
    maxLength = 0
    for sentence in cleanDf['tokenized_text']:
        wordCount = len(sentence.split(' '))
        maxLength = max(maxLength, wordCount)
        

    inputStrs = []
    outputStrs = []
    for i in range(cleanDf.shape[0]):
        wordList = np.zeros((maxLength))
        tagList = np.zeros((maxLength))
        words = cleanDf.loc[i, 'tokenized_text'].split(' ')
        tags = cleanDf.loc[i, 'tokenized_pos'].split(' ')
        for j in range(len(words)):
            wordList[j] = wordDict[words[j]]
            tagList[j] = tagDict[tags[j]]
        inputStrs.append(arrayToString(wordList))
        outputStrs.append(arrayToString(tagList))
            
    numberDf = pd.DataFrame({
        'filename' : cleanDf['filename'],
        'para_id' : cleanDf['para_id'],
        'sent_id' : cleanDf['sent_id'],
        'input' : inputStrs,
        'output' : outputStrs
    })
            
    numberDf.to_csv('../data/number-data.csv')
    
    shuffledNumberDf = numberDf.sample(frac=1)
    
    trainCount = int(shuffledNumberDf.shape[0] * 0.6)
    valCount = int(shuffledNumberDf.shape[0] * 0.2)
    testCount = shuffledNumberDf.shape[0] - trainCount - valCount
    
    trainDf = shuffledNumberDf.iloc[:trainCount]
    valDf = shuffledNumberDf.iloc[(trainCount):(trainCount+valCount)]
    testDf = shuffledNumberDf.iloc[(trainCount + valCount):]
    
    trainFileCount = ceil(trainCount // 64)
    valFileCount = ceil(valCount // 64)
    testFileCount = ceil(testCount // 64)
    
    for i in range(trainFileCount):
        subDf = trainDf.iloc[(64*i):(64*i + 64)]
        subDf.to_csv('../data/processed/train/train-{}.csv'.format(i))
    
    for i in range(valFileCount):
        subDf = valDf.iloc[(64*i):(64*i + 64)]
        subDf.to_csv('../data/processed/validation/val-{}.csv'.format(i))
        
    for i in range(testFileCount):
        subDf = testDf.iloc[(64*i):(64*i+64)]
        subDf.to_csv('../data/processed/test/test-{}.csv'.format(i))
    