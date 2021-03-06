#make necessary imports
import os
import random
import glob
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import string
import textstat as ts
import json
import ast

#utility function to save the file in csv
def savetocsv(df,path):
    df.to_csv(path)

# Check if string contains non-English characters
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

#utility function to get the file in form of a pandas dataframe
def getframe(path):
    df=pd.read_csv(path, sep = ',', nrows = 20000, encoding = "ISO-8859-1")
    # Remove rows with non-English text
    df = df[df['text'].map(isEnglish) == True]
    # Drop duplicate rows
    df=df.drop_duplicates()
    # Drop the rows even with single NaN or single missing values
    df.dropna()
    return df

#features - countword
def CountWord(df):
    df['no_of_words'] = df['text'].map(lambda x:len(x.split()))
    return df

#features - CountQuestionMarks
def CountQuestionMarks(df):
    df['no_of_question_marks'] = df['text'].map(lambda x:x.count('?'))
    return df

#features - CountExclamationMarks
def CountExclamationMarks(df):
    df['no_of_exclamation_marks'] = df['text'].map(lambda x:x.count('!'))
    return df

#features - CountMentions
def CountMentions(df):
    df['no_of_mentions'] = df['mentions'].map(lambda x: len(json.loads(x)))
    return df

#features - CountHashtags
def CountHashtags(df):
    df['no_of_hashtags'] = df['hashtags'].map(lambda x: len(ast.literal_eval(x)))
    return df

#features - CountUrls
def CountUrls(df):
    df['no_of_urls'] = df['urls'].map(lambda x: len(ast.literal_eval(x)))
    return df

#features - SentimentScore
def SentimentScore(df):
    df['SentimentScore'] = df['text'].map(lambda x:SentimentIntensityAnalyzer().polarity_scores(x)['compound'])
    return df

#features - readability
def Readability(df):
    df['Readability'] = df['text'].map(lambda x:ts.automated_readability_index(x))
    return df

#features - count of punctuations
def CountPunc(df):
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    df['CountPunc'] = df.text.apply(lambda s: count(s, string.punctuation))
    return df

#utility function to add the linguistic features
def addfeatures(df):
    df = SentimentScore(df)
    print('SentimentScore finished')
    df = CountPunc(df)
    print('CountPunc finished')
    df = Readability(df)
    print('Readability finished')
    df = CountWord(df)
    print('CountWord finished')
    df = CountQuestionMarks(df)
    print('CountQuestionMarks finished')
    df = CountExclamationMarks(df)
    print('CountExclamationMarks finished')
    df = CountHashtags(df)
    print('CountHashtags finished')
    df = CountUrls(df)
    print('CountUrls finished')
    return df

#utility function to create file containing linguistic features
def createfile(path):
    df = getframe(path)
    df = addfeatures(df)
    x = path.find('.')
    df.to_csv('../Dataset/TrainFinalDS_AdditionalFeatures1.csv')
    return df


createfile("../Dataset/TrainFinalDS_AdditionalFeatures.csv")