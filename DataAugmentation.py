import nltk
import pandas as pd

path = "./Dataset/TrainDataSet.csv"
df1 = pd.read_csv(path, sep = ',', encoding = "ISO-8859-1")

#WordNet-based augmentation
df2 = pd.read_csv(path, sep = ',', encoding = "ISO-8859-1")
from textaugment import Wordnet
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
t = Wordnet()
df2['text'] = df1['text'].map(lambda x: t.augment(x))
df2['Unique ID'] = df1['Unique ID'].map(lambda x: x+250)
bigData = df1.append(df2, ignore_index= True)

#Synonym Replacement
df3 = pd.read_csv(path, sep = ',', encoding = "ISO-8859-1")
from textaugment import EDA
t = EDA()
df3['text'] = df3['text'].map(lambda x: t.synonym_replacement(x))
df3['Unique ID'] = df3['Unique ID'].map(lambda x: x+500)
bigData = bigData.append(df3, ignore_index= True)

#Random Insertion
df4 = pd.read_csv(path, sep = ',', encoding = "ISO-8859-1")
from textaugment import EDA
t = EDA()
df4['text'] = df4['text'].map(lambda x: t.random_insertion(x))
df4['Unique ID'] = df4['Unique ID'].map(lambda x: x+750)
bigData = bigData.append(df4, ignore_index= True)


bigData.to_csv('./Dataset/AugmentedTrainDataSet.csv')