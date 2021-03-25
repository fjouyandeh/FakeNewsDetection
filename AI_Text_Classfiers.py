#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import confusion_matrix 


# In[42]:


np.random.seed(500)


# In[43]:


df = pd.read_csv("E:\MASTER'S\WINDSOR DOCUMENTS\WINTER 2021\COMP_8700_Intro_to_AI\Project\Projects\FakeNewsDetection-main\Dataset\TrainFinalDS_AdditionalFeatures.csv")


# In[44]:


count_vect = CountVectorizer()


# In[45]:


#X = df["tokenized_text"].values
X = df["token_texts"].values


# In[46]:


X_train_counts = count_vect.fit_transform(X)


# In[47]:


tfidf_transformer = TfidfTransformer()


# In[48]:


X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# In[49]:


target_col_array = df["Final Label"].values
#print(target_col_array)


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, target_col_array, test_size = 0.20, random_state = 0) 


# In[51]:


classifier = GaussianNB()


# In[52]:


classifier.fit(X_train.toarray(), y_train)


# In[53]:


y_pred = classifier.predict(X_test.toarray()) 
  
# making the confusion matrix 
cm = confusion_matrix(y_test, y_pred) 
cm 


# In[54]:


Naive = naive_bayes.MultinomialNB()
Naive.fit(X_train.toarray(), y_train)


# In[55]:


# predict the labels on validation dataset
predictions_NB = Naive.predict(X_test)


# In[36]:


# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, y_test)*100)


# In[37]:


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=2.0, kernel='linear', degree=4, gamma='auto')


# In[38]:


SVM.fit(X_train, y_train)


# In[39]:


# predict the labels on validation dataset
predictions_SVM = SVM.predict(X_test)


# In[40]:


# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)


# In[ ]:





# In[ ]:





# In[ ]:




