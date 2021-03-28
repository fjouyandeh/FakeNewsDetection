import matplotlib
import pandas as pd
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("..\Dataset\TrainFinalDS_AdditionalFeatures.csv")

sns.countplot(df['Final Label'])
plt.xlabel('Label')
plt.title('Number of REAL and FAKE Tweets')
# plt.show()

from sklearn.preprocessing import LabelEncoder
#initializing an object of class LabelEncoder
labelencoder= LabelEncoder()
#fitting and transforming the desired categorical column
df['Final Label'] = labelencoder.fit_transform(df['Final Label'])
# print(df['Final Label'])

selectedColumns = [
    'retweets_count', 'likes_count', 'no_of_words', 'no_of_question_marks', 'no_of_exclamation_marks',
    'no_of_hashtags', 'no_of_urls', 'SentimentScore', 'Readability'
]

targetColumn = ['Final Label']

X = pd.DataFrame(df[list(selectedColumns)].values, columns = selectedColumns, index = df.index)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['token_texts']).toarray()
X_all_data_tfidf = np.hstack((X[selectedColumns].values, X_tfidf))

# print(X_all_data_tfidf.shape)
y = df[targetColumn]

#===========================================================
# Plot heat map

AllCols =  selectedColumns + targetColumn
PlotData = pd.DataFrame(df[list(AllCols)].values, columns = AllCols, index = df.index)

corrmat = PlotData.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (20,20))

g = sns.heatmap(PlotData[top_corr_features].corr(),annot = True, cmap = "RdYlGn")
# plt.show()

#===========================================================
# Split data
X_train, X_test, y_train, y_test = train_test_split(X_all_data_tfidf, y, test_size = 0.20, random_state = 0)

#===========================================================
# KNN Classifier
print('\nKNN Classifier:\n')
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test, y_pred))
print('\nKNN Classifier:\n')
print(classification_report(y_test, y_pred))

print("KNN Accuracy Rate -> ",accuracy_score(y_pred, y_test) * 100)
#===========================================================
# SVM Classifier
SVM = svm.SVC(C = 2.0, kernel = 'linear', degree = 4, gamma = 'auto')

SVM.fit(X_train, y_train.values.ravel())

# predict the labels on validation dataset
predictions_SVM = SVM.predict(X_test)
print('\nSVM Classifier:\n')
print(classification_report(y_test, predictions_SVM))

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Rate -> ",accuracy_score(predictions_SVM, y_test) * 100)

#===========================================================
# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train.values.ravel())
y_pred = lr.predict(X_test)
print('\nLogistic Regression Classifier:\n')
print(classification_report(y_test, y_pred))
print("Logistic Regression Accuracy Rate -> ", lr.score(X_test, y_test)  * 100)

#===========================================================
# Random Forest Classifier
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
print('\nRandom Forest Classifier:\n')
print(classification_report(y_test, y_pred))

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Random Forest Accuracy ->", metrics.accuracy_score(y_test, y_pred)  * 100)

#============================================================
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print('\nDecision Tree Classifier:\n')
print(classification_report(y_test, y_pred))
# Model Accuracy, how often is the classifier correct?
print("Decision Tree Accuracy ->", metrics.accuracy_score(y_test, y_pred ) * 100)
#============================================================
# Getting the final result
targetDf = pd.read_csv("..\Dataset\TestFinalDS_AdditionalFeatures.csv")
X_target_raw = pd.DataFrame(df[list(selectedColumns)].values, columns = selectedColumns, index = df.index)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['token_texts']).toarray()
X_target = np.hstack((X[selectedColumns].values, X_tfidf))
finalPredict = SVM.predict((X_target))
print(finalPredict.shape)

labelColumn = []
labelColumn[:] = ['REAL' if x == 1 else 'FAKE' for x in finalPredict]

import pandas
df = pandas.DataFrame(data={"Final Labels": labelColumn})
df.to_csv("..\Result\FinalLabel.csv", sep=',',index=False)


