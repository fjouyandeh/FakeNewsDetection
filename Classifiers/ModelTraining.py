import matplotlib
import pandas as pd
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("..\Dataset\AugmentedTrainDataSet.csv")

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
X_train, X_test, y_train, y_test = train_test_split(X_all_data_tfidf, y, test_size = 0.30, random_state = 0)

#===========================================================
# KNN Classifier
print('\nKNN Classifier:\n')
from sklearn.neighbors import KNeighborsClassifier
for k in [3,5,7,9]:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print('\nKNN Classifier: K = ', k)
    print(classification_report(y_test, y_pred))

    print("KNN Accuracy Rate -> ",accuracy_score(y_pred, y_test) * 100)

#===========================================================
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
classifiers = [
    LogisticRegression(),
    # SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    # DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    # QuadraticDiscriminantAnalysis()
]

for classifier in classifiers:
    clf = classifier
    clf.fit(X_train, y_train.values.ravel())
    predictions = clf.predict(X_test)
    print(classification_report(y_test, predictions))
    print(classifier," Rate -> ", accuracy_score(predictions, y_test) * 100)




#============================================================
# Getting the final result
targetDf = pd.read_csv("..\Dataset\TestFinalDS_AdditionalFeatures.csv")
X_target_raw = pd.DataFrame(df[list(selectedColumns)].values, columns = selectedColumns, index = df.index)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['token_texts']).toarray()
X_target = np.hstack((X[selectedColumns].values, X_tfidf))

clf = AdaBoostClassifier()
clf.fit(X_train, y_train.values.ravel())
predictions = clf.predict(X_test)
finalPredict = clf.predict(X_target)
print(finalPredict.shape)

labelColumn = []
labelColumn[:] = ['REAL' if x == 1 else 'FAKE' for x in finalPredict]

import pandas
df = pandas.DataFrame(data={"Final Labels": labelColumn})
df.to_csv("..\Result\FinalLabel.csv", sep=',',index=False)