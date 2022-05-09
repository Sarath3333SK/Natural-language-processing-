import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
df=pd.read_csv('news.csv', index_col=None)
#Get shape and head
df.shape
df.head()
df = df.set_index('Unnamed: 0')
#DataFlair - Get the labels
labels=df.label
labels.head()


y = df.label

df = df.drop('label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
print(count_train)
count_test = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

hash_vectorizer = HashingVectorizer(stop_words='english',alternate_sign=False)
hash_train = hash_vectorizer.fit_transform(X_train)
hash_test = hash_vectorizer.transform(X_test)



clf = MultinomialNB()

clf.fit(tfidf_train, y_train)
pred = clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)

cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])



clf_count = MultinomialNB()

clf_count.fit(count_train, y_train)
pred = clf_count.predict(count_test)
score = metrics.accuracy_score(y_test, pred)
reportcVm=metrics.classification_report(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])



linear_clf = PassiveAggressiveClassifier(max_iter=50)

linear_clf.fit(tfidf_train, y_train)
pred = linear_clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])


linear_clf_count = PassiveAggressiveClassifier(max_iter=50)
linear_clf_count.fit(count_train, y_train)
pred = linear_clf_count.predict(count_test)
score = metrics.accuracy_score(y_test, pred)
reportcVm=metrics.classification_report(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])

clf = MultinomialNB(alpha=.01)

clf.fit(hash_train, y_train)
pred = clf.predict(hash_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])


clf = PassiveAggressiveClassifier(max_iter=50)

clf.fit(hash_train, y_train)
pred = clf.predict(hash_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])


def most_informative_feature_for_binary_classification(vectorizer, classifier, n=100):
    
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    for coef, feat in topn_class1:
        #classs=class_labels[0] 
        #co=coef 
        #name=feat
        print(class_labels[0], coef, feat)
       

    for coef, feat in reversed(topn_class2):
        print(class_labels[1], coef, feat)
    return class_labels[0],class_labels[1],topn_class1,topn_class2
    
        

label1,label2,clas1,clas2=most_informative_feature_for_binary_classification(tfidf_vectorizer, linear_clf, n=30)

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':150})

import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Real', 'Fake']); ax.yaxis.set_ticklabels(['Real', 'Fake']);
plt.figure()
plt.savefig("cm.png" )









