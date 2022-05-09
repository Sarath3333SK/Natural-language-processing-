import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle

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

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

linear_clf = PassiveAggressiveClassifier(max_iter=50)

model=linear_clf.fit(tfidf_train, y_train)
filename = 'fakemodel.sav'
pickle.dump(model, open(filename, 'wb'))
pred = model.predict(tfidf_test)
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

test=tfidf_vectorizer.transform(["campaign ad president lyndon johnson featured purported republican voter expressing concerns eerily echoed threads debate gop march facebook page web site quartz shared video calledconfessions originally political advertisement clip rapidly gained along skepticism viewpoints expressed neatly echoed political schisms "])

gh=model.predict(test)
print(gh)
