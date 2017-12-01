#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]

# decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
predicted_train = clf.predict(features_train)
predicted_test = clf.predict(features_test)
print "## Datasets dimensions ##"
print "Number of training data points", len(labels_train)
print "Number of test     data points", len(labels_test)
print
print "## Accuracy scores ##"
print "Accuracy score on training set:", accuracy_score(labels_train, predicted_train)
print "Accuracy score on test     set:", accuracy_score(labels_test, predicted_test)
print
print "## Features importance ##"
feat_class = clf.feature_importances_
vocab = vectorizer.get_feature_names()
cpt =0
for feat in feat_class:
    if feat >.2 :
        print "Feature #:", cpt,"Importance:", feat, "word:", vocab[cpt]
        impt_feat_num = cpt
    cpt += 1
print vocab[impt_feat_num]
