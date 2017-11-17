#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
# sk-learn import
from sklearn import svm
from sklearn.metrics import accuracy_score
#
# sample training datasets
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
# training
clf = svm.SVC(kernel = 'rbf', C = 10000.)
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
# prediction
t0 = time()
predicted = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"
# accuracy
accuracy = accuracy_score(labels_test, predicted)
print 'accuracy=', accuracy

print '10', predicted[10]
print '26', predicted[26]
print '50', predicted[50]
print ''
print 'Total of Chris emails', predicted.sum()
#########################################################
