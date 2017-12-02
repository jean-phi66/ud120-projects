#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### split train/test dataset
from sklearn.model_selection import train_test_split
input_train, input_test, labels_train, labels_test = train_test_split(features,
labels, test_size = 0.3, random_state = 42)
print "train dataset size", len(labels_train)
print "test  dataset size:", len(labels_test)

### Decision tree Classifier first attempt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf2 = DecisionTreeClassifier()
clf2.fit(input_train, labels_train)
predicted = clf2.predict(input_test)
print "Accuracy:", float(accuracy_score(labels_test, predicted))

# Investigate test set content
print "Number of POI in testing set:", sum(labels_test)

# check if we have some true positive
TP = [x for x in labels_test == predicted if labels_test ==1.]
print TP

# display recall and presicion values to have better insight
from sklearn.metrics import confusion_matrix, recall_score, precision_score
print "Recall    score:", recall_score(predicted, labels_test)
print "Precision score:", precision_score(predicted, labels_test)
print confusion_matrix(predicted, labels_test)

# made up results
print "Made up results for practice"
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print "Confusion matrix:", confusion_matrix(true_labels, predictions)
print "Recall    score:", recall_score(true_labels, predictions)
print "Precision score:", precision_score(true_labels, predictions)
