#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# additionnal libraries required
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

print
print "------- Task 1 - Features selection ------"
print "Initial Dataset review"
print "Number of    features:", len(data_dict['SKILLING JEFFREY K'].keys())
print "Number of individuals:", len(data_dict)
print "Features names:"
pprint(data_dict['SKILLING JEFFREY K'].keys())
# create dataframe
names = data_dict.keys()
df_data = pd.DataFrame.from_dict(data_dict, orient = 'index', dtype = np.float)
print 'number of poi=', sum(df_data['poi'])

# Missing value audit
print
print 'Task 1.1 -- Auditing missing values'
print 'Calculating missing value ratio'
missing_rate = df_data.isna().sum(axis=0)/df_data.shape[0]*100.
missing_rate.sort_values(inplace = True)
nb_feat = len(missing_rate)
fig, ax = plt.subplots()
ax.bar(np.arange(nb_feat),  missing_rate)
ax.set_xticks(np.arange(nb_feat))
ax.set_xticklabels( missing_rate.index, rotation=90)
ax.set_ylabel('Percentage of NaN values')
ax.set_xlabel('Feature Names')
ax.set_title('Missing Values ratio')
plt.plot([0, nb_feat], [60, 60], color='Red')
plt.savefig('Missing_rate_features.png')
plt.close()
print missing_rate
print
print "Task 1.2 -- Dropping out features having more than 85% of values missing"
# dropping columns with more than 60% of data missing
idx = missing_rate[missing_rate>85.].index.tolist()
df_data = df_data.drop(idx, axis = 1)
print "Removed features:"
pprint(idx)
print "Remaining features:"
pprint(df_data.columns.tolist())

### Task 2: Remove outliers
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'] # Records that do not correspond to a physical person
print
print "Task 2 -- Dropping out individuals that are not real persons:"
print "Individuals removed:", outliers
df_data = df_data.drop(outliers)
print
print "End of Task 1 & Task 2 -- Dataset after features selection and outliers removal"
print "Number of individuals:", df_data.shape[0]
print "Number of    features:", df_data.shape[1]
print 'Number of         poi:', int(sum(df_data['poi']))

### Task 3: Create new feature(s)
print
print "------ Task 3 : New features creation ------"
print "Split DataSet between features and labels"
labels = df_data['poi']
features = df_data.drop(['poi', 'email_address'], axis = 1).fillna(0)
# import functions generating new features
from my_tools import transform_emails_df
from my_tools import transform_financial_df

print "Task 3.1 -- Introduce ratio on e-mails traffic"
features = transform_emails_df(features)
print "Dataset after ratio on e-mails introduction"
print "Number of individuals:", features.shape[0]
print "Number of    features:", features.shape[1]
print
print "Task 3.2 -- Introduce ratio on financial figures"
features = transform_financial_df(features)
print "Dataset after ratio on financial introduction"
print "Number of individuals:", features.shape[0]
print "Number of    features:", features.shape[1]
print
print
print "Task 3.3 -- Using selectKbest to evaluate feature importance"
# credit to https://datascience.stackexchange.com/questions/10773/how-does-selectkbest-work
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
selector = SelectKBest(k='all', score_func=chi2)
selector.fit(scaled_features, labels)
features_names = features.columns.tolist()

scores = -np.log10(selector.pvalues_)
idx = np.argsort(-scores)
print "idx=", idx
sorted_features_names = [features_names[i] for i in idx]
# Plot the scores
plt.bar(range(len(features_names)), -np.sort(-scores))
plt.xticks(range(len(features_names)), sorted_features_names, rotation='vertical',
           fontsize=8)
plt.ylabel('Score (-log10(p_value))')
plt.title("Features ranking")
#plt.show()
plt.tight_layout()
plt.savefig('Features ranking.png')
### Task 4: Try a varity of classifiers
print
print "------ Task 4 : Try a variety of classifiers ------"
print "Initialisation..."
# Comparison of classifiers
# credit to : http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# import classifiers to be tested
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# import pipeline related libraries
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler
# Create instances for pipeline
scaler = MinMaxScaler()
select = SelectKBest(k=10, score_func = chi2)

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "AdaBoost",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(min_samples_split=5),
    RandomForestClassifier(max_depth=10, n_estimators=20, max_features=1),
    AdaBoostClassifier(),
    GaussianNB()]

# Metrics used for classifiers comparison
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# test/train split
print "Train/test split"
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=4)#429
print "Training dataset dimensions:", features_train.shape
print "with ", sum(labels_train), " poi"
print "Testing  dataset dimensions:", features_test.shape
print "with ", sum(labels_test), " poi"

# loop on classifiers
results_compa = {}
for name, clf in zip(names, classifiers):
    pipe = Pipeline(steps=[('scaling', scaler), ('selection', select),
                            ('classifier', clf)])
    print '_______'
    print name
    #clf = DecisionTreeClassifier()
    print "training..."
    pipe.fit(features_train, labels_train)
    # Prediction
    print "Prediction..."
    predicted = pipe.predict(features_test)
    score = accuracy_score(predicted, labels_test)
    recall = recall_score(predicted, labels_test)
    precision = precision_score(predicted, labels_test)
    cm = confusion_matrix(labels_test, predicted)
    results_compa[name]= {'Confusion Matrix':cm, 'score':score, 'recall':recall,
                          'precision':precision}
    print 'Results'
    print "Accuracy :", score
    print "Recall   :", recall
    print "Precision:", precision
    print "Confusion matrix"
    pprint(cm)

print "Final results summary"
df_results_compa = pd.DataFrame.from_dict(results_compa, orient = 'index',
                                          dtype = np.float)
pprint(df_results_compa)
plt.close()
df_results_compa[['score', 'recall', 'precision']].plot(kind='bar')
plt.xticks(rotation='vertical',
           fontsize=14)
plt.tight_layout()
plt.savefig('compare_algorithms.png')

print
print "------ Task 5 : Tuning Classifier ------"
print

print "Building pipeline..."
scaler = MinMaxScaler()
select = SelectKBest(chi2)
adab = AdaBoostClassifier()
pipe = Pipeline(steps=[('scaling', scaler), ('selection', select),
                        ('classifier', adab)])
print "Testing pipeline..."
pipe.fit(features_train, labels_train)
predicted = pipe.predict(features_test)
print "OK"
print
print "GridSearchCV on pipeline..."
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

# parameters for GridSearchCV => several candidates tested
# Remark : parameters_pipe_full corresponds to the test described in
# the descriptive report of this analysis
parameters_pipe_full = {'selection__k':(2,4,6,8,10,15,20,26),
                   'classifier__n_estimators':(5, 10, 50, 100, 500, 1000),
                   'classifier__learning_rate':(0.25, 0.5, 1.0, 2.0, 5.0, 7.5)}
parameters_pipe_interm = {'selection__k':(2,4,8,10),
                   'classifier__n_estimators':(50, 100, 150),
                   'classifier__learning_rate':(0.5, 2., 4, 7.5)}
parameters_pipe_light = {'selection__k':(2,10),
                   'classifier__n_estimators':(50, 100),
                   'classifier__learning_rate':(1., 2.)}
parameters_pipe_final = {'selection__k':(10,15),
                   'classifier__n_estimators':(40,50,60),
                   'classifier__learning_rate':(1., 2.)}

parameters_pipe = parameters_pipe_final
print "Parameters used in gridsearchCV"
pprint(parameters_pipe)

shuffle = StratifiedShuffleSplit(labels, n_iter = 20, test_size = 0.5, random_state = 0)


clf = GridSearchCV(pipe, parameters_pipe, scoring=['f1', 'recall', 'precision'],
                   cv=shuffle, refit='f1',
                   verbose=2)
clf.fit(features, labels)
#print "GridSearch results"
#print pd.DataFrame.from_dict(clf.cv_results_)
print
print "Best parameters found:"
print clf.best_params_
# Results of GridsearchCV"
results_cv = clf.cv_results_
res_recall = results_cv['mean_test_recall']
res_precision = results_cv['mean_test_precision']
res_f1 = results_cv['mean_test_f1']
par_k = results_cv['param_selection__k']
par_learning_rate = results_cv['param_classifier__learning_rate']
par_n_estimators = results_cv['param_classifier__n_estimators']
# Storage in dataframe for easier treatment/plot
df_results_gscv = pd.DataFrame({'k':par_k,
                                'Learning_Rate':par_learning_rate,
                                'N_Estimators':par_n_estimators,
                                'F1':res_f1,
                                'Recall':res_recall,
                                'Precision':res_precision})
print
print "GridSearchCV results summary"
pprint(df_results_gscv)
print "Exporting graphs..."
# Export graphs for analysis of GridSearchCv results
sns.factorplot(x="Learning_Rate", y="F1",
                    hue="k", col="N_Estimators",
                    data=df_results_gscv, kind="strip",
                    jitter=True,
                    size=4, aspect=.7);
sns.despine()
plt.savefig('GSCV_F1.png')
#plt.show()
sns.factorplot(x="Learning_Rate", y="Recall",
                    hue="k", col="N_Estimators",
                    data=df_results_gscv, kind="strip",
                    jitter=True,
                    size=4, aspect=.7);
sns.despine()
plt.savefig('GSCV_Recall.png')
#plt.show()
sns.factorplot(x="Learning_Rate", y="Precision",
                    hue="k", col="N_Estimators",
                    data=df_results_gscv, kind="strip",
                    jitter=True,
                    size=4, aspect=.7);
sns.despine()
plt.savefig('GSCV_Precision.png')
#plt.show()
print "Done"

print
print "------ Task 6 : Dumping results ------"
print

# Way to extract useful information from gridsearchCV and pipeline
# https://discussions.udacity.com/t/what-should-i-do-next/196671/7
extract = clf.best_estimator_.named_steps['selection']
features_selected=extract.get_support(indices=True)
features_selected = features_selected + 1
features['poi'] = labels
complete_features_list = (features.columns.values)
features_output = [complete_features_list[i] for i in features_selected]
features_output.insert(0, 'poi')
print
print "Selected features names:"
pprint(features_output)

my_dataset = features.to_dict('index')

# dumping classifier & all to be able to run tester.py
dump_classifier_and_data(clf.best_estimator_, my_dataset, features_output)
print
print "Analysis completed"
