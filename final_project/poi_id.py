#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import Imputer

sys.path.append("../tools/")
sys.path.append(".")

from my_tools import PCA_for_outliers
from my_tools import index_in_list
from my_tools import split_dataset


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# complete feature list:
complete_features_list = ['poi','salary','to_messages','deferral_payments',
 'total_payments','exercised_stock_options',
 'bonus','restricted_stock','shared_receipt_with_poi','restricted_stock_deferred','total_stock_value',
 'expenses', 'loan_advances', 'from_messages', 'other',
 'from_this_person_to_poi', 'director_fees', 'deferred_income',
 'long_term_incentive', 'email_address', 'from_poi_to_this_person']
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# List feature names (dict keys)
pprint(data_dict['SKILLING JEFFREY K'].keys())
print "Number of    features:", len(data_dict['SKILLING JEFFREY K'].keys())
print "Number of individuals:", len(data_dict)
# create dataframe
names = data_dict.keys()
df_data = pd.DataFrame.from_dict(data_dict, orient = 'index', dtype = np.float)
df_data.to_csv("ENRON_dataset.csv") # export for external data review/X-check

# Imputing values to enable numeric analysis (clustering & PCA)
# Imputation ref : https://stackoverflow.com/questions/29420737/pca-with-missing-values-in-python
df_for_Imputation = df_data.drop(['poi', 'email_address', 'loan_advances'], axis = 1)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
df_Imputed = imp.fit_transform(df_for_Imputation)
df_Imputed = pd.DataFrame(df_Imputed, index=df_for_Imputation.index,
                                      columns=df_for_Imputation.columns)
if(False):
    # correlation matrix
    # credit to : https://stackoverflow.com/questions/29432629/correlation-matrix-using-pandas
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df_Imputed.corr()
    corplot = sns.heatmap(corr, linewidths=.5)
    corplot.set_yticklabels(corplot.get_yticklabels(), rotation = 0, fontsize = 8)
    corplot.set_xticklabels(corplot.get_xticklabels(), rotation = 90, fontsize = 8)
    plt.show()
# clustermap
# credit to : http://seaborn.pydata.org/examples/structured_heatmap.html
clustmap = sns.clustermap(df_Imputed.corr(), center=0)
plt.setp(clustmap.ax_heatmap.xaxis.get_majorticklabels(), rotation=60, fontsize=6)
plt.setp(clustmap.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.show()
# Definition of two variable families based on clustering seen
df_financial, df_emails = split_dataset(df_Imputed)
# TO BE TESTED : Facet Plot
# http://seaborn.pydata.org/generated/seaborn.FacetGrid.html

PCA_financial = PCA_for_outliers(df_financial, df_data, n_comp=5)
PCA_emails = PCA_for_outliers(df_emails, df_data, n_comp=5, verbose=True)


outliers = ['TOTAL']
df_data = df_data.drop(outliers)
df_for_Imputation = df_data.drop(['poi', 'email_address', 'loan_advances'], axis = 1)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
df_Imputed = imp.fit_transform(df_for_Imputation)
df_Imputed = pd.DataFrame(df_Imputed, index=df_for_Imputation.index,
                                      columns=df_for_Imputation.columns)
# Definition of two variable families based on clustering seen
df_financial, df_emails = split_dataset(df_Imputed)

PCA_financial = PCA_for_outliers(df_financial, df_data, n_comp=5)
PCA_emails = PCA_for_outliers(df_emails, df_data, n_comp=5)
def plot_var_exp(PCA, title_str):
    var_expl = PCA.explained_variance_ratio_
    print range(len(var_expl)), var_expl
    fig, ax = plt.subplots()
    ax.bar(range(len(var_expl)), var_expl )
    ax.set_ylabel('Variance ratio explained')
    ax.set_xlabel('Principal Component')
    ax.set_title(title_str)
    plt.show()
    return var_expl

plot_var_exp(PCA_financial, "PCA - Financial")
plot_var_exp(PCA_emails, "PCA - Emails")

# From this analysis:
# retains 3 first PC for financial-related variables
# retains 3 first PC for emails-related variables
#
# create basis dataset and split feature/labels
df_data = pd.DataFrame.from_dict(data_dict, orient = 'index', dtype = np.float)
df_for_ML = df_data.drop(['email_address','loan_advances'], axis = 1)
df_for_ML = df_for_ML.drop(outliers)
labels = df_for_ML['poi']
df_for_ML = df_for_ML.drop('poi', axis=1)

#train/test split of initial dataset
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(df_for_ML, labels, test_size=0.3, random_state=420)
print "Initial dataset size", len(labels)
print "Training dataset size", len(labels_train)
print "Testing dataset size", len(labels_test)
# Imputation on training dataset
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
df_Imputed = imp.fit_transform(features_train)
print 'features_train size', features_train.shape
print 'df_Imputed     size', df_Imputed.shape
df_Imputed = pd.DataFrame(df_Imputed, index=features_train.index,
                                      columns=features_train.columns)
# Definition of two variable families based on clustering seen
features_train_financial, features_train_emails = split_dataset(df_Imputed)

# PCA on financial variables
from sklearn.decomposition import PCA as sklearnPCA
n_comp = 3
pca_train_financial = sklearnPCA(n_components=n_comp, whiten = True)
transformed_features_train_financial = pca_train_financial.fit_transform(features_train_financial)
print 'transformed_features_train_financial', transformed_features_train_financial.shape
# PCA on emails variables
n_comp = 3
pca_train_emails = sklearnPCA(n_components=n_comp, whiten = True)
transformed_features_train_emails = pca_train_emails.fit_transform(features_train_emails)
print 'transformed_features_train_emails size:', transformed_features_train_emails.shape
# Merge two features training sets
transformed_features_train = np.concatenate((transformed_features_train_financial,
                                            transformed_features_train_emails),
                                            axis=1)
print 'transformed_features_train size:', transformed_features_train.shape
### Task 2: Remove outliers

# Train classifiers
# Test 1 : Decision tree
# Training
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(transformed_features_train, labels_train)
# Testing
imp_test = Imputer(missing_values='NaN', strategy='mean', axis=0)
print 'features_test           size', features_test.shape
print features_test
df_Imputed = imp_test.fit_transform(features_test)
print 'df_Imputed size', df_Imputed.shape
#print 'df_imputed', df_Imputed
df_Imputed = pd.DataFrame(df_Imputed, index=features_test.index,
                                      columns=features_test.columns)
features_test_financial, features_test_emails = split_dataset(df_Imputed)
print 'features_test           size', features_test.shape
print 'features_test_financial size', features_test_financial.shape
print 'features_test_emails    size', features_test_emails.shape

# PCA application
transformed_features_test_financial = pca_train_financial.transform(features_test_financial)
transformed_features_test_emails = pca_train_emails.transform(features_test_emails)
# Merge two features testing sets
transformed_features_test = np.concatenate((transformed_features_test_financial,
                                            transformed_features_test_emails),
                                            axis=1)

# Prediction
predicted = clf.predict(transformed_features_test)
print "Accuracy=", accuracy_score(predicted, labels_test)



### TO BE CONSIDERED : use of pipeline and featureunion
# https://stackoverflow.com/questions/36113686/multiple-pipelines-that-merge-within-a-sklearn-pipeline
# http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
# https://medium.com/@literallywords/sklearn-identity-transformer-fcc18bac0e98
if(False):
    my_dataset = data_dict
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features, labels)
    print "Accuracy=", accuracy_score(clf.predict(features), labels)


    ### Task 3: Create new feature(s)
    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html

    # Provided to give you a starting point. Try a variety of classifiers.
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()

    ### Task 5: Tune your classifier to achieve better than .3 precision and recall
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info:
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

    # Example starting point. Try investigating other evaluation techniques!
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.

    dump_classifier_and_data(clf, my_dataset, features_list)
