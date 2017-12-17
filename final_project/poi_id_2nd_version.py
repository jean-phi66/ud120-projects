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
from my_tools import transform_emails_df


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
print
print "------- Task 1 - Features selection ------"
print "Dataset review"
print "Number of    features:", len(data_dict['SKILLING JEFFREY K'].keys())
print "Number of individuals:", len(data_dict)
print "Features names:"
pprint(data_dict['SKILLING JEFFREY K'].keys())
# create dataframe
names = data_dict.keys()
df_data = pd.DataFrame.from_dict(data_dict, orient = 'index', dtype = np.float)
#df_data.to_csv("ENRON_dataset.csv") # export for external data review/X-check
print 'number of poi=', sum(df_data['poi'])
# Review missing values
print
print 'Auditing missing values'
missing_rate = df_data.isna().sum(axis=0)/df_data.shape[0]*100.
missing_rate.sort_values(inplace = True)
nb_feat = len(missing_rate)
if(False):
    fig, ax = plt.subplots()
    ax.bar(np.arange(nb_feat),  missing_rate)
    ax.set_xticks(np.arange(nb_feat))
    ax.set_xticklabels( missing_rate.index, rotation=90)
    ax.set_ylabel('Percentage of NaN values')
    ax.set_xlabel('Feature Names')
    ax.set_title('Missing Values ratio')
    plt.plot([0, nb_feat], [60, 60], color='Red')
    plt.show()
print missing_rate
print
print "Dropping out features having more than 60% of values missing"
# dropping columns with more than 60% of data missing
idx = missing_rate[missing_rate>60.].index.tolist()
df_data = df_data.drop(idx, axis = 1)
#
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'] # Records that do not correspond to a physical person
print "Dropping out individuals that are not real persons:", outliers
df_data = df_data.drop(outliers)
print
print "Dataset after first cleaning"
print "Number of individuals:", df_data.shape[0]
print "Number of    features:", df_data.shape[1]
print 'Number of         poi:', sum(df_data['poi'])
print
print "Using SelectPercentile as feature selection method"
### Store to my_dataset for easy export below.
#my_dataset = data_dict
### Extract features and labels from dataset for local testing
features_list = ['poi','salary','to_messages','deferral_payments',
 'total_payments','exercised_stock_options',
 'bonus','restricted_stock','shared_receipt_with_poi','restricted_stock_deferred','total_stock_value',
 'expenses', 'loan_advances', 'from_messages', 'other',
 'from_this_person_to_poi', 'director_fees', 'deferred_income',
 'long_term_incentive', 'from_poi_to_this_person']
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

labels = df_data['poi']
features = df_data.drop(['poi', 'email_address'], axis = 1).fillna(0)

from sklearn.feature_selection import SelectPercentile, f_classif
selector = SelectPercentile(f_classif, percentile=90)
selector.fit(features, labels)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
# credit to : https://stackoverflow.com/questions/41724432/ml-getting-feature-names-after-feature-selection-selectpercentile-python
columns = np.asarray(features.columns.values)
support = np.asarray(selector.get_support())
columns_with_support = columns[support]
print "Selected features:", columns_with_support
df_for_ML = features[columns_with_support]
print "Dataframe for ML dimension:", df_for_ML.shape
print "Scaling features"
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(features)
featured_scaled = scaler.transform(features)
#features = pd.DataFrame(data=featured_scaled, index=features.index, columns=features.columns)
### Task 2: Remove outliers
print "------ Task 2 : Outliers removal ------"
print "Done..."
### Task 3: Create new feature(s)
print "------ Task 3 : New features creation ------"
print "On hold..."
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
print "------ Task 4 : Try a variety of classifiers ------"
print "Splitting dataset in training/testing subsets"
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(df_for_ML, labels, test_size=0.3, random_state=42)#429
print "Training dataset dimensions:", features_train.shape
print "with ", sum(labels_train), " poi"
print "Testing  dataset dimensions:", features_test.shape
print "with ", sum(labels_test), " poi"
# Comparison of classifiers
# credit to : http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
#    SVC(kernel="linear", C=0.025),
#    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(min_samples_split=5),
    RandomForestClassifier(max_depth=10, n_estimators=20, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(n_estimators = 100, learning_rate = 5.,
                       algorithm = 'SAMME'),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
results_compa = {}
for name, clf in zip(names, classifiers):
    print '_______'
    print name
    #clf = DecisionTreeClassifier()
    print "training..."
    clf.fit(features_train, labels_train)
    # Prediction
    print "Prediction..."
    predicted = clf.predict(features_test)
    score = accuracy_score(predicted, labels_test)
    results_compa[name]= score
    recall = recall_score(predicted, labels_test)
    precision = precision_score(predicted, labels_test)
    print 'Results'
    print "Accuracy :", score
    print "Recall   :", recall
    print "Precision:", precision
    print "Confusion matrix"
    pprint(confusion_matrix(labels_test, predicted))

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
print "------ Task 5 : Tuning Classifier ------"
from sklearn.model_selection import GridSearchCV
parameters = {#'base_estimator':('DecisionTreeClassifier', 'RandomForestClassifier', 'SVC'),
              'n_estimators':(50, 100, 500, 1000),
              'learning_rate':(0.5, 1.0, 2.0, 5.0, 7.5)}
adab = AdaBoostClassifier(algorithm='SAMME')
clf = GridSearchCV(adab, parameters, scoring='f1')
clf.fit(features_train, labels_train)
#print "GridSearch results"
#print pd.DataFrame.from_dict(clf.cv_results_)
print "Best parameters found:"
print clf.best_params_

print "Use of tuned parameters for cross-validation"
clf_tuned = AdaBoostClassifier(n_estimators=clf.best_params_['n_estimators'],
                               learning_rate=clf.best_params_['learning_rate'],
                               algorithm='SAMME')
clf_tuned.fit(features_train, labels_train)
predicted = clf_tuned.predict(features_test)
score = accuracy_score(predicted, labels_test)
recall = recall_score(predicted, labels_test)
precision = precision_score(predicted, labels_test)
print 'Results of tuned classifier'
print "Accuracy :", score
print "Recall   :", recall
print "Precision:", precision
print "Confusion matrix"
pprint(confusion_matrix(labels_test, predicted))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
features_list = columns_with_support#.tolist()#.insert(0, 'poi')
print "features_list type", type(features_list)
features_list = features_list.tolist()
print "features_list type after tolist", type(features_list), features_list
features_list.insert(0, 'poi')
print features_list, columns_with_support
my_dataset = data_dict
del my_dataset['TOTAL']
del my_dataset['THE TRAVEL AGENCY IN THE PARK']
dump_classifier_and_data(clf_tuned, my_dataset, features_list)

exit()
print df_data.describe()
# Imputing values to enable numeric analysis (clustering & PCA)
# Imputation ref : https://stackoverflow.com/questions/29420737/pca-with-missing-values-in-python
df_for_Imputation = df_data.drop(['email_address', 'loan_advances'], axis = 1)
df_for_Imputation['poi'] = df_for_Imputation.poi.astype(np.int)
df_for_Imputation['poi'] = df_for_Imputation.poi.astype('category')
print df_for_Imputation.dtypes
#df_for_Imputation = df_data.drop(['poi', 'email_address', 'loan_advances'], axis = 1)
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
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
if(False):
    clustmap = sns.clustermap(df_Imputed.corr(), center=0)
    plt.setp(clustmap.ax_heatmap.xaxis.get_majorticklabels(), rotation=60, fontsize=6)
    plt.setp(clustmap.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.show()

#pgrid = sns.PairGrid(df_Imputed)
#pgrid = pgrid.map_diag(plt.hist)
#pgrid = pgrid.map_offdiag(plt.scatter)


# Definition of two variable families based on clustering seen
df_financial, df_emails = split_dataset(df_Imputed)
poi_dict = {'0':'not-poi', '1':'poi'}
df_emails['poi'] = df_emails.poi.astype(np.int)
df_emails['poi'] = df_emails.poi.astype(str)
df_emails = df_emails.replace({'poi': poi_dict})
print df_emails.dtypes
print df_emails['poi']
#df_emails['poi'] = df_emails.poi.astype('category')
print "df_emails type"
print df_emails.dtypes
# TO BE TESTED : Facet Plot
# http://seaborn.pydata.org/generated/seaborn.FacetGrid.html
pgrid = sns.PairGrid(df_emails, hue = 'poi')
pgrid = pgrid.map_diag(plt.hist)
pgrid = pgrid.map_offdiag(plt.scatter)
#pgrid = pgrid.map(plt.scatter)
plt.show()

# normalization of df_emails by tot_messages
df_emails_norm = transform_emails_df(df_emails)
pgrid = sns.PairGrid(df_emails_norm, hue = 'poi')
pgrid = pgrid.map_diag(plt.hist)
pgrid = pgrid.map_offdiag(plt.scatter)
#pgrid = pgrid.map(plt.scatter)
plt.show()

df_financial = df_financial.drop(['poi'], axis = 1)
df_emails = df_emails.drop(['poi'], axis = 1)

PCA_financial = PCA_for_outliers(df_financial, df_data, n_comp=5)
PCA_emails = PCA_for_outliers(df_emails, df_data, n_comp=5, verbose=False)


outliers = ['TOTAL']
df_data = df_data.drop(outliers)
df_for_Imputation = df_data.drop(['poi', 'email_address', 'loan_advances'], axis = 1)
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
df_Imputed = imp.fit_transform(df_for_Imputation)
df_Imputed = pd.DataFrame(df_Imputed, index=df_for_Imputation.index,
                                      columns=df_for_Imputation.columns)
# Definition of two variable families based on clustering seen
df_financial, df_emails = split_dataset(df_Imputed)
df_emails_norm = transform_emails_df(df_emails)

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
    train_test_split(df_for_ML, labels, test_size=0.3, random_state=42)
print "Initial dataset size", len(labels)
print "Training dataset size", len(labels_train), 'with ', sum(labels_train), 'poi'
print "Testing dataset size", len(labels_test), 'with ', sum(labels_test), 'poi'
# Imputation on training dataset
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
df_Imputed = imp.fit_transform(features_train)
print 'features_train size', features_train.shape
print 'df_Imputed     size', df_Imputed.shape
df_Imputed = pd.DataFrame(df_Imputed, index=features_train.index,
                                      columns=features_train.columns)
# Definition of two variable families based on clustering seen
features_train_financial, features_train_emails = split_dataset(df_Imputed)
features_train_emails = transform_emails_df(features_train_emails)

# PCA on financial variables
from sklearn.decomposition import PCA as sklearnPCA
n_comp = 10
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

# Comparison of classifiers
# credit to : http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(min_samples_split=5),
    RandomForestClassifier(max_depth=10, n_estimators=20, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(n_estimators = 300, learning_rate = .75,
                       algorithm = 'SAMME'),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

# Preparation of testing dataset
imp_test = Imputer(missing_values='NaN', strategy='median', axis=0)
print 'features_test           size', features_test.shape
#print features_test
df_Imputed = imp_test.fit_transform(features_test)
print 'df_Imputed size', df_Imputed.shape
#print 'df_imputed', df_Imputed
df_Imputed = pd.DataFrame(df_Imputed, index=features_test.index,
                                      columns=features_test.columns)
features_test_financial, features_test_emails = split_dataset(df_Imputed)
features_test_emails = transform_emails_df(features_test_emails)

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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
results_compa = {}
for name, clf in zip(names, classifiers):
    #clf = DecisionTreeClassifier()
    clf.fit(transformed_features_train, labels_train)
    # Prediction
    predicted = clf.predict(transformed_features_test)
    score = accuracy_score(predicted, labels_test)
    results_compa[name]= score
    recall = recall_score(predicted, labels_test)
    precision = precision_score(predicted, labels_test)
    print '__'
    print name
    print "Accuracy :", score
    print "Recall   :", recall
    print "Precision:", precision
    print "Confusion matrix"
    pprint(confusion_matrix(labels_test, predicted))

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
