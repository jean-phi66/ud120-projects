#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop( 'TOTAL', 0 )
data = featureFormat(data_dict, features)


### your code below
maxi = 0
for point in data:
    salary = point[0]
    bonus = point[1]
    if salary > maxi:
        maxi = salary
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

for k,v in data_dict.iteritems():
    if v['salary'] == maxi:
        print 'outlier name:',k

for k,v in data_dict.iteritems():
    if v['salary'] > 1000000. and v['bonus'] > 5000000.:
        if(type(v['salary'])==int):
            print 'Bandit:', k, v['salary'], v['bonus']
