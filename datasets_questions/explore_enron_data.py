#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print "Enron Dataset type", type(enron_data)
print "Number of data points", len(enron_data)
print "Number of features per person", len(enron_data['ELLIOTT STEVEN'])
# count number of POI
cpt = 0
for k,v in enron_data.iteritems():
    if v['poi'] :
        cpt += 1
print "Number of POI in dataset:", cpt
print("names", enron_data.keys())
print enron_data['PRENTICE JAMES']['total_stock_value']
print enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print "Skilling :", enron_data['SKILLING JEFFREY K']
names = ['FASTOW ANDREW S', 'SKILLING JEFFREY K', 'LAY KENNETH L']
for name in names:
    print name, enron_data[name]['total_payments']
cpt = 0
for k,v in enron_data.iteritems():
    if v['salary'] != 'NaN':
        cpt += 1
print 'number with salary', cpt
cpt = 0
for k,v in enron_data.iteritems():
    if v['email_address'] != 'NaN':
        cpt += 1
print 'number with email_address', cpt
def missing_data(enron_data, data_name):
    cpt = 0
    for k,v in enron_data.iteritems():
        if v[data_name] == 'NaN':
            cpt += 1
    print 'missing', data_name, ':', cpt,'=>', float(cpt)/len(enron_data)*100, '%'
    return cpt

def missing_data_poi(enron_data, data_name):
    cpt = 0
    for k,v in enron_data.iteritems():
        if v[data_name] == 'NaN':
            if v['poi'] : cpt += 1
    print 'missing poi', data_name, ':', cpt,'=>', float(cpt)/len(enron_data)*100, '%'
    return cpt

missing_data(enron_data, 'total_payments')
missing_data_poi(enron_data, 'total_payments')



#['exercised_stock_options']
