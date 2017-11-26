#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    print "Entering outlierCleaner function"
    print len(predictions), len(ages), len(net_worths)
    cleaned_data = []

    ### your code goes here
    # calculate residual errors
    error = []
    for i in range(0,len(predictions)):
        error.append((predictions[i] - net_worths[i])**2)

    # identify 10 max
    idx = sorted(range(len(error)), key=error.__getitem__,  reverse=True)
    print idx
    print len(idx)
    for i in range(0,10):
        pt = idx[i]
        print i, ages[pt], net_worths[pt], error[pt]
    for i in range(10,len(idx)):
        pt = idx[i]
#        print i, ages[i], net_worths[i], error[i]
        cleaned_data.append((ages[pt], net_worths[pt], error[pt]))
    print len(cleaned_data)
    return cleaned_data
