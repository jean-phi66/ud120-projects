#!/usr/bin/python


# function performing PCA for outliers detection
# performs PCA and display individual graph with names and color
# code for poi (red = poi, blue = non_poi)
def PCA_for_outliers(df, df_data, show_plot=True, n_comp=2, verbose=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd


    # another reference for PCA https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn
    from sklearn.decomposition import PCA as sklearnPCA
    sklearn_pca = sklearnPCA(n_components=n_comp, whiten = True)
    Y_sklearn = sklearn_pca.fit_transform(df)
    if verbose:
        print 'PCA dimension:', Y_sklearn.shape
        print "df.columns=", df.columns
    # display Principal Components vs initial variables
    # credit to https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn
    index_print = list()
    for i in range(n_comp):
        index_print.append('PC-' + str(i))
    if verbose:
        print index_print
        print pd.DataFrame(sklearn_pca.components_,columns=df.columns,
                           index = index_print)
    # Plot of individuals
    # PCA and plot reference : http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html#exploratory-visualization
    # color poi : red non-poi:blue
    # credit to https://stackoverflow.com/questions/9470056/learning-python-changing-value-in-list-based-on-condition
    reps = {1.0: 'red', 0.0: 'blue'}
    col = [reps.get(x,x) for x in df_data['poi']]
    names = df.index.tolist()
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        plt.scatter(Y_sklearn[:, 0],
                    Y_sklearn[:, 1], c=col)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(loc='lower center')
        plt.tight_layout()
        # annotation
        cpt =0
        for name in names:
            #if col[cpt]=='red':
            plt.text(Y_sklearn[cpt,0],Y_sklearn[cpt,1], df.index.values[cpt], size=7.5,
            ha='center', va='top')
            cpt += 1
        plt.show()
    return sklearn_pca

def split_dataset(df):
    financials = ['salary','deferral_payments', 'total_payments','exercised_stock_options',
     'bonus','restricted_stock','restricted_stock_deferred','total_stock_value',
     'expenses', 'other', 'director_fees', 'deferred_income',
     'long_term_incentive']
# removed:     loan_advances
    emails = ['to_messages','shared_receipt_with_poi','from_messages',
     'from_this_person_to_poi','from_poi_to_this_person', 'interaction_with_poi']
    df_financial = df
    df_emails = df
    for email in emails:
        if email in df_financial.columns:
             df_financial = df_financial.drop(email, axis = 1, errors = 'ignore')
    for financial in financials:
        if financial in df_emails.columns:
            df_emails = df_emails.drop(financial, axis = 1, errors = 'ignore')
    return [df_financial, df_emails]



def index_in_list(lst, value):
    return [i for i, x in enumerate(lst) if x==value]

def transform_emails_df(df_in):
    df = df_in.copy()
    df['total_messages'] = df['to_messages'] + df['from_messages']
    df['interaction_with_poi'] = (df['shared_receipt_with_poi'] +
                                       df['from_this_person_to_poi'] +
                                       df['from_poi_to_this_person'])/df.total_messages
    df['shared_receipt_with_poi_ratio'] = df.shared_receipt_with_poi / df.to_messages
    df['from_this_person_to_poi_ratio'] = df.from_this_person_to_poi / df.from_messages
    df['from_poi_to_this_person_ratio'] = df.from_poi_to_this_person / df.to_messages
    df['from_messages_ratio'] = df.from_messages / df.total_messages
    df['to_messages_ratio'] = df.to_messages / df.total_messages
    df = df.fillna(0.)
    return df
def transform_financial_df(df_in):
    df = df_in.copy()
    df['salary_ratio'] = df.salary / df.total_payments
    df['bonus_ratio'] = df.bonus / df.total_payments
    df['deferral_payments_ratio'] = df.deferral_payments / df.total_payments
    #df['exercised_stock_options_ratio'] = df.exercised_stock_options / df.total_payments
    #df['total_stock_value_ratio'] = df.total_stock_value / df.total_payments
    #df['restricted_stock_deferred'] = df.restricted_stock_deferred / df.total_payments
    df = df.fillna(0.)
    return df
#    complete_features_list = ['poi','salary','to_messages','deferral_payments',
#     'total_payments','exercised_stock_options',
#     'bonus','restricted_stock','shared_receipt_with_poi','restricted_stock_deferred','total_stock_value',
#     'expenses', 'loan_advances', 'from_messages', 'other',
#     'from_this_person_to_poi', 'director_fees', 'deferred_income',
#     'long_term_incentive', 'email_address', 'from_poi_to_this_person']
