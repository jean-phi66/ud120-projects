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
    df = df.fillna(0.)
    return df

def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    plt.close()
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.savefig('Griv.png')
    plt.show()
    return
