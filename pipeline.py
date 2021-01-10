import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import (plot_confusion_matrix, classification_report, plot_roc_curve,
                            plot_precision_recall_curve)
import datetime

# Config: Dictionaries of models and hyperparameters
MODELS = {
    'LogisticRegression': LogisticRegression(), 
    'LinearSVC': LinearSVC(), 
    'GaussianNB': GaussianNB()
}

GRID = {
    'LogisticRegression': [{'penalty': x, 'C': y, 'random_state': 0} 
                           for x in ('l2', 'none') \
                           for y in (0.01, 0.1, 1, 10, 100)],
    'GaussianNB': [{'priors': None}],
    'LinearSVC': [{'C': x, 'random_state': 0} \
                  for x in (0.01, 0.1, 1, 10, 100)]
}


def import_data(filename):
    return pd.read_csv(filename)


def describe_columns(df):
    x = df.corr()
    f = plt.figure(figsize=(12,10))
    plt.matshow(x, fignum=f.number)
    plt.xticks(range(x.shape[1]), x.columns, fontsize=12)
    plt.yticks(range(x.shape[1]), x.columns, fontsize=12)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    return df.describe()

def feature_distributions(df, column_names):
    df = df[column_names]
    dfm = df.melt(var_name='columns')
    g = sns.FacetGrid(dfm, col='columns')
    g = (g.map(sns.distplot, 'value'))
    return None

def train_test(df, test_size, random_state):
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test

def boolean_to_numeric(df, colname):
    return df[colname].astype(int)

def impute_values(df, colnames, method='median'):
    '''
    imputes missing values using either the mean or median of the column
    '''
    new_df = df.copy()
    vals = {}
    for i in colnames:
        if method == 'median':
            vals[i] = vals.get(i, 0) + new_df[i].median()
        else:
            vals[i] = vals.get(i, 0) + new_df[i].mean()

    return new_df.fillna(value=vals)
        

def normalize(list_of_colnames, training_data, testing_data):
    train_df_norm = training_data.drop(columns = list_of_colnames)
    test_df_norm = testing_data.drop(columns = list_of_colnames)
    for i in list_of_colnames:
        norm_train = (training_data.loc[:,i] - training_data[i].mean())/training_data[i].std(ddof=0)
        norm_test = (testing_data.loc[:,i] - training_data[i].mean())/training_data[i].std(ddof=0)
        train_df_norm[i] = norm_train
        test_df_norm[i] = norm_test
    return train_df_norm, test_df_norm

def encode_categoricals(train_df, test_df, cat_columns):
    train_processed = pd.get_dummies(train_df, prefix_sep="__", columns=cat_columns)
    cat_dummies = [col for col in train_processed 
                   if "__" in col 
                   and col.split("__")[0] in cat_columns]
    processed_columns = list(train_processed.columns[:])
    df_test_processed = pd.get_dummies(test_df, prefix_sep="__", 
                                   columns=cat_columns)
    for col in df_test_processed.columns:
        if ("__" in col) and (col.split("__")[0] in cat_columns) and col not in cat_dummies:
            df_test_processed.drop(col, axis=1, inplace=True)
    for col in cat_dummies: #handles case where a value is present in training data, but not testing
        if col not in df_test_processed.columns:
            df_test_processed[col] = 0
    df_test_processed = df_test_processed[processed_columns]
    return train_processed, df_test_processed

def make_models(train_df, test_df, target_col):
    # Begin timer 
    start = datetime.datetime.now()

    # Initialize results data frame 
    results = pd.DataFrame()

    # Loop over models 
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    for model_key in MODELS.keys():
    
        # Loop over parameters 
        for params in GRID[model_key]: 
            print("Training model:", model_key, "|", params)
        
            # Create model 
            model = MODELS[model_key]
            model.set_params(**params)
        
            model.fit(X_train, y_train)
        
            # Predict on testing set 
            y_pred = model.predict(X_test)
        
            # Evaluate predictions 
            acc_score = metrics.accuracy_score(y_test, y_pred)
            prec_score = metrics.precision_score(y_test, y_pred)
            rec_score = metrics.recall_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred)
        
            # Store results in your results data frame 
            new_row = {'model': model_key, 'parameters': params, 'accuracy': acc_score,
                  'precision': prec_score, 'recall': rec_score, 'f1':f1}
            results = results.append(new_row, ignore_index=True)

        
    # End timer
    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)
    return results