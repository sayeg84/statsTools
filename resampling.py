from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import resample

def bootstrap(df):
    return resample(df,n_samples=df.shape[0]).reset_index(drop=True)

def kfold_strat(X,y,**kwargs):
    splitter = StratifiedKFold(**kwargs)
    iterator = splitter.split(X,y)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for train_index, test_index in iterator:
        X_train.append(X.iloc[train_index,:])
        y_train.append(y.iloc[train_index])
        X_test.append(X.iloc[test_index,:])
        y_test.append(y.iloc[test_index])
    return X_train,X_test,y_train,y_test

def kfold(X,y,**kwargs):
    splitter = KFold(**kwargs)
    iterator = splitter.split(X,y)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for train_index, test_index in iterator:
        X_train.append(X.iloc[train_index,:])
        y_train.append(y.iloc[train_index])
        X_test.append(X.iloc[test_index,:])
        y_test.append(y.iloc[test_index])
    return X_train,X_test,y_train,y_test
    