import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler

def scaleData(df):
    cat = [var for var in df.columns if not(np.issubdtype(df[var].dtype,np.number))]
    num = df.drop(cat,axis=1)
    # Creating dictionary to store the different data frames
    data = {"original":df}
    # Standarizing data to have mean 0 and variance 1
    scaler = StandardScaler()
    scaler.fit(num)
    data["standarized"] = pd.DataFrame(scaler.transform(num),index=num.index,columns=num.columns)
    data["standarized"][cat] = df[cat]
    data["standarized"] = data["standarized"][df.columns]
    # Centering data to have variance 1 
    scaler = StandardScaler(with_mean=False)
    scaler.fit(num)
    data["withmean"] = pd.DataFrame(scaler.transform(num),index=num.index,columns=num.columns)
    data["withmean"][cat] = df[cat]
    data["withmean"] = data["withmean"][df.columns]
    return data

def boxcoxLambdaTable(X,alpha=0.05):
    names = []
    lambdas = []
    intervalsBot = []
    intervalsTop = []
    for col in X.columns:
        if np.issubdtype(df[col].dtype,np.number):
            if (df[col]>0).prod():
                names.append(col)
                bx = boxcox(df[col],alpha=alpha)
                lambdas.append(bx[1])
                intervalsBot.append(bx[2][0])
                intervalsTop.append(bx[2][1])
            else:
                print("Can't convert column {0}: not entirely positive".format(col) )
    fin = pd.DataFrame.from_dict({"lambda":lambdas,"Lower confidence interval, alpha = {0}".format(alpha):intervalsBot,"Upper confidence interval, alpha = {0}".format(alpha):intervalsTop})
    fin.index = names
    return fin.transpose()


def getRDataset(name,package=None,verbose=False,index_col=0):
    db = pd.read_csv("http://vincentarelbundock.github.com/Rdatasets/datasets.csv")
    reduc = db[db["Item"]==name]
    if reduc.shape[0]==0:
        raise ValueError("No dataset with name {0} was found".format(name))
    elif reduc.shape[0] > 1 and package is None:
        raise ValueError("Dataset with name {0} is available in packages {1}. Please specify package with argument \"package\"".format(name,reduc["Package"].to_list()))
    elif not(package is None):
        if not(package in set(db["Package"])):
            raise ValueError("No package with name {0} exists".format(package))
        reduc = reduc[reduc["Package"]==package]
        if reduc.shape[0]==0:
            raise ValueError("Dataset {0} in package {1} not found".format(name,package))
    # Converting DataFrame to series
    reduc = reduc.iloc[0]
    if verbose:
        print("Name of Dataset: {0}".format(name))
        print("Name of Package: {0}".format(package))
        print("Description: {0}".format(reduc["Title"]))
        print("Rows: {0}".format(reduc["Rows"]))
        print("Columns: {0}".format(reduc["Cols"]))
        print("Link for donwload: {0}".format(reduc["CSV"]))    
    return pd.read_csv(reduc["CSV"],index_col=index_col)

def makeDataFrame(X,y):
    if X.shape[0]!=len(y):
        raise ValueError("Observations in X and y do not match")
    varNames = ["x{0}".format(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X)
    df.columns = varNames
    df["y"] = y
    return df

def addInteractions(X,squares=False):
    aux = X.copy()
    interactions = []
    errors = []
    for index1,col1 in enumerate(X.columns):
        if col1 != "const":
            for index2,col2 in enumerate(X.columns[index1:]):
                if col2 != "const" and (col1!=col2 or squares):
                    inter = "{0}:{1}".format(col1,col2)
                    interactions.append(inter)
                    aux[inter] = X[col1]*X[col2]
    return aux