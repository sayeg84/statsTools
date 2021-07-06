import pandas as pd
import itertools   
import numpy as np 
def drop1(model,X,y,errorFunc):
    base = errorFunc(model,X,y)
    vals = []
    for var in X.columns:
        aux = X.drop(var,axis=1)
        curr = errorFunc(model,aux,y)
        vals.append(curr)
    newInfo = pd.DataFrame.from_dict({"Dropped":X.columns,errorFunc.__name__:vals,"Change":[base - err for err in vals]})
    return newInfo

def dropRecursively(model,X,y,errorFunc,maxiter=10,verbose=False):
    X1 = X.copy()
    newInfo = drop1(model,X1,y,errorFunc)
    if verbose:
        print(newInfo)
    c = 0
    while newInfo["Change"].max()>0 and c < maxiter:
        droped = newInfo["Dropped"][newInfo["Change"].idxmax()]
        X1 = X1.drop(droped,axis=1)
        newInfo = drop1(model,X1,y,errorFunc)
        if verbose:
            print(newInfo)
        c += 1
    return X1

def add1(model,X,y,errorFunc):
    base = errorFunc(model,X,y)
    newVars = []
    interactions = []
    errors = []
    for index1,col1 in enumerate(X.columns):
        if col1 != "const":
            for index2,col2 in enumerate(X.columns[index1:]):
                if col2 != "const":
                    interactions.append("{0}:{1}".format(col1,col2))
    for inter in interactions:
        aux = X.copy()
        col1,col2 = inter.split(":")
        aux[inter] = aux[col1]*aux[col2]
        new = errorFunc(model,aux,y)
        errors.append(new)
    newInfo = pd.DataFrame.from_dict({"Added":interactions,errorFunc.__name__:errors,"Change":[base-val for val in errors]})
    return newInfo

def addRecursively(model,X,y,errorFunc,maxiter=10,verbose=False):
    X1 = X.copy()
    newInfo = add1(model,X1,y,errorFunc)
    if verbose:
        print(newInfo)
    c = 0
    while newInfo["Change"].max()>0 and c < maxiter:
        added = newInfo["Added"][newInfo["Change"].idxmax()]
        spl = added.split(":")
        col1,col2 = spl 
        X1["*".join(spl)] = X1[col1]*X1[col2]
        newInfo = add1(model,X1,y,errorFunc)
        if verbose:
            print(newInfo)
        c += 1
    return X1

def step(model,X,y,errorFunc,maxiter=10,verbose=False):
    X1 = X.copy()
    dropInfo = drop1(model,X1,y,errorFunc)
    addInfo = add1(model,X1,y,errorFunc)
    if verbose:
        print(dropInfo)
        print(addInfo)
    max1 = dropInfo["Change"].max()
    max2 = addInfo["Change"].max()
    totalmax = max(max1,max2)
    c = 0
    while totalmax > 0 and c < maxiter:
        if totalmax == max1:
            droped = dropInfo["Dropped"][dropInfo["Change"].idxmax()]
            X1 = X1.drop(droped,axis=1)
        else:
            added = addInfo["Added"][addInfo["Change"].idxmax()]
            spl = added.split(":")
            col1,col2 = spl 
            X1["*".join(spl)] = X1[col1]*X1[col2]
        dropInfo = drop1(model,X1,y,errorFunc)
        addInfo = add1(model,X1,y,errorFunc)
        if verbose:
            print(dropInfo)
            print(addInfo)
        max1 = dropInfo["Change"].max()
        max2 = addInfo["Change"].max()
        totalmax = max(max1,max2)
        c += 1
    return X1

def subsetComb(model,X,y,errorFunc,k=3):
    base = errorFunc(model,X,y)
    vals = []
    subsets = [list(sub) for sub in itertools.combinations(X.columns,k)]
    for sub in subsets:
        aux = X[list(sub)]
        val = errorFunc(model,aux,y)
        vals.append(val)
    newInfo = pd.DataFrame.from_dict({"Variables":subsets,errorFunc.__name__:vals,"Change":[base - err for err in vals]})
    return newInfo

def exhaustiveSearch(model,X,y,errorFunc,kmin,kmax,verbose=False):
    infos = []
    for i in range(kmin,kmax+1):
        if verbose:
            print("Trying models with {0}".format(i))
        infos.append(subsetComb(model,X,y,errorFunc,k=i))
    return pd.concat(infos,ignore_index=True)

def upwardsSearch(model,X,y,errorFunc,kmin,kmax,verbose=False):
    if verbose:
        print("Selecting initial model")
    info = subsetComb(model,X,y,errorFunc,k=kmin)
    initVars = info["Variables"][info["Change"].idxmax()]
    if verbose:
        print("Selected initial model with {0}".format(initVars))
    X1 = X[initVars].copy()
    c = 0
    while X1.shape[1] < kmax and c<1000:
        if verbose:
            print("Selecting model with {0} variables".format(X1.shape[1]+1))
        baseErr = errorFunc(model,X1,y)
        otherVars = [var for var in X.columns if not(var in X1.columns)] 
        errors = []
        for var in otherVars:
            aux = X1.copy()
            aux[var] = X[var]
            errors.append(errorFunc(model,aux,y))
        bestVar =  otherVars[np.argmin(errors)]
        if verbose:
            print("Adding {0} to model".format(bestVar))
            if baseErr < np.min(errors):
                print("Warning: adding {0} to {1} doesn't decrease {2}".format(bestVar,X1.columns.to_list(),errorFunc.__name__))
        X1[bestVar] = X[bestVar]
        c += 1
    return X1

def backwardsSearch(model,X,y,errorFunc,kmin,kmax,verbose=False):
    if verbose:
        print("Selecting initial model")
    info = subsetComb(model,X,y,errorFunc,k=kmax)
    initVars = info["Variables"][info["Change"].idxmax()]
    if verbose:
        print("Selected initial model with {0}".format(initVars))
    X1 = X[initVars].copy()
    c = 0
    while X1.shape[1] > kmin and c<1000:
        if verbose:
            print("Selecting model with {0} variables".format(X1.shape[1]-1))
        baseErr = errorFunc(model,X1,y)
        errors = [errorFunc(model,X1.drop(var,axis=1),y) for var in X1.columns]
        bestVar =  X1.columns[np.argmin(errors)]
        if verbose:
            print("Removing {0} from model".format(bestVar))
            if baseErr < np.min(errors):
                print("Warning: removing {0} from {1} doesn't decrease {2}".format(bestVar,X1.columns.to_list(),errorFunc.__name__))
        X1 = X1.drop(bestVar,axis=1)
        c += 1
    return X1


