import numpy as np
import pandas as pd
import statsmodels.api as sm

def oneVsRestLogisticRegrresionClassifier(X,y,verbo=False,**kwargs):
    classes = list(set(y))
    nclasses = len(classes)
    classifiers = []
    if verbo:
        print("Classes {0}".format(classes))
    for index,cla in enumerate(classes):
        if verbo:
            print("Classifiyng class {0}".format(cla))
        newY = (y==cla).astype("float64")
        mod = sm.Logit(exog=X,endog=newY)
        fit = mod.fit()
        classifiers.append(fit)
    return classes,classifiers

def multinomialLogisticRegressionClassifier(X,y,reference,**kwargs):
    classes = list(set(y))
    nclasses = len(classes)
    classifiers = []
    newY = y.astype("category").cat.codes
    refCode = newY[y==reference].iloc[0]
    newY = (newY - refCode) % nclasses
    newY.name = y.name
    conversion = {}
    for c in classes:
        code = newY[y==c].iloc[0]
        conversion.update({c:code})
    mod = sm.MNLogit(exog=X,endog=newY)
    fit = mod.fit(maxiters=10**5)
    return conversion,fit

def drop1Clasifier(model,df,resCol,errorFunc,verbose=True):
    X = df.drop(resCol,axis=1)
    y = df[resCol]
    currFit = model.fit(X,y)
    baseError = errorFunc(model,df,resCol)[0][0]
    errors = []
    for var in X.columns:
        aux = df.drop(var,axis=1)
        err = errorFunc(model,aux,resCol)
        # Use global error as function
        errors.append(err[0][0])
    newInfo = pd.DataFrame.from_dict({"Dropped":X.columns,"Error rate":errors,"Error rate change":[baseError-e for e in errors]})
    return newInfo

def dropRecursivelyClasifier(model,df,resCol,errorFunc,maxiter=10,verbose=True):
    newdf = df.copy()
    newInfo = drop1Clasifier(model,newdf,resCol,errorFunc,verbose=verbose)
    if verbose:
        print(newInfo)
    c = 0
    while newInfo["Error rate change"].max()>0 and c < maxiter:
        droped = newInfo["Dropped"][newInfo["Error rate change"].idxmax()]
        newdf = newdf.drop(droped,axis=1)
        newInfo = drop1Clasifier(model,newdf,resCol,errorFunc,verbose=verbose)
        if verbose:
            print(newInfo)
        c += 1
    return newdf

def add1Clasifier(model,df,resCol,errorFunc,verbose=True):
    X = df.drop(resCol,axis=1)
    y = df[resCol]
    currFit = model.fit(X,y)
    baseError = errorFunc(model,df,resCol)[0][0]
    errors = []
    interactions = []
    for index1,col1 in enumerate(X.columns):
        if col1 != "const":
            for index2,col2 in enumerate(X.columns[index1:]):
                if col2 != "const":
                    interactions.append("{0}:{1}".format(col1,col2))
    for inter in interactions:
        aux = df.copy()
        col1,col2 = inter.split(":")
        aux[inter] = aux[col1]*aux[col2]
        err = errorFunc(model,aux,resCol)
        # Use global error as function
        errors.append(err[0][0])
    newInfo = pd.DataFrame.from_dict({"Added":interactions,"Error rate":errors,"Error rate change":[baseError-e for e in errors]})
    return newInfo

def addRecursivelyClasifier(model,df,resCol,errorFunc,maxiter=10,verbose=True):
    newdf = df.copy()
    newInfo = add1Clasifier(model,newdf,resCol,errorFunc,verbose=verbose)
    if verbose:
        print(newInfo)
    c = 0
    while newInfo["Error rate change"].max()>0 and c < maxiter:
        added = newInfo["Added"][newInfo["Error rate change"].idxmax()]
        spl = added.split(":")
        col1,col2 = spl 
        newdf["*".join(spl)] = newdf[col1]*newdf[col2]
        newInfo = add1(model,newdf,resCol,errorFunc,verbose=verbose)
        if verbose:
            print(newInfo)
        c += 1
    return newdf

def stepErrorRateClasifier(model,df,resCol,errorFunc,maxiter=10,verbose=True):
    newdf = df.copy()
    dropInfo = drop1Clasifier(model,newdf,resCol,errorFunc,verbose=verbose)
    addInfo = add1Clasifier(model,newdf,resCol,errorFunc,verbose=verbose)
    if verbose:
        print(dropInfo)
        print(addInfo)
    max1 = dropInfo["Error rate change"].max()
    max2 = addInfo["Error rate change"].max()
    totalmax = max(max1,max2)
    c = 0
    while totalmax > 0 and c < maxiter:
        if totalmax == max1:
            droped = dropInfo["Dropped"][dropInfo["Error rate change"].idxmax()]
            newdf = newdf.drop(droped,axis=1)
        else:
            added = addInfo["Added"][addInfo["Error rate change"].idxmax()]
            spl = added.split(":")
            col1,col2 = spl 
            newdf["*".join(spl)] = newdf[col1]*newdf[col2]
        dropInfo = drop1Clasifier(model,newdf,resCol,errorFunc,verbose=verbose)
        addInfo = add1Clasifier(model,newdf,resCol,errorFunc,verbose=verbose)
        if verbose:
            print(dropInfo)
            print(addInfo)
        max1 = dropInfo["Error rate change"].max()
        max2 = addInfo["Error rate change"].max()
        totalmax = max(max1,max2)
        c += 1
    return newdf