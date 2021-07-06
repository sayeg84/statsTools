import pandas as pd
from statsmodels.stats import anova_lm

def cleanVarName(var):
    if len(var) > 1:
        if var[0]=="C" and var[1]=="(":
            if len(var.split("[")) > 1:
                return var.split(",")[0][2:] + " [" + var.split("[")[1]
            else:
                return var.split(",")[0][2:] 
        elif len(var.split(":")) > 1:
            fin = ""
            for v in var.split(":"):
                fin += " : " + v 
            return fin
        else:
            return var
    else:
        return var

def significanceTable(fit):
    df = pd.DataFrame(columns = ["R-squared",
                                 "AIC",
                                 "BIC",
                                 "Log-Likelihood",
                                "F-statistic",
                                "Prob (F-statistic)"])
    df.loc[0]=[fit.rsquared,
    fit.aic,
    fit.bic,
    fit.llf,
    fit.fvalue,
    fit.f_pvalue]
    return df

def coefficientTable(fit):
    df = fit.summary2().tables[1]
    df = df.rename(columns={"P>|t|":"$P (> |t|)$"})
    df.index = [cleanVarName(var) for var in df.index]
    return df

def anovaTable(fit):
    df = sm.stats.anova_lm(fit)
    df = df.rename(columns={"PR(>F)":"$P (> F$)"})
    df.index = [cleanVarName(var) for var in df.index]
    df.index.name = "Variable"
    return df 

def comparisonTable(fits):
    dfs = [significanceTable(fit) for fit in fits]
    df = dfs[0]
    for i in range(1,len(dfs)):
        df.loc[i] = dfs[i].iloc[0]
    formulas = [fit.model.formula for fit in fits]
    df.insert(loc=0,column="Model",value=formulas)
    return df

def drop1(model,X,y,verbose=True):
    currFit = model(exog=X,endog=y).fit()
    if verbose:
        print(currFit.summary())
    aics = []
    residuals = []
    for var in X.columns:
        aux = X.drop(var,axis=1)
        fit = model(exog=aux,endog=y).fit()
        aics.append(fit.aic)
        residuals.append(fit.ssr)
    newInfo = pd.DataFrame.from_dict({"Dropped":X.columns,"RSS":residuals,"RSS Change":[currFit.ssr-rss for rss in residuals],"AIC":aics,"AIC Change":[currFit.aic-aic for aic in aics]})
    return newInfo


def dropRecursively(model,X,y,maxiter=10,verbose=True):
    X1 = X.copy()
    newInfo = drop1(model,X1,y,verbose=verbose)
    if verbose:
        print(newInfo)
    c = 0
    while newInfo["AIC Change"].max()>0 and c < maxiter:
        droped = newInfo["Dropped"][newInfo["AIC Change"].idxmax()]
        X1 = X1.drop(droped,axis=1)
        newInfo = drop1(model,X1,y,verbose=verbose)
        if verbose:
            print(newInfo)
        c += 1
    return X1

def add1(model,X,y,verbose=True):
    currFit = model(exog=X,endog=y).fit()
    if verbose:
        print(currFit.summary())
    aics = []
    residuals = []
    newVars = []
    interactions = []
    for index1,col1 in enumerate(X.columns):
        if col1 != "const":
            for index2,col2 in enumerate(X.columns[index1:]):
                if col2 != "const":
                    interactions.append("{0}:{1}".format(col1,col2))
    for inter in interactions:
        aux = X.copy()
        col1,col2 = inter.split(":")
        aux[inter] = aux[col1]*aux[col2]
        fit = model(exog=aux,endog=y).fit()
        aics.append(fit.aic)
        residuals.append(fit.ssr)
    newInfo = pd.DataFrame.from_dict({"Added":interactions,"RSS":residuals,"RSS Change":[currFit.ssr-rss for rss in residuals],"AIC":aics,"AIC Change":[currFit.aic-aic for aic in aics]})
    return newInfo

def addRecursively(model,X,y,maxiter=10,verbose=True):
    X1 = X.copy()
    newInfo = add1(model,X1,y,verbose=verbose)
    if verbose:
        print(newInfo)
    c = 0
    while newInfo["AIC Change"].max()>0 and c < maxiter:
        added = newInfo["Added"][newInfo["AIC Change"].idxmax()]
        spl = added.split(":")
        col1,col2 = spl 
        X1["*".join(spl)] = X1[col1]*X1[col2]
        newInfo = add1(model,X1,y,verbose=verbose)
        if verbose:
            print(newInfo)
        c += 1
    return X1

def stepAIC(model,X,y,maxiter=10,verbose=True):
    X1 = X.copy()
    dropInfo = drop1(model,X1,y,verbose=verbose)
    addInfo = add1(model,X1,y,verbose=verbose)
    if verbose:
        print(dropInfo)
        print(addInfo)
    max1 = dropInfo["AIC Change"].max()
    max2 = addInfo["AIC Change"].max()
    totalmax = max(max1,max2)
    c = 0
    while totalmax > 0 and c < maxiter:
        if totalmax == max1:
            droped = dropInfo["Dropped"][dropInfo["AIC Change"].idxmax()]
            X1 = X1.drop(droped,axis=1)
        else:
            added = addInfo["Added"][addInfo["AIC Change"].idxmax()]
            spl = added.split(":")
            col1,col2 = spl 
            X1["*".join(spl)] = X1[col1]*X1[col2]
        dropInfo = drop1(model,X1,y,verbose=verbose)
        addInfo = add1(model,X1,y,verbose=verbose)
        if verbose:
            print(dropInfo)
            print(addInfo)
        max1 = dropInfo["AIC Change"].max()
        max2 = addInfo["AIC Change"].max()
        totalmax = max(max1,max2)
        c += 1
    return X1