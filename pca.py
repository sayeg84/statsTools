
import numpy as np
import pandas as pd
import os



# function to export the PCA coefficient table
def compCoefTable(model,data):
    p = len(data.columns)
    c = int(np.floor(np.log10(p)+1))
    names = ["PC"+"{0}".format(i).zfill(c) for i in range(len(data.columns))]
    return pd.DataFrame(model.components_, index = names,columns = data.columns)

# function to export the singular value table
def singvalTable(model,data):
    p = len(data.columns)
    c = int(np.floor(np.log10(p)+1))
    names = ["PC"+"{0}".format(i).zfill(c) for i in range(len(data.columns))]
    return pd.DataFrame(model.singular_values_,index =names).transpose()

# function to create the PVE table using the scikit-learn package
def pveTable(model,data):
    p = len(data.columns)
    c = int(np.floor(np.log10(p)+1))
    names = ["PC"+"{0}".format(i).zfill(c) for i in range(len(data.columns))]
    return pd.DataFrame(model.explained_variance_ratio_,index=names,columns = ["PVE"]).transpose()

#function to calculate the PVE of the m-th principal component
def pve(m,model,data):
    n,p = df.shape
    norm = 0
    s = 0
    for i in range(n):
        temp = 0
        for j in range(p):
            #print("s = {0}".format(s))
            #print("norm = {0}".format(norm))
            temp = temp + model.components_[m][j]*data.iloc[i][j]
            norm = norm + data.iloc[i][j]**2
        s = s + temp**2
    return s/norm

#function to create the PVE table using the PVE(model,data) function
def pveFromFormulaTable(model,data):
    p = len(data.columns)
    c = int(np.floor(np.log10(p)+1))
    names = ["PC"+"{0}".format(i).zfill(c) for i in range(len(data.columns))]
    vals = [pve(i,model,data) for i in range(p)]
    return pd.DataFrame(vals,index=names,columns = ["PVE"]).transpose()
