import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn.decomposition import PCA

def correlationPlots(df,saveName="",cmap=cm.viridis_r,f=16,width=50,**kwargs):
    variables = [var for var in df.columns if np.issubdtype(df[var].dtype,np.number)]
    nvar = len(variables)
    values = [df[var].to_numpy() for var in variables]
    correlations = np.corrcoef(values)
    norm = mcolors.Normalize(vmin=-1,vmax=1)
    fig , ax = plt.subplots(ncols=nvar,nrows=nvar,figsize=(19,16),constrained_layout=True)
    for i in range(nvar):
        for j in range(nvar):
            if i!=j:
                ax[i][j].scatter(values[i],values[j],color=cmap(norm(correlations[i][j])),**kwargs)
                ax[i][j].grid()
        ax[0][i].set_title(str(variables[i]),fontsize=f)
        ax[i][nvar-1].yaxis.set_label_position("right")
        ax[i][nvar-1].set_ylabel(str(variables[i]),fontsize=f)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax.ravel().tolist(),aspect=width)
    cbar.ax.tick_params(labelsize=1.5*f)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.set_ylabel("Correlation",fontsize=1.5*f)
    if bool(saveName):
        plt.savefig(saveName)
    plt.show()

def classPlots(X,y,saveName="",cmap=cm.Dark2,f=16,width=50,**kwargs):
    variables = [var for var in X.columns if np.issubdtype(X[var].dtype,np.number)]
    nvar = len(variables)
    values = [X[var].to_numpy() for var in variables]
    classes = list(set(y))
    nclass = len(classes)
    norm = mcolors.Normalize(vmin=0,vmax=nclass-1)     
    colors = [cmap(norm(classes.index(clas))) for clas in y]
    fig , ax = plt.subplots(ncols=nvar,nrows=nvar,figsize=(19,16),constrained_layout=True)
    for i in range(nvar):
        for j in range(nvar):
            if i!=j:
                ax[i][j].scatter(values[i],values[j],color=colors,**kwargs)
                ax[i][j].grid()
        ax[0][i].set_title(str(variables[i]),fontsize=f)
        ax[i][nvar-1].yaxis.set_label_position("right")
        ax[i][nvar-1].set_ylabel(str(variables[i]),fontsize=f)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax.ravel().tolist(),aspect=width,ticks=range(nclass))
    cbar.ax.tick_params(labelsize=1.5*f)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.set_ylabel("Class",fontsize=1.5*f)
    cbar.ax.set_yticklabels(classes)
    if bool(saveName):
        plt.savefig(saveName)
    plt.show()

def pcaPlots(X,y,saveName="",cmap=cm.Dark2,f=16,width=50,ncomps=0,**kwargs):
    variables = [var for var in X.columns if np.issubdtype(X[var].dtype,np.number)]
    nvar = len(variables)
    values = [X[var].to_numpy() for var in variables]
    aux = np.transpose(values)
    pca = de.PCA()
    pca.fit(aux)
    Z = pca.transform(aux)
    variables = ["PC{0}".format(i) for i in range(len(variables))]
    values = np.transpose(Z)
    classes = list(set(y))
    nclass = len(classes)
    norm = mcolors.Normalize(vmin=0,vmax=nclass-1)     
    colors = [cmap(norm(classes.index(clas))) for clas in y]
    if ncomps==0:
        ncomps=nvar
    fig , ax = plt.subplots(ncols=ncomps,nrows=ncomps,figsize=(19,16),constrained_layout=True)
    for i in range(ncomps):
        for j in range(ncomps):
            if i!=j:
                ax[i][j].scatter(values[i],values[j],color=colors,**kwargs)
                ax[i][j].grid()
        ax[0][i].set_title(str(variables[i]),fontsize=f)
        ax[i][nvar-1].yaxis.set_label_position("right")
        ax[i][nvar-1].set_ylabel(str(variables[i]),fontsize=f)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax.ravel().tolist(),aspect=width,ticks=range(nclass))
    cbar.ax.tick_params(labelsize=1.5*f)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.set_ylabel("Class",fontsize=1.5*f)
    cbar.ax.set_yticklabels(classes)
    if bool(saveName):
        plt.savefig(saveName)
    plt.show()

    
def indVarDistPlots(df,resCol,saveName="",cmap=cm.Dark2,**kwargs):
    if not(resCol in df.columns):
        raise ValueError("Name {0} is not in columns".format(resCol))
    classes = list(set(df[resCol]))
    nclass = len(classes)
    norm = mcolors.Normalize(vmin=0,vmax=nclass-1)
    colors = [cmap(norm(i)) for i in range(nclass)]
    for col in df.columns:
        if col != resCol and np.issubdtype(df[col].dtype,np.number):
            fig, ax = plt.subplots(ncols=2,nrows=1,figsize=(6,2.5))
            df.boxplot(column=col,by=resCol,ax=ax[0])
            for i,val in enumerate(classes):
                df[df[resCol]==val][col].hist(ax=ax[1],label=str(val),color=colors[i],**kwargs)
            ax[0].set_title("Boxplot")
            ax[1].legend()
            ax[1].set_title("Histogram")
            fig.suptitle("{0} distribution".format(col),y=1.1)
        elif col != resCol and (df[col].dtype == np.dtype('O')):
            fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(4,2.5))
            labels = [str(val) for val in set(df[resCol])]
            weights = [[1/df[df[resCol]==val][col].shape[0] for i in range(df[df[resCol]==val][col].shape[0])] for val in set(df[resCol])]
            weights = np.transpose(weights)
            ax.hist(np.transpose([df[df[resCol]==val][col].to_numpy() for val in set(df[resCol])]),
                    weights=weights,
                    label=labels,
                    color=colors,
                    **kwargs)
            ax.legend()
            ax.grid()
            ax.set_ylabel("Normed counts")
            ax.set_title("{0} histogram".format(col))
        elif col != resCol:
            print("Column {0} cannot be interpreted".format(col))
        if bool(saveName):
            plt.tight_layout()
            plt.savefig("{0}_{1}_dist.pdf".format(saveName,col))
        plt.show()
