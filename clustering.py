from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Auxiliar function taken from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
# for plotting dendogram of agglomerative clustering
def plotDendogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


"""
Example:

fig = plt.figure(figsize=(12,6))
plotClusteringResults(fit,data["original"],s=80)
plt.tight_layout()
plt.savefig(os.path.join("tarea","3-initClus.pdf"))
plt.show()
"""
def plotClusteringResults(model,df,s=20):
    #labels = (model.labels_-model.labels_[0]) % nclus
    labels = model.labels_
    clusColors = ["C{0}".format(i) for i in labels]
    normColors = ["C{0}".format(i) for i in df["initG"]]

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],s=s,c=clusColors)
    ax1.view_init(30, 45)
    ax1.set_title("Clustering")

    ax2.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],s=s,c=normColors)
    ax2.view_init(30, 45)
    ax2.set_title("Normal Groups")