import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

"""
Bünyamin Şentürk - 150116007
Kemal Barbaros - 150116070
"""


data = pd.read_csv("data.csv")              # read data

"""print(data.isna().sum())  # check if there are missing values"""
data = data.interpolate(method='linear', limit_direction='forward')     # fill missing values with linear forwarding
data = data.drop(['date'], axis=1)                                  # we don't need these columns
data = data.drop(['percent_change_price'], axis=1)
data = data.drop(['percent_change_volume_over_last_wk'], axis=1)
data = data.drop(['previous_weeks_volume'], axis=1)

stocknum = LabelEncoder()           # converting 'stock' to numerical value
stocknum.fit(data['stock'])
data['stock'] = stocknum.transform(data['stock'])

data = data.iloc[1:]            # dropping first row because it can not be interpolated

scale = StandardScaler()                                       # scaling data
data_scaled = scale.fit_transform(data)

data_normalized = normalize(data_scaled)                        # normalizing data
data_normalized = pd.DataFrame(data_normalized, columns=data.columns)
data_normalized.head()

# data_normalized.to_csv(r'normalized_data.csv', index=False)     # exporting data to check if it has problems

pca = PCA(n_components=2)                           # reducing it 2 dimensions for better visuality
data_2d = pca.fit_transform(data_normalized)
data_2d = pd.DataFrame(data_2d)
data_2d.columns = ['x', 'y']

# data_2d.to_csv(r'2d.csv', index=False)        check export


def agnes():
    timer1 = time.perf_counter()        # timer begins

    agnes = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')     # clustering
    agnes.fit_predict(data_2d)
    plt.figure(figsize=(10, 7))                                 # graphic output
    plt.scatter(data_2d['x'], data_2d['y'], c=agnes.labels_)
    plt.xlabel('')
    plt.ylabel('Percent Change of Price')

    timer2 = time.perf_counter()        # timer ends
    print(f"\nAGNES done in {timer2 - timer1:0.4f} seconds")
    plt.show()


def k_means():
    timer1 = time.perf_counter()        # timer begins

    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)    # clustering
    kmeans.fit_predict(data_2d)

    plt.figure(figsize=(10, 7))         # graphic output
    plt.scatter(data_2d['x'], data_2d['y'], c=kmeans.labels_)
    plt.xlabel('')
    plt.ylabel('Percent Change of Price')

    timer2 = time.perf_counter()        # timer ends
    print(f"\nK Means done in {timer2 - timer1:0.4f} seconds")

    plt.show()


def dbscan():
    timer1 = time.perf_counter()        # timer begins

    dbs = DBSCAN(eps=0.1, min_samples=5)        # clustering
    dbs.fit(data_2d)

    plt.figure(figsize=(10, 7))         # graphic output
    plt.scatter(data_2d['x'], data_2d['y'], c=dbs.labels_)
    plt.xlabel('')
    plt.ylabel('Percent Change of Price')

    timer2 = time.perf_counter()        # timer ends
    print(f"\nDBSCAN done in {timer2 - timer1:0.4f} seconds")

    plt.show()


def dendogram():
    plt.figure(figsize=(10, 7))
    plt.title("Dendrograms")
    dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
    plt.axhline(y=3, color='g', linestyle='--')
    plt.show()


def elbow():
    wcss = []
    for i in range(1,10):       # try clusters from 1 to 10
        kmeans=KMeans(n_clusters= i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1,10),wcss)
    plt.title('Elbow Method Graphic')
    plt.xlabel('Clusters')
    plt.ylabel('wcss')
    plt.show()


"""
elbow() #ELBOW METHOD FUNCTION, TAKES TOO MUCH TIME TO RUN, ONLY RAN FOR ONCE
dendogram() #TO DECIDE CLUSTER NUMBER FOR AGNES, ONLY RAN FOR ONCE
"""

agnes()

k_means()

dbscan()
