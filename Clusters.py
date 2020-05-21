"""Bünyamin Şentürk - 150116007"""
"""Kemal Barbaros - 150116070"""

import inline as inline
import matplotlib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
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


data = pd.read_csv("data.csv")              #read data
X = data.iloc[:, [8]]

"""print(data.isna().sum())  # check if there are missing values"""
data = data.interpolate(method='linear', limit_direction='forward')
data = data.drop(['date'], axis=1)      #we don't need date

labelEncoder = LabelEncoder()           # converting 'stock' to numerical value
labelEncoder.fit(data['stock'])
data['stock'] = labelEncoder.transform(data['stock'])

data = data.iloc[1:]            #dropping first row because it can not be interpolated

scaler = StandardScaler()                                       #scaling data
data_scaled = scaler.fit_transform(data)

data_normalized = normalize(data_scaled)                        #normalizing data
data_normalized = pd.DataFrame(data_normalized, columns=data.columns)
data_normalized.head()

data_normalized.to_csv(r'normalized_data.csv', index=False)     #exporting data to check if it has problems


def agnes():
    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    cluster.fit_predict(data_scaled)
    plt.figure(figsize=(10, 7))
    plt.scatter(data_normalized['previous_weeks_volume'], data_normalized['percent_change_next_weeks_price'], c=cluster.labels_)
    plt.show()


def dendogram():
    plt.figure(figsize=(10, 7))
    plt.title("Dendrograms")
    dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
    plt.axhline(y=3, color='g', linestyle='--')
    plt.show()


def elbow():
    wcss = []
    for i in range(1,11):
        kmeans=KMeans(n_clusters= i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1,11),wcss)
    plt.title('Elbow Method Graphic')
    plt.xlabel('Clusters')
    plt.ylabel('wcss')
    plt.show()


def kMeans():
    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit_predict(data)

    """plt.scatter(X[kmeans == 0, 0], X[kmeans == 0, 1], s=100, c='red', label='Careful')
    plt.scatter(X[kmeans == 1, 0], X[kmeans == 1, 1], s=100, c='blue', label='Standard')
    plt.scatter(X[kmeans == 2, 0], X[kmeans == 2, 1], s=100, c='green', label='Target')
    plt.scatter(X[kmeans == 3, 0], X[kmeans == 3, 1], s=100, c='cyan', label='Careless')
    # apply centroid for all cluster
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.title('Cluster of Client')
    plt.xlabel('Annual income (k$)')
    plt.ylabel('Spending Score(1-100)')
    plt.legend()
    plt.show()"""

#elbow() #ELBOW METHOD FUNCTION, TAKES TOO MUCH TIME TO RUN, ONLY RAN FOR ONCE
#dendogram() #TO DECIDE CLUSTER NUMBER FOR AGNES, ONLY RAN FOR ONCE

agnes()
timer1 = time.perf_counter()                     #timer starts
#kMeans()
timer2 = time.perf_counter()                     #timer ends
print(f"\nK Means done in {timer2 - timer1:0.4f} seconds")