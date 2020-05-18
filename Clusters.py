#Bünyamin Şentürk - 150116007
#Kemal Barbaros - 150116070
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


data = pd.read_csv("data.csv")              #read data
data = shuffle(data)                        #shuffle data
data.reset_index(inplace=True, drop=True)   #reset indexes to make shuffle work

pd.DataFrame(data).fillna(data.mean)
#print(data.isna().sum())  # check if there are missing values

data = data.drop(['date'], axis=1)      #we don't need date

labelEncoder = LabelEncoder()           # converting stock to numerical value
labelEncoder.fit(data['stock'])
data['stock'] = labelEncoder.transform(data['stock'])

X = np.array(data.drop(['percent_change_next_weeks_price'], 1).astype(float))
y = np.array(data['percent_change_next_weeks_price'])


kmeans = KMeans(n_clusters=2)
kmeans.fit(X)