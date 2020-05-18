#Bünyamin Şentürk - 150116007
#Kemal Barbaros - 150116070
import inline as inline
import matplotlib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from sklearn.naive_bayes import GaussianNB
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

x = data[['quarter', 'stock', 'date', 'open', 'high', 'low', 'close', 'volume', 'percent_change_price',
          'percent_change_volume_over_last_wk', 'previous_weeks_volume', 'next_weeks_open', 'next_weeks_close',
          'percent_change_next_weeks_price', 'days_to_next_dividend', 'percent_return_next_dividend']]      #features

#print(data.isna().sum())  # check if there are missing values

data = data.drop(['date'], axis=1)      #we don't need date
