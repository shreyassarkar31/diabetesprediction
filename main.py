#KNN Predict whether a person will have diabetes or not


import pandas as p
import numpy as n

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

dataset = p.read_csv('KNN Dataset.csv')

print(len(dataset))