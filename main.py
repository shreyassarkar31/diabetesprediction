#KNN Predict whether a person will have diabetes or not


import pandas as p
import numpy as n

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

dataset = p.read_csv('KNN_Dataset.csv')

zero_not_accepted_= ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI','Insulin']

for column in zero_not_accepted_:
    dataset[column] = dataset[column].replace(0,n.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(n.NaN, mean)

X = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier  = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
