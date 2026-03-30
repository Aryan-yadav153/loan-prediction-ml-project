import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing

# show all columns properly
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df = pd.read_csv('loan_train.csv')

# remove extra columns if present
df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')

print(df.head())
print(df.shape)

df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])

print(df.head())

print(df['loan_status'].value_counts())

df['dayofweek'] = df['effective_date'].dt.dayofweek
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x > 3) else 0)

print(df.head())

print(df.groupby(['Gender'])['loan_status'].value_counts(normalize=True))

df['Gender'].replace(to_replace=['male', 'female'], value=[0, 1], inplace=True)

print(df.head())

print(df.groupby(['education'])['loan_status'].value_counts(normalize=True))

print(df[['Principal', 'terms', 'age', 'Gender', 'education']].head())

Feature = df[['Principal', 'terms', 'age', 'Gender', 'weekend']]

Feature = pd.concat([Feature, pd.get_dummies(df['education'])], axis=1)

Feature.drop(['Master or Above'], axis=1, inplace=True)

print(Feature.head())

X = Feature
print(X[0:5])

y = df['loan_status'].values
print(y[0:5])

X = preprocessing.StandardScaler().fit(X).transform(X)

print(X[0:5])


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

#KNN
n = 5
neigh = KNeighborsClassifier(n_neighbors=n)
neigh.fit(X_train, y_train)

yhat_knn = neigh.predict(X_test)

print("KNN Accuracy:")
print(metrics.accuracy_score(y_test, yhat_knn))

#Decision Tree
modeltree = DecisionTreeClassifier()
modeltree.fit(X_train, y_train)

predTree = modeltree.predict(X_test)

print("Decision Tree Accuracy:")
print(metrics.accuracy_score(y_test, predTree))

#SVM
clf = svm.SVC()
clf.fit(X_train, y_train)

yhat_svm = clf.predict(X_test)

print("SVM Accuracy:")
print(accuracy_score(y_test, yhat_svm))


from sklearn.linear_model import LogisticRegression

#Logistic Regression
LR = LogisticRegression()
LR.fit(X_train, y_train)

yhat_lr = LR.predict(X_test)

print("Logistic Regression Accuracy:")
print(accuracy_score(y_test, yhat_lr))

#Final Comparison
print(" Accuracy Comparison")
print("KNN:", metrics.accuracy_score(y_test, yhat_knn))
print("Decision Tree:", metrics.accuracy_score(y_test, predTree))
print("SVM:", accuracy_score(y_test, yhat_svm))
print("Logistic Regression:", accuracy_score(y_test, yhat_lr))


DecisionTreeClassifier(random_state=4)
LogisticRegression(max_iter=1000)



import matplotlib.pyplot as plt

models = ['KNN', 'Decision Tree', 'SVM', 'Logistic Regression']
accuracies = [
    metrics.accuracy_score(y_test, yhat_knn),
    metrics.accuracy_score(y_test, predTree),
    accuracy_score(y_test, yhat_svm),
    accuracy_score(y_test, yhat_lr)
]

plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")

plt.show()
