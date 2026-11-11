
# pip install micromlgen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from micromlgen import port
from micromlgen import platforms

print("Libraries updated")
seed = 7
classmap = {1: 'Ragi', 2: 'Rice', 3: 'Maize'}


print(platforms.ALLOWED_CLASSIFIERS)


df = pd.read_csv('Crop Prediction Soil.csv')
print(df.shape)
print(list(df.columns))
print(df['CropLabel'].value_counts(), '\n')
print( df.head(5), '\n' )
print( df.info(), '\n' )
print( df.describe(), '\n' )
print(df.isnull().sum(), '\n')
df.drop(["State","District","Soil"],axis=1, errors="ignore", inplace=True)


##sns.heatmap(df, cmap="Oranges", annot=True)
##plt.savefig('HeatMap')
##plt.show()


X = df.iloc[:,0:4]
print("X = ",'\n', X)
y = df.iloc[:,-1]
print('\n', y)


train_X,test_X,train_y,test_y = train_test_split(
    X,y,test_size=0.25, random_state=seed )

print('\n test_X = \n', len(test_X))
print('\n test_y = \n', len(test_y))


# Logistic Regression
LR = LogisticRegression()
classifier = LR.fit(train_X,train_y)
pred = LR.predict(test_X)
print("LR Accuracy_score:", accuracy_score(test_y,pred)*100)
print(pred[0:5], '\n')


# SVM
SVC = SVC(kernel = 'linear')
classifier = SVC.fit(train_X,train_y)
pred = SVC.predict(test_X)
print("SVC Accuracy_score:", accuracy_score(test_y,pred)*100)
print(pred[0:5], '\n')


# Random Forest Classifier
RF = RandomForestClassifier(20, max_depth=10)
classifier = RF.fit(train_X,train_y)
pred = RF.predict(test_X)
print("RF Accuracy_score:", accuracy_score(test_y,pred)*100)
print(pred[0:5], '\n')


# Gaussian Naive Bayes
GaussianNB_Model = GaussianNB()
classifier = GaussianNB_Model.fit(train_X,train_y)
pred = GaussianNB_Model.predict(test_X)
print("Gaussian Accuracy_score:", accuracy_score(test_y,pred)*100)
print(pred[0:5], '\n')


c_code = port(classifier, classmap=classmap)
print(c_code)

filename = "AgriGaussianNBmodel.h"
with open(filename, 'w') as f:
    f.write(c_code)



cm = confusion_matrix(test_y, pred)
print(cm,'\n')


print("\nClassification Report :\n")
print(classification_report(test_y, pred))

print("Project End")


