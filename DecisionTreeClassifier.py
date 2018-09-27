
#import modules
https://www.codechef.com/getting-startedimport pandas as pd
import numpy as np

#can be highly used for making graphs(SEABORN can also be used.)
import matplotlib.pyplot as plt
%matplotlib inline


#reading the dataset using pandas.
df = pd.read_csv('kyphosis.csv')

#importing train test split 
from sklearn.cross_validation import train_test_split

X = df.drop('Kyphosis',axis=1)

#target: 
y = df['Kyphosis']

 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
 
 #importing DecisionTreeClassifier
 from sklearn.tree import DecisionTreeClassifier
 
 #just using a small name :)
 dtree = DecisionTreeClassifier()
 dtree.fit(X_train,y_train)
 
 
 predictions = dtree.predict(X_test)
 predictions
 
 from sklearn.metrics import confusion_matrix,classification_report

#prints a classififcation report
 print(classification_report(y_test,predictions))
