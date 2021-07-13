# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('sample.csv')
df.head()

# a = ""
# d = list(range(0,4096))

# for i in range(0,len(d)-1):
#     a = a + str(d[i])+","
#     print(a)



from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)



from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()  
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))



# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier(n_estimators=100)
# rfc.fit(X_train, y_train)  

# rfc_pred = rfc.predict(X_test)
# print(confusion_matrix(y_test,rfc_pred))
# print(classification_report(y_test,rfc_pred))


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
params = {'max_leaf_nodes': list(range(2, 10)), 'min_samples_split': [2, 3, 4]}

grid_search_cv = GridSearchCV(DecisionTreeClassifier(), params, verbose=2, cv=3)

grid_search_cv.fit(X_train, y_train)

grid_search_cv.best_estimator_

"""

"""













