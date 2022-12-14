import sklearn
from sklearn.linear_model import LogisticRegression, LinearRegression
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



datasetpath = 'pulmonary_adenocarcinoma_dataset.csv'
dataset = pd.read_csv(datasetpath)


X = np.vstack((dataset['lobular_speciticular'], dataset['air_bronchus_sign'], dataset['vessel_passthrough'], dataset['long_diameter'], dataset['avarage_CT_value'])).T
y = np.array(dataset['pathology'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  



logisticregressionclassifier = LogisticRegression()
logisticregressionclassifier.fit(X_train, y_train)

'''
#svm
svcmodel = SVC(kernel='linear', C=1E10)
svcmodel.fit(X_train, y_train)
y_pred = svcmodel.predict(X_train)
'''

'''
#decisiontree
decisiontreemodel = DecisionTreeClassifier(criterion = 'entropy')
decisiontreemodel.fit(X_train,y_train)
y_pred = decisiontreemodel.predict(X_test)
'''

'''
#knn
knnmodel = KNeighborsClassifier()
knnmodel.fit(X_train, y_train) 
'''

'''
#randomforestmodel
randomforestmodel = RandomForestClassifier(n_estimators=10,criterion="entropy")
randomforestmodel.fit(X_train, y_train)
'''

'''
#bayesianmodel

bayesianmodel = GaussianNB()
model.fit(X, y)
'''

y_pred = logisticregressionclassifier.predict(X_test)
print(accuracy_score(y_test, y_pred))


'''
# neural network

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense#, Activation

import numpy as np

def onehot(X):                                 #Onehot
    T = np.zeros((X.size, 4))
    for idx, row in enumerate(T):
        row[X[idx]-1] = 1
    return T

y = onehot(y)

model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

kf = KFold(5)

overallaccuracy = 0

for train, test in kf.split(X):
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]

    model.fit(X_train,y_train,verbose=2,epochs=1000)
    pred = model.predict(X_test)
    pred = np.argmax(pred,axis=1)
    y_compare = np.argmax(y_test,axis=1) 
    score = metrics.accuracy_score(y_compare, pred)
    overallaccuracy = overallaccuracy + score
    print("Accuracy score: {}".format(score))
print ('overall accuracy:', overallaccuracy/5)

'''


