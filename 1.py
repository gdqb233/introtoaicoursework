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
