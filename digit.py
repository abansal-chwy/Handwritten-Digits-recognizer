import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import  cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
"""
myData=np.genfromtxt('train.csv', delimiter="," ,skip_header=1);
myData_test=np.genfromtxt('test.csv', delimiter="," ,skip_header=1);

X=myData[:,1:785]

Y = myData[:,0]

X_test=myData_test[:,:784]
Y_Pred=[]
#kf = KFold(n_splits=5, shuffle=False)
classifier = [DecisionTreeClassifier(criterion="gini", max_leaf_nodes=20)]
name= ["K_Nearest_Neigh"]
for name, clf in zip(name, classifier):
     # for train_index, test_index in kf.split(myData):
            #x_train, x_test = X[train_index], X[test_index]
            #y_train, y_test = Y[train_index], Y[test_index]

      clf.fit(X, Y)
        # Predict
      Y_pred = clf.predict(X_test)
      Y_Pred.append(Y_pred)

      print (Y_pred)
import pandas
df = pandas.DataFrame(data={"col1": Y_pred})
df.to_csv("results.csv", sep=',',index=False)"""
import pandas
data = pd.read_csv("train.csv").as_matrix()
myData_test = pd.read_csv("test.csv").as_matrix()
X=data[:,1:785]

y = data[:,0]
myList = list(range(1,50))

cv_scores = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#training dataset
xtrain=data[0:21000,1:]
ytrain=data[0:21000,0]
clf= KNeighborsClassifier(weights="distance")
clf.fit(X_train,y_train)

#testing

xtest=data[21000:,1:]
ytest=data[21000:,0]
X_test=myData_test[:,:784]
#d=xtest[40]
##d.shape=(28,28)
#plt.imshow(255-d,cmap='gray')
#print(clf.predict([xtest[40]]    ))
#plt.show()

pred=clf.predict(X_test)
df = pandas.DataFrame(data={"col1": pred})
df.to_csv("results.csv", sep=',',index=False)
#print (accuracy_score(y_test, pred))
#count=0
#for i in range(0,21000):
 #     if p[i]==ytest[i]:
  #          count+=1
#print(count/21000)