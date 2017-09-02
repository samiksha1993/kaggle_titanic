#kaggle -titan
import pandas as pd
import numpy as np
from math import *
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn import svm, tree, linear_model
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import random_projection
from extra_funct import child, senior_citizen, female_married
import matplotlib.pyplot as plt
#training
train_in = pd.read_csv('train.csv',header=0)
train_in['Sex'] = train_in['Sex'].map({'female':0,'male':1}).astype(int)
pattern = ['Miss.','Master']
train_in['sur_name'] = train_in['Name'].str.contains('Master|Miss',na=False)
train_in['sur_name'] = train_in['sur_name'].map({True:1,False:0})
train_in['C'] = train_in['Embarked']
train_in['Q'] = train_in['Embarked']
train_in['S'] = train_in['Embarked']
train_in['Child'] = train_in.apply(child, axis=1)
train_in['senior_citizen'] = train_in.apply(senior_citizen, axis=1)
train_in['post'] = train_in['Name'].str.split(',',1).str[1].str.split('.',1).str[0].str.strip()
['Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms',
       'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'the Countess',
       'Jonkheer']
train_in['post_map'] = train_in['post'].map({'Mr':13, 'Mrs':12, 'Miss':11, 'Master':10, 'Don':8, 'Rev':6, 'Dr':7, 'Mme':15, 'Ms':14,'Major':5, 'Lady':3, 'Sir':2, 'Mlle':9, 'Col':13, 'Capt':4, 'the Countess':1,'Jonkheer':16})
train_in.post_map.fillna(value=0, inplace=True)
train_in['female_married'] = train_in.apply(female_married, axis=1)
np.random.seed(42)

if len(train_in.Embarked[train_in.Embarked.isnull()])>0:
	train_in.Embarked[train_in.Embarked.isnull()] = train_in.Embarked.dropna().mode().values

train_in['Embarked'] = train_in['Embarked'].map({'C':100,'Q':010,'S':001}).astype(int)
train_in['C'] = train_in['Embarked'].map({100:1,010:0,001:0})
train_in['Q'] = train_in['Embarked'].map({100:0,010:1,001:0})
train_in['S'] = train_in['Embarked'].map({100:0,010:0,001:1})


train_in['Pclass'] = train_in['Pclass'].map({1:3,2:2,3:1}).astype(int)
train_in['class1'] = train_in['Pclass'].map({100:1,010:0,001:0})
train_in['class2'] = train_in['Pclass'].map({100:0,010:1,001:0})
train_in['class3'] = train_in['Pclass'].map({100:0,010:0,001:1})



# train_in = train_in[train_in.Embarked.notnull()]
# train_in['Embarked'] = train_in['Embarked'].map({'C':0,'Q':1,'S':2}).astype(int)
median_age = train_in['Age'].dropna().mean()
if len(train_in.Age[train_in.Age.isnull()])>0:
	train_in.Age[train_in.Age.isnull()] = median_age
# train_in = train_in[train_in.Age.notnull()]
# train_in['Fare'] = train_in['Fare'].dropna().round()
median_fare = train_in['Fare'].dropna().mean()
if len(train_in.Fare[train_in.Fare.isnull()])>0:
	train_in.Fare[train_in.Fare.isnull()] = median_fare

if len(train_in.Cabin)>0:
	train_in.Cabin[train_in.Cabin.notnull()] = 1
if len(train_in.Cabin[train_in.Cabin.isnull()])>0:
	train_in.Cabin[train_in.Cabin.isnull()] = 0

# train_in['Fare'] = train_in['Fare'].round()
# train_in = train_in[train_in.Fare.notnull()]
X = train_in[['Pclass','Age','Sex','Embarked','Parch','post_map','female_married']]
Y = train_in[['Survived']]


#plot
X[X.dtypes[(X.dtypes=="float64")|(X.dtypes=="int64")]
                        .index.values].hist(figsize=[11,11])




X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# print X_train
train_size = len(X_train) 
test_size = len(X_test)
# print train_size
# print test_size
# print len(X_train)*100/len(X)
# print len(X_test)*100/len(X)


#K-fold



#random forest
# clf= RandomForestClassifier(n_estimators=50)
# clf.fit(X_train, y_train)
# #SVC with linear kernel
# clf = svm.LinearSVC(random_state=42)
# clf.fit(X_train, y_train)
# predicted = clf.predict(X_test)
# print "LinearSVC: {}".format(100*accuracy_score(y_test, predicted))

# predicted = clf.predict(X_test)
# print clf
# print "RandomForest: {}".format(100*accuracy_score(y_test, predicted))

#Test set 
# test_in = pd.read_csv('test.csv',header=0)
# test_in['Sex'] = test_in['Sex'].map({'female':0,'male':1}).astype(int)
# test_in['Child'] = train_in.apply(child, axis=1)
# test_in['senior_citizen'] = train_in.apply(senior_citizen, axis=1)
# test_in['Pclass'] = test_in['Pclass'].map({1:3,2:2,3:1}).astype(int)

# if len(test_in.Embarked[test_in.Embarked.isnull()])>0:
# 	test_in.Embarked[test_in.Embarked.isnull()] = test_in.Embarked.dropna().mode().values

# test_in['Embarked'] = test_in['Embarked'].map({'C':100,'Q':010,'S':001}).astype(int)
# test_in['C'] = test_in['Embarked'].map({100:1,010:0,001:0})
# test_in['Q'] = test_in['Embarked'].map({100:0,010:1,001:0})
# test_in['S'] = test_in['Embarked'].map({100:0,010:0,001:1})


# median_age = test_in['Age'].dropna().mean()
# if len(test_in.Age[test_in.Age.isnull()])>0:
# 	test_in.Age[test_in.Age.isnull()] = median_age

# median_fare = test_in['Fare'].dropna().median()
# if len(test_in.Fare[test_in.Fare.isnull()])>0:
# 	test_in.Fare[test_in.Fare.isnull()] = median_fare

# X = test_in[['Pclass','Age','Sex','Fare','C','Q','S','Child','senior_citizen']]
# # Y = test_in[['Survived']]
# passenger_id = test_in['PassengerId']
# predicted = clf.predict(X)
# # print predicted
# # print passenger_id
# data = {'PassengerId':passenger_id,'Survived':predicted}
# df = pd.DataFrame(data, columns = ['PassengerId', 'Survived'])
# df.to_csv('result.csv', sep=',')

# #linear regression
# clf = linear_model.LinearRegression()
# clf.fit(X_train, y_train)
# predicted = clf.predict(X_test)
# # print predicted
# predicted =  predicted.round().astype(int)
# print "Linear Regression: {}".format(100*accuracy_score(y_test, predicted))

# # SVC classifiers kernel = rbf
# clf = svm.SVC(random_state=42)
# clf.fit(X_train, y_train)
# predicted = clf.predict(X_test)
# print "SVC(rbf): {}".format(100*accuracy_score(y_test, predicted))

#SVC with linear kernel
# clf = svm.SVC()
# clf.set_params(kernel='linear', random_state=42).fit(X_train, y_train)
# predicted = clf.predict(X_test)
# print "SVC(linear): {}".format(100*accuracy_score(y_test, predicted))

# #SVC with linear kernel
# clf = svm.LinearSVC(random_state=42)
# clf.fit(X_train, y_train)
# predicted = clf.predict(X_test)
# print "LinearSVC: {}".format(100*accuracy_score(y_test, predicted))

# #decision tree
# clf=tree.DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# predicted = clf.predict(X_test)
# print "DecisionTree: {}".format(100*accuracy_score(y_test, predicted))


# #decision tree
# clf=tree.DecisionTreeClassifier(criterion = "gini",max_depth=3, min_samples_leaf=3)
# clf.fit(X_train, y_train)
# predicted = clf.predict(X_test)
# print "DecisionTree(gini): {}".format(100*accuracy_score(y_test, predicted))


# clf= tree.DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)
# clf.fit(X_train, y_train)
# predicted = clf.predict(X_test)
# print "DecisionTree(entropy): {}".format(100*accuracy_score(y_test, predicted))

# #random forest
# clf= RandomForestClassifier()
# clf.fit(X_train, y_train)
# predicted = clf.predict(X_test)
# print clf
# print "RandomForest: {}".format(100*accuracy_score(y_test, predicted))


clf= RandomForestClassifier(n_estimators=50,criterion='gini')
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print "RandomForest(n_estimators=50): {}".format(100*accuracy_score(y_test, predicted))
for feature in zip(X_train.head(), clf.feature_importances_):
    print(feature) 	