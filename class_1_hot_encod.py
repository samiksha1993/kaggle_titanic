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
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
#training
enc=OneHotEncoder(sparse=False)
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
train_in['post_map'] = train_in['post'].map({'Mr':1, 'Mrs':2, 'Miss':3, 'Master':4, 'Don':5, 'Rev':6, 'Dr':7, 'Mme':8, 'Ms':8,'Major':9, 'Lady':10, 'Sir':11, 'Mlle':12, 'Col':13, 'Capt':14, 'the Countess':15,'Jonkheer':16})
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
X = train_in[['Pclass','Sex','Parch','SibSp','Embarked','Age','Fare','female_married']]
Y = train_in[['Survived']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


X_train_1 = X_train
X_test_1 = X_test
columns = ['Pclass','Embarked','Parch']
for col in columns:
   # creating an exhaustive list of all possible categorical values
   data=X_train[[col]].append(X_test[[col]])
   enc.fit(data)
   # Fitting One Hot Encoding on train data
   temp = enc.transform(X_train[[col]])
   # Changing the encoded features into a data frame with new column names
   temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
        .value_counts().index])
   # In side by side concatenation index values should be same
   # Setting the index values similar to the X_train data frame
   temp=temp.set_index(X_train.index.values)
   # adding the new One Hot Encoded varibales to the train data frame
   X_train_1=pd.concat([X_train_1,temp],axis=1)
   # fitting One Hot Encoding on test data
   temp = enc.transform(X_test[[col]])
   # changing it into data frame and adding column names
   temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
        .value_counts().index])
   # Setting the index for proper concatenation
   temp=temp.set_index(X_test_1.index.values)
   # adding the new One Hot Encoded varibales to test data frame
   X_test_1=pd.concat([X_test_1,temp],axis=1)


X_test_1['Age'] = X_test['Age']
X_train_1['Age'] = X_train['Age']
X_test_1['Fare'] = X_test['Fare']
X_train_1['Fare'] = X_train['Fare']
[X_train_1.__delitem__(x) for x in ['Pclass','Embarked','SibSp','Parch']]
[X_test_1.__delitem__(x) for x in ['Pclass','Embarked','SibSp','Parch']]
clf= RandomForestClassifier(n_estimators=200,criterion='gini')
clf.fit(X_train_1, y_train)
predicted = clf.predict(X_test_1)
print "RandomForest(n_estimators=50): {}".format(100*accuracy_score(y_test, predicted))
for feature in zip(X_train_1.head(), clf.feature_importances_):
    print(feature)