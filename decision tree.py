# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 20:10:33 2017

@author: Sean
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# for seaborn issue:
import warnings
warnings.filterwarnings("ignore")

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)


# Import the Numpy library
import numpy as NP

# Import 'tree' from scikit-learn library
from sklearn import tree
# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == 'male'] = 0
train["Sex"][train["Sex"] == 'female'] = 1

train["Age"] = train["Age"].fillna(train["Age"].median())

# Impute the Embarked variable
train["Embarked"] = train.fillna("S")

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == 'S'] = 0
train["Embarked"][train["Embarked"] == 'C'] = 1
train["Embarked"][train["Embarked"] == 'Q'] = 2


# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))

plt.figure(figsize=(14,12))
foo = sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=0.6, square=True, annot=True)