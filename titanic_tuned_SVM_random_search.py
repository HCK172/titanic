# data analysis and wrangling
import pandas as pd
import numpy as np
import scipy

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.svm import SVR, SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# utility
from time import time

training_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


## Set of functions to transform features into more convenient format.
#
# Code performs three separate tasks:
#   (1). Pull out the first letter of the cabin feature.
#          Code taken from: https://www.kaggle.com/jeffd23/titanic/scikit-learn-ml-from-start-to-finish
#   (2). Add column which is binary variable that pertains
#        to whether the cabin feature is known or not.
#        (This may be relevant for Pclass = 1).
#   (3). Recasts cabin feature as number.
def simplify_cabins(data):
    data.Cabin = data.Cabin.fillna('N')
    data.Cabin = data.Cabin.apply(lambda x: x[0])

    cabin_mapping = {'N': 0, 'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1,
                     'F': 1, 'G': 1, 'T': 1}
    data['Cabin_Known'] = data.Cabin.map(cabin_mapping)

    le = preprocessing.LabelEncoder().fit(data.Cabin)
    data.Cabin = le.transform(data.Cabin)

    return data


# Recast sex as numerical feature.
def simplify_sex(data):
    sex_mapping = {'male': 0, 'female': 1}
    data.Sex = data.Sex.map(sex_mapping).astype(int)

    return data


# Recast port of departure as numerical feature.
def simplify_embark(data):
    # Two missing values, assign the most common port of departure.
    data.Embarked = data.Embarked.fillna('S')

    le = preprocessing.LabelEncoder().fit(data.Embarked)
    data.Embarked = le.transform(data.Embarked)

    return data


# Extract title from names, then assign to one of five ordinal classes.
# Function based on code from: https://www.kaggle.com/startupsci/titanic/titanic-data-science-solutions
def add_title(data):
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    data.Title = data.Title.replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                                     'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data.Title = data.Title.replace('Mlle', 'Miss')
    data.Title = data.Title.replace('Ms', 'Miss')
    data.Title = data.Title.replace('Mme', 'Mrs')

    # Map from strings to ordinal variables.
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    data.Title = data.Title.map(title_mapping)
    data.Title = data.Title.fillna(0)

    return data


# Drop all unwanted features (name, ticket).
def drop_features(data):
    return data.drop(['Name', 'Ticket'], axis=1)

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# Perform all feature transformations.
def transform_all(data):
    data = simplify_cabins(data)
    data = simplify_sex(data)
    data = simplify_embark(data)
    data = add_title(data)
    data = drop_features(data)

    return data


training_data = transform_all(training_data)
test_data = transform_all(test_data)
# Impute single missing 'Fare' value with median
training_data['Fare'] = training_data['Fare'].fillna(training_data['Fare'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

# Add Age_Known variable
training_data['Age_Known'] = 1
test_data['Age_Known'] = 1
select_null = pd.isnull(training_data['Age'])
training_data.loc[select_null,'Age_Known'] = 0
select_null = pd.isnull(test_data['Age'])
test_data.loc[select_null,'Age_Known'] = 0


all_data = [training_data, test_data]
combined = pd.concat(all_data, ignore_index=True)


# age imputation
train_not = training_data[pd.notnull(training_data['Age'])]
test_not = test_data[pd.notnull(test_data['Age'])]
train_null = training_data[pd.isnull(training_data['Age'])].drop('Age',axis=1)
test_null = test_data[pd.isnull(test_data['Age'])].drop('Age',axis=1)

# Drop 'Survived' as it is the target variable
droplist = 'Survived'.split()
train_not = train_not.drop(droplist, axis=1)
train_null = train_null.drop(droplist, axis=1)

# define training sets
age_train_x = train_not.drop('Age', axis=1)
age_train_y = train_not['Age']

# SVR
svr_rbf = SVR(kernel='rbf', C=1e2, gamma=0.01)
train_null['Age'] = svr_rbf.fit(age_train_x, age_train_y).predict(train_null).round()
test_null['Age'] = svr_rbf.fit(test_not.drop('Age', axis=1), test_not['Age']).predict(test_null).round()

# replace null values in original data frame
training_data.update(train_null)
test_data.update(test_null)
# print(training_data.columns.values)
# ----------------------------------

# Support Vector Machines
droplist = 'Survived PassengerId Age_Known Cabin_Known'.split()
data = training_data.drop(droplist, axis=1)
# Define features and target values
X, y = data, training_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
# param_dist = {'C': scipy.stats.uniform(0.1, 1000), 'gamma': scipy.stats.uniform(.001, 1.0),
#   'kernel': ['rbf'], 'class_weight':['balanced', None]}
#
# clf = SVC()
#
# # run randomized search
# n_iter_search = 1000
# random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                    n_iter=n_iter_search, n_jobs=-1, cv=4)
#
# start = time()
# random_search.fit(X, y)
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
#       " parameter settings." % ((time() - start), n_iter_search))
# report(random_search.cv_results_)

"""
RandomizedSearchCV took 155.39 seconds for 1000 candidates parameter settings.
Model with rank: 1
Mean validation score: 0.758 (std: 0.031)
Parameters: {'kernel': 'rbf', 'C': 844.42928120986517, 'gamma': 0.002443261152098474, 'class_weight': 'balanced'}

Model with rank: 2
Mean validation score: 0.745 (std: 0.032)
Parameters: {'kernel': 'rbf', 'C': 574.80477276490637, 'gamma': 0.0050353231162486569, 'class_weight': 'balanced'}

Model with rank: 3
Mean validation score: 0.743 (std: 0.043)
Parameters: {'kernel': 'rbf', 'C': 52.899622890128853, 'gamma': 0.0082409957265795475, 'class_weight': 'balanced'}
"""

# params = {'kernel': 'rbf', 'C': 1000, 'gamma': 0.0001}
# clf = SVC(**params)
# clf.fit(X_train, y_train)
# print(clf.score(X_test, y_test))
# print('------------')
# scores = cross_val_score(clf, X, y, cv=4, n_jobs=-1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))