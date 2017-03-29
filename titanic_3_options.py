# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt
#import seaborn as sns
from pandas.tools.plotting import scatter_matrix

# machine learning
from sklearn.svm import SVR, SVC
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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
print(training_data.columns.values)

# gender based model
Y_pred_gender = test_data['Sex']
print("Gender model has %d survivors" %np.sum(Y_pred_gender))

# MLP
droplist = 'Survived PassengerId Age_Known Cabin_Known'.split()
data = training_data.drop(droplist, axis=1)
# ensmeble training and test set
X, y = data, training_data['Survived']
offset = int(data.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# MLP
clf = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(15, 8), random_state=1)
clf.fit(X, y)
Y_pred_MLP = clf.predict(test_data.drop(droplist[1:], axis=1))
print("MLP model has %d survivors" %np.sum(Y_pred_MLP))

# ensemble
droplist = 'Survived PassengerId Age_Known Cabin_Known Title Fare Cabin Embarked'.split()
data = training_data.drop(droplist, axis=1)
# ensmeble training and test set
X, y = data, training_data['Survived']
offset = int(data.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
# ensemble GBC
params = {'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 2,
          'learning_rate': 0.11}
clf = ensemble.GradientBoostingClassifier(**params)
clf.fit(X_train, y_train)

# predict survival status
Y_pred_GBC = clf.predict(test_data.drop(droplist[1:], axis=1))
Y_pred_GBC = Y_pred_GBC.round().astype(float)
print("GBC model has %d survivors" %np.sum(Y_pred_GBC))

# Support Vector Machines
droplist = 'Survived PassengerId Age_Known Cabin_Known Pclass Sex Age Embarked'.split()
data = training_data.drop(droplist, axis=1)
# ensmeble training and test set
X, y = data, training_data['Survived']

svc = SVC()
svc.fit(X, y)
Y_pred_svm = svc.predict(test_data.drop(droplist[1:], axis=1)).astype(float)
print("SVM model has %d survivors" %np.sum(Y_pred_svm))

# combine the models
Y_pred = np.add(0.3*Y_pred_MLP, 0.3*Y_pred_gender, 0.4*Y_pred_GBC)
Y_pred = Y_pred.round().astype(int)
print("Combined model has %d survivors" %np.sum(Y_pred))


prev_df = pd.read_csv("../submission_3_models.csv")
best = pd.read_csv("../submission_GBC.csv")
print("The best model has %d survivors" %np.sum(best['Survived']))
print("number of different elements is %d" %np.sum(np.abs(np.add(Y_pred, -1.0*best['Survived']))))
Y_pred = Y_pred_MLP.astype(int)

# submission file
submission = pd.DataFrame({
        "PassengerId": test_data['PassengerId'].astype(int),
        "Survived": Y_pred
    })

submission.to_csv('../submission.csv', index=False)