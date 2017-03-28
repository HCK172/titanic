# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt

# machine learning
from sklearn.svm import SVR, SVC
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.metrics import mean_squared_error

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

# gender based model
Y_pred_gender = test_data['Sex']

droplist = 'Survived PassengerId Age_Known Sex Cabin_Known'.split()
data = training_data.drop(droplist, axis=1)
# ensmeble training and test set
X, y = data, training_data['Survived']
offset = int(data.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

params = {'n_estimators': 420, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'lad'}
clf = ensemble.GradientBoostingRegressor(**params)


clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)

# compute test set deviance
# test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
#
# for i, y_pred in enumerate(clf.staged_predict(X_test)):
#     test_score[i] = clf.loss_(y_test, y_pred)
#
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title('Deviance')
# plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
#          label='Training Set Deviance')
# plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
#          label='Test Set Deviance')
# plt.legend(loc='upper right')
# plt.xlabel('Boosting Iterations')
# plt.ylabel('Deviance')
# plt.show()
#
# feature_importance = clf.feature_importances_
# # make importances relative to max importance
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
# sorted_idx = np.argsort(feature_importance)
# pos = np.arange(sorted_idx.shape[0]) + .5
# plt.subplot(1, 2, 2)
# plt.barh(pos, feature_importance[sorted_idx], align='center')
# plt.yticks(pos, X.columns.values[sorted_idx])
# plt.xlabel('Relative Importance')
# plt.title('Variable Importance')
# plt.show()

# predict survival status
Y_pred_clf = clf.predict(test_data.drop(droplist[1:], axis=1))
Y_pred_clf = Y_pred_clf.round().astype(int)

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

# submission file
# submission = pd.DataFrame({
#         "PassengerId": test_data['PassengerId'].astype(int),
#         "Survived": Y_pred
#     })
#
# submission.to_csv('../submission.csv', index=False)