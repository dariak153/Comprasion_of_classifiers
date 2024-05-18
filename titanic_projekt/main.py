import sklearn
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
import numpy as np
from scipy.stats import uniform, randint

import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb

from itertools import combinations
import warnings
import json

def title_categories() -> dict:
    return {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master", "Dr": "Dr", "Rev": "Rev",
                      "Col": "Military", "Mlle": "Miss", "Major": "Military", "Ms": "Miss", "Mme": "Mrs", "Sir": "Nobility",
                      "Capt": "Military", "Lady": "Nobility", "the Countess": "Nobility", "Jonkheer": "Nobility",
                      "Don": "Nobility", "Dona": "Nobility"}

def main():
    # import the data
    titanic = fetch_openml("Titanic", version=1, as_frame=True)
    pd.set_option('display.max_columns', None)
    #print(titanic.data.describe())

    # remove column home.dest, boat and body
    titanic.data = titanic.data.drop(columns=['home.dest', 'boat', 'body'], inplace=False, errors='ignore')
    titanic.data.to_csv("titanic_dataset.csv", index=False)
    
    #print(titanic.data.head())

    # print datatypes and number of missing values in each column
    #print(titanic.data.dtypes)
    print(titanic.data.isnull().sum())

    # group values in age column by each 10 years
    bins = list(np.arange(0, 90, 10))
    titanic.data['age_group'] = pd.cut(titanic.data['age'], bins=bins, labels=bins[:-1])

    # get title from name column
    titanic.data['title'] = titanic.data['name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    #print(titanic.data['title'].value_counts())
    title_category = title_categories()
    titanic.data['title_categorical'] = titanic.data['title'].map(title_category)
    #print(titanic.data['title_categorical'].value_counts())

    # sum of sibsp and parch columns
    titanic.data['family_size'] = titanic.data['sibsp'] + titanic.data['parch'] + 1
    #print(titanic.data['family_size'].value_counts())
    titanic.data['fare_per_person'] = titanic.data['fare'] / titanic.data['family_size']
    print(titanic.data['fare_per_person'].describe())


    # divide the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(titanic['data'], titanic['target'], test_size=0.1, random_state=42)

    # check distribution of the dataset in training and testing
    # print(y_train.value_counts())
    # create random guess for x_test
    y_test_guess = np.random.choice([0, 1], p=[0.625, 0.375], size=X_test.shape[0])
    #print(y_test_guess)
    # calculate accuracy
    accuracy_dict = {}
    accuracy = (np.int_(y_test.to_numpy()) == y_test_guess).mean()
    #print('Random guess accuracy: ', accuracy)
    accuracy_dict['Random guess'] = accuracy

    # check if someone was travelling alone
    X_train['is_alone'] = (X_train['family_size'] == 1).astype(int)
    X_test['is_alone'] = (X_test['family_size'] == 1).astype(int)

    # check if someone had bough multiple cabins by counting number of letters in cabin column
    X_train['multiple_cabins'] = X_train['cabin'].apply(lambda x: -1 if pd.isna(x) else (1 if len(x.split()) > 1 else 0))
    X_test['multiple_cabins'] = X_test['cabin'].apply(lambda x: -1 if pd.isna(x) else (1 if len(x.split()) > 1 else 0))
    #print(X_train['multiple_cabins'].value_counts())

    # impute missing values for fare and fare_per_person by using data from sibsp, parch and pclass columns and KNNImputer
    imputer = KNNImputer(n_neighbors=2, missing_values=np.nan)
    X_train[['fare_per_person', 'sibsp', 'parch', 'pclass']] = imputer.fit_transform(X_train[['fare_per_person', 'sibsp', 'parch', 'pclass']])
    X_test[['fare_per_person', 'sibsp', 'parch', 'pclass']] = imputer.transform(X_test[['fare_per_person', 'sibsp', 'parch', 'pclass']])
    # search for missing values in column fare and replace it with fare_per_person * family_size
    X_train['fare'] = X_train['fare'].fillna(X_train['fare_per_person'] * X_train['family_size'])
    X_test['fare'] = X_test['fare'].fillna(X_test['fare_per_person'] * X_test['family_size'])

    # get cabin letter from cabin column if it is not missing
    X_train['deck'] = X_train['cabin'].str[0]
    X_test['deck'] = X_test['cabin'].str[0]
    # if deck is equal T then replace it with A - only one person was in deck T
    X_train['deck'] = X_train['deck'].replace('T', 'A')
    X_test['deck'] = X_test['deck'].replace('T', 'A')

    # use LabelEncoder to encode the deck column
    deck_le = LabelEncoder()
    X_train['deck_encoded'] = deck_le.fit_transform(X_train['deck'])
    X_test['deck_encoded'] = deck_le.transform(X_test['deck'])
    #print(X_train['deck_encoded'].value_counts())

    # use LabelEncoder to encode the age_group column
    age_group_le = LabelEncoder()
    X_train['age_group_encoded'] = age_group_le.fit_transform(X_train['age_group'])
    X_test['age_group_encoded'] = age_group_le.transform(X_test['age_group'])
    #print(X_train['age_group_encoded'].value_counts())

    # use LabelEncoder to encode the title_categorical column
    title_categorical_le = LabelEncoder()
    X_train['title_categorical_encoded'] = title_categorical_le.fit_transform(X_train['title_categorical'])
    X_test['title_categorical_encoded'] = title_categorical_le.transform(X_test['title_categorical'])
    #print(X_train['title_categorical_encoded'].value_counts())

    # use LabelEncoder to encode the sex column
    sex_le = LabelEncoder()
    X_train['sex_encoded'] = sex_le.fit_transform(X_train['sex'])
    X_test['sex_encoded'] = sex_le.transform(X_test['sex'])
    # print(X_train['sex_encoded'].value_counts())

    # use LabelEncoder to encode the embarked column
    embarked_le = LabelEncoder()
    X_train['embarked_encoded'] = embarked_le.fit_transform(X_train['embarked'])
    X_test['embarked_encoded'] = embarked_le.transform(X_test['embarked'])
    # print(X_train['embarked_encoded'].value_counts())

    # find correlation between columns
    corr = X_train[['deck_encoded', 'pclass', 'fare_per_person', 'family_size', 'title_categorical_encoded', 'sex_encoded',
                    'embarked_encoded', 'sibsp', 'parch', 'age_group_encoded', 'is_alone', 'multiple_cabins']].corr()

    # find correlation between columns where deck is not missing
    correlation_no_missing = X_train[X_train['deck_encoded'] != 7][['deck_encoded', 'pclass', 'fare_per_person', 'family_size',
                                        'title_categorical_encoded', 'sex_encoded', 'embarked_encoded', 'sibsp',
                                        'parch', 'age_group_encoded', 'is_alone', 'multiple_cabins']].corr()

    # replacing missing values in deck column with -1
    X_train['deck_encoded_no_missing'] = X_train['deck_encoded'].replace(7, -1)
    X_test['deck_encoded_no_missing'] = X_test['deck_encoded'].replace(7, -1)

    # impute missing values for deck column by using data from pclass and fare_per_person columns and IterativeImputer
    deck_imputer = KNNImputer(n_neighbors=2, missing_values=-1)
    X_train[['deck_encoded_no_missing', 'pclass', 'fare_per_person', 'embarked_encoded', 'is_alone']] = (
        deck_imputer.fit_transform(X_train[['deck_encoded_no_missing', 'pclass', 'fare_per_person',
                                            'embarked_encoded', 'is_alone']]))
    X_test[['deck_encoded_no_missing', 'pclass', 'fare_per_person', 'embarked_encoded', 'is_alone']] = (
        deck_imputer.transform(X_test[['deck_encoded_no_missing', 'pclass', 'fare_per_person',
                                       'embarked_encoded', 'is_alone']]))


    # check correlation after imputing missing values
    correlation_imputed = X_train[['deck_encoded_no_missing', 'pclass', 'fare_per_person', 'family_size', 'title_categorical_encoded', 'sex_encoded',
                    'embarked_encoded', 'sibsp', 'parch', 'age_group_encoded', 'is_alone', 'multiple_cabins']].corr()

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    # Plot correlation matrix 1
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
    axes[0].set_title('Initial correlation Matrix')

    # Plot correlation matrix 2
    sns.heatmap(correlation_no_missing, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
    axes[1].set_title('Correlation Matrix without missing values for deck column')

    # Plot correlation matrix 3
    sns.heatmap(correlation_imputed, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[2])
    axes[2].set_title('Correlation Matrix after imputing missing values for deck column')
    plt.show()

    # replace missing values in age_group_encoded column with -1
    X_train['age_group_encoded'] = X_train['age_group_encoded'].replace(8, -1)
    X_test['age_group_encoded'] = X_test['age_group_encoded'].replace(8, -1)

    # impute missing values for age_group_encoded column by using IterativeImputer
    age_group_imputer = IterativeImputer(max_iter=10, random_state=42, missing_values=-1)
    X_train[['age_group_encoded', 'sibsp', 'title_categorical_encoded', 'is_alone', 'parch']] = (
        age_group_imputer.fit_transform(X_train[['age_group_encoded', 'sibsp', 'title_categorical_encoded', 'is_alone', 'parch']]))
    X_test[['age_group_encoded', 'sibsp', 'title_categorical_encoded', 'is_alone', 'parch']] = (age_group_imputer.transform(
        X_test[['age_group_encoded', 'sibsp', 'title_categorical_encoded', 'is_alone', 'parch']]))

    # check correlation after imputing missing values
    correlation_no_missing_age = X_train[pd.notna(X_train['age'])][['deck_encoded', 'pclass', 'fare_per_person', 'family_size',
                                        'title_categorical_encoded', 'sex_encoded', 'embarked_encoded', 'sibsp',
                                        'parch', 'age_group_encoded', 'is_alone', 'multiple_cabins']].corr()

    correlation_imputed_age = X_train[['deck_encoded', 'pclass', 'fare_per_person', 'family_size',
                                        'title_categorical_encoded', 'sex_encoded', 'embarked_encoded', 'sibsp',
                                        'parch', 'age_group_encoded', 'is_alone', 'multiple_cabins']].corr()

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    # Plot correlation matrix 1
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
    axes[0].set_title('Initial correlation Matrix')
    #Plot correlation matrix 2
    sns.heatmap(correlation_no_missing_age, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
    axes[1].set_title('Correlation Matrix without missing values for age_group_encoded column')
    #Plot correlation matrix 3
    sns.heatmap(correlation_imputed_age, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[2])
    axes[2].set_title('Correlation Matrix after imputing missing values for age_group_encoded column')
    plt.show()

    # impute missing values for embarked_encoded column by using IterativeImputer
    embarked_imputer = IterativeImputer(max_iter=10, random_state=42, missing_values=-1)
    X_train[['embarked_encoded', 'fare_per_person', 'deck_encoded', 'pclass']] = (
        embarked_imputer.fit_transform(X_train[['embarked_encoded', 'fare_per_person', 'deck_encoded', 'pclass']]))
    X_test[['embarked_encoded', 'fare_per_person', 'deck_encoded', 'pclass']] = (
        embarked_imputer.transform(X_test[['embarked_encoded', 'fare_per_person', 'deck_encoded', 'pclass']]))

    X_train['fare_per_person'].plot.hist(bins=10, color='skyblue', edgecolor='black')

    # create a robust scaller
    scaller = RobustScaler()
    X_train['fare_per_person_scaled'] = scaller.fit_transform(X_train[['fare_per_person']])
    # create bins
    bins = [-1, -0.25, -0.05, 0.1, 1.5, 4, np.inf]
    # create labels
    labels = np.linspace(0, 1, len(bins)-1)
    # create new column with scaled values
    X_train['fare_per_person_binned'] = pd.cut(X_train['fare_per_person_scaled'], bins=bins, labels=labels).astype(float)
    # plot histogram
    X_train['fare_per_person_binned'].plot.hist(bins=10, color='green', edgecolor='black')
    X_test['fare_per_person_scaled'] = scaller.transform(X_test[['fare_per_person']])
    X_test['fare_per_person_binned'] = pd.cut(X_test['fare_per_person_scaled'], bins=bins, labels=labels).astype(float)
    #X_train['fare_per_person_scaled'].plot.hist(bins=10, color='red', edgecolor='black')


    # Customize the plot (optional)
    plt.title('Histogram of fare_per_person')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    #plt.show()

    # check final correlation
    data_train = pd.concat([X_train, y_train], axis=1)
    #print(data_train.columns)
    correlation_matrix = data_train[['survived', 'pclass', 'sibsp', 'parch',
       'family_size', 'fare_per_person', 'is_alone', 'deck_encoded',
       'age_group_encoded', 'title_categorical_encoded', 'sex_encoded',
       'embarked_encoded', 'deck_encoded_no_missing', 'multiple_cabins', 'fare_per_person_binned']].corr()

    # Create plot
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Final correlation Matrix')
    plt.show()

    ### Classification

    ## Basic classifiers

    # create Random Forest Classifier
    rfc_clf = RandomForestClassifier(random_state=42)
    # create a Logistic Regression Classifier
    lrc_clf = LogisticRegression(random_state=42, max_iter=1000)
    # create a Decision Tree Classifier
    dtc_clf = DecisionTreeClassifier(random_state=42)
    # create a SVM Classifier
    svm_clf = SVC(random_state=42, probability=True)
    # create a KNN Classifier
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    # create a Naive Bayes Classifier
    nb_clf = GaussianNB()
    # create a XGBoost Classifier
    xgb_clf = xgb.XGBClassifier(random_state=42)
    # create a LightGBM Classifier
    lgbm_clf = lgb.LGBMClassifier(random_state=42, verbosity=0)
    # create a Voting Classifier
    voting_clf_h = VotingClassifier(estimators=[('rf', rfc_clf), ('lr', lrc_clf), ('dt', dtc_clf), ('svm', svm_clf),
                                              ('knn', knn_clf), ('nb', nb_clf), ('xgb', xgb_clf), ('lgbm', lgbm_clf)], voting='hard')
    voting_clf_s = VotingClassifier(estimators=[('rf', rfc_clf), ('lr', lrc_clf), ('dt', dtc_clf), ('svm', svm_clf),
                                              ('knn', knn_clf), ('nb', nb_clf), ('xgb', xgb_clf), ('lgbm', lgbm_clf)], voting='soft')

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    classifiers = {'Decision Tree Classifier': dtc_clf, 'Random Forest Classifier': rfc_clf, 'Logistic Regression Classifier': lrc_clf,
                   'SVM Classifier': svm_clf, 'KNN Classifier': knn_clf, 'Naive Bayes Classifier': nb_clf,
                   'XGBoost Classifier': xgb_clf, 'LightGBM Classifier': lgbm_clf, 'Voting Classifier Hard': voting_clf_h,
                   'Voting Classifier Soft': voting_clf_s}


    all_usable_features = ['pclass', 'sibsp', 'parch',
       'family_size', 'fare_per_person', 'is_alone', 'deck_encoded',
       'age_group_encoded', 'title_categorical_encoded', 'sex_encoded',
       'embarked_encoded', 'deck_encoded_no_missing', 'multiple_cabins', 'fare_per_person_binned', 'fare', 'fare_per_person_scaled']

    # Filter XGBoost warnings
    warnings.filterwarnings('once', module='xgboost')

    cross_val_dict = {}
    for key, value in classifiers.items():

        # do cross validation
        cv_scores = cross_val_score(value, X_train[all_usable_features], y_train, cv=10, n_jobs=-1).mean()
        cross_val_dict[key+" basic"] = cv_scores

        value.fit(X_train[all_usable_features], y_train)
        prediction = value.predict(X_test[all_usable_features])
        # calculate accuracy
        accuracy = accuracy_score(y_test, prediction)
        accuracy_dict[key+" basic"] = accuracy
        if key == 'XGBoost Classifier':
            # plot feature importance
            plot_importance(value)

    plt.show()

    ### Create a clasifier
    svm_clf = SVC(C=2.3089382562214897, gamma='auto', probability=True, random_state=42)
    features_for_svm = ['family_size', 'fare_per_person', 'is_alone', 'age_group_encoded', 'sex_encoded', 'embarked_encoded', 'fare_per_person_scaled']
    svm_clf.fit(X_train[features_for_svm], y_train)
    prediction = svm_clf.predict(X_test[features_for_svm])
    accuracy = accuracy_score(y_test, prediction)
    print("Accuracy for SVM Classifier with parameters: C=2.3089382562214897, gamma='auto', probability=True, random_state=42; \n using features: 'family_size', 'fare_per_person', 'is_alone', 'age_group_encoded', 'sex_encoded', 'embarked_encoded', 'fare_per_person_scaled' :\n",accuracy)

def killer_loop():
    # import the data
    titanic = fetch_openml("Titanic", version=1, as_frame=True)
    pd.set_option('display.max_columns', None)
    #print(titanic.data.describe())

    # remove column home.dest, boat and body
    titanic.data = titanic.data.drop(columns=['home.dest', 'boat', 'body'], inplace=False, errors='ignore')

    #print(titanic.data.head())

    # print datatypes and number of missing values in each column
    #print(titanic.data.dtypes)
    print(titanic.data.isnull().sum())

    # group values in age column by each 10 years
    bins = list(np.arange(0, 90, 10))
    titanic.data['age_group'] = pd.cut(titanic.data['age'], bins=bins, labels=bins[:-1])

    # get title from name column
    titanic.data['title'] = titanic.data['name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    #print(titanic.data['title'].value_counts())
    title_category = title_categories()

    titanic.data['title_categorical'] = titanic.data['title'].map(title_category)
    #print(titanic.data['title_categorical'].value_counts())

    # sum of sibsp and parch columns
    titanic.data['family_size'] = titanic.data['sibsp'] + titanic.data['parch'] + 1
    #print(titanic.data['family_size'].value_counts())
    titanic.data['fare_per_person'] = titanic.data['fare'] / titanic.data['family_size']
    print(titanic.data['fare_per_person'].describe())


    # divide the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(titanic['data'], titanic['target'], test_size=0.1, random_state=42)

    # check distribution of the dataset in training and testing
    # print(y_train.value_counts())
    # create random guess for x_test
    y_test_guess = np.random.choice([0, 1], p=[0.625, 0.375], size=X_test.shape[0])
    #print(y_test_guess)
    # calculate accuracy
    accuracy_dict = {}
    accuracy = (np.int_(y_test.to_numpy()) == y_test_guess).mean()
    #print('Random guess accuracy: ', accuracy)
    accuracy_dict['Random guess'] = accuracy

    # check if someone was travelling alone
    X_train['is_alone'] = (X_train['family_size'] == 1).astype(int)
    X_test['is_alone'] = (X_test['family_size'] == 1).astype(int)

    # check if someone had bough multiple cabins by counting number of letters in cabin column
    X_train['multiple_cabins'] = X_train['cabin'].apply(lambda x: -1 if pd.isna(x) else (1 if len(x.split()) > 1 else 0))
    X_test['multiple_cabins'] = X_test['cabin'].apply(lambda x: -1 if pd.isna(x) else (1 if len(x.split()) > 1 else 0))
    #print(X_train['multiple_cabins'].value_counts())

    # impute missing values for fare and fare_per_person by using data from sibsp, parch and pclass columns and KNNImputer
    imputer = KNNImputer(n_neighbors=2, missing_values=np.nan)
    X_train[['fare_per_person', 'sibsp', 'parch', 'pclass']] = imputer.fit_transform(X_train[['fare_per_person', 'sibsp', 'parch', 'pclass']])
    X_test[['fare_per_person', 'sibsp', 'parch', 'pclass']] = imputer.transform(X_test[['fare_per_person', 'sibsp', 'parch', 'pclass']])
    # search for missing values in column fare and replace it with fare_per_person * family_size
    X_train['fare'] = X_train['fare'].fillna(X_train['fare_per_person'] * X_train['family_size'])
    X_test['fare'] = X_test['fare'].fillna(X_test['fare_per_person'] * X_test['family_size'])

    # get cabin letter from cabin column if it is not missing
    X_train['deck'] = X_train['cabin'].str[0]
    X_test['deck'] = X_test['cabin'].str[0]
    # if deck is equal T then replace it with A - only one person was in deck T
    X_train['deck'] = X_train['deck'].replace('T', 'A')
    X_test['deck'] = X_test['deck'].replace('T', 'A')

    # use LabelEncoder to encode the deck column
    deck_le = LabelEncoder()
    X_train['deck_encoded'] = deck_le.fit_transform(X_train['deck'])
    X_test['deck_encoded'] = deck_le.transform(X_test['deck'])
    #print(X_train['deck_encoded'].value_counts())

    # use LabelEncoder to encode the age_group column
    age_group_le = LabelEncoder()
    X_train['age_group_encoded'] = age_group_le.fit_transform(X_train['age_group'])
    X_test['age_group_encoded'] = age_group_le.transform(X_test['age_group'])
    #print(X_train['age_group_encoded'].value_counts())

    # use LabelEncoder to encode the title_categorical column
    title_categorical_le = LabelEncoder()
    X_train['title_categorical_encoded'] = title_categorical_le.fit_transform(X_train['title_categorical'])
    X_test['title_categorical_encoded'] = title_categorical_le.transform(X_test['title_categorical'])
    #print(X_train['title_categorical_encoded'].value_counts())

    # use LabelEncoder to encode the sex column
    sex_le = LabelEncoder()
    X_train['sex_encoded'] = sex_le.fit_transform(X_train['sex'])
    X_test['sex_encoded'] = sex_le.transform(X_test['sex'])
    # print(X_train['sex_encoded'].value_counts())

    # use LabelEncoder to encode the embarked column
    embarked_le = LabelEncoder()
    X_train['embarked_encoded'] = embarked_le.fit_transform(X_train['embarked'])
    X_test['embarked_encoded'] = embarked_le.transform(X_test['embarked'])
    # print(X_train['embarked_encoded'].value_counts())

    # find correlation between columns
    corr = X_train[['deck_encoded', 'pclass', 'fare_per_person', 'family_size', 'title_categorical_encoded', 'sex_encoded',
                    'embarked_encoded', 'sibsp', 'parch', 'age_group_encoded', 'is_alone', 'multiple_cabins']].corr()

    # find correlation between columns where deck is not missing
    correlation_no_missing = X_train[X_train['deck_encoded'] != 7][['deck_encoded', 'pclass', 'fare_per_person', 'family_size',
                                        'title_categorical_encoded', 'sex_encoded', 'embarked_encoded', 'sibsp',
                                        'parch', 'age_group_encoded', 'is_alone', 'multiple_cabins']].corr()

    # replacing missing values in deck column with -1
    X_train['deck_encoded_no_missing'] = X_train['deck_encoded'].replace(7, -1)
    X_test['deck_encoded_no_missing'] = X_test['deck_encoded'].replace(7, -1)

    # impute missing values for deck column by using data from pclass and fare_per_person columns and IterativeImputer
    deck_imputer = KNNImputer(n_neighbors=2, missing_values=-1)
    X_train[['deck_encoded_no_missing', 'pclass', 'fare_per_person', 'embarked_encoded', 'is_alone']] = (
        deck_imputer.fit_transform(X_train[['deck_encoded_no_missing', 'pclass', 'fare_per_person',
                                            'embarked_encoded', 'is_alone']]))
    X_test[['deck_encoded_no_missing', 'pclass', 'fare_per_person', 'embarked_encoded', 'is_alone']] = (
        deck_imputer.transform(X_test[['deck_encoded_no_missing', 'pclass', 'fare_per_person',
                                       'embarked_encoded', 'is_alone']]))


    # check correlation after imputing missing values
    correlation_imputed = X_train[['deck_encoded_no_missing', 'pclass', 'fare_per_person', 'family_size', 'title_categorical_encoded', 'sex_encoded',
                    'embarked_encoded', 'sibsp', 'parch', 'age_group_encoded', 'is_alone', 'multiple_cabins']].corr()

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
    # Plot correlation matrix 1
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
    axes[0].set_title('Initial correlation Matrix')

    # Plot correlation matrix 2
    sns.heatmap(correlation_no_missing, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
    axes[1].set_title('Correlation Matrix without missing values for deck column')

    # Plot correlation matrix 3
    sns.heatmap(correlation_imputed, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[2])
    axes[2].set_title('Correlation Matrix after imputing missing values for deck column')
    plt.show()

    # replace missing values in age_group_encoded column with -1
    X_train['age_group_encoded'] = X_train['age_group_encoded'].replace(8, -1)
    X_test['age_group_encoded'] = X_test['age_group_encoded'].replace(8, -1)

    # impute missing values for age_group_encoded column by using IterativeImputer
    age_group_imputer = IterativeImputer(max_iter=10, random_state=42, missing_values=-1)
    X_train[['age_group_encoded', 'sibsp', 'title_categorical_encoded', 'is_alone', 'parch']] = (
        age_group_imputer.fit_transform(X_train[['age_group_encoded', 'sibsp', 'title_categorical_encoded', 'is_alone', 'parch']]))
    X_test[['age_group_encoded', 'sibsp', 'title_categorical_encoded', 'is_alone', 'parch']] = (age_group_imputer.transform(
        X_test[['age_group_encoded', 'sibsp', 'title_categorical_encoded', 'is_alone', 'parch']]))

    # check correlation after imputing missing values
    correlation_no_missing_age = X_train[pd.notna(X_train['age'])][['deck_encoded', 'pclass', 'fare_per_person', 'family_size',
                                        'title_categorical_encoded', 'sex_encoded', 'embarked_encoded', 'sibsp',
                                        'parch', 'age_group_encoded', 'is_alone', 'multiple_cabins']].corr()

    correlation_imputed_age = X_train[['deck_encoded', 'pclass', 'fare_per_person', 'family_size',
                                        'title_categorical_encoded', 'sex_encoded', 'embarked_encoded', 'sibsp',
                                        'parch', 'age_group_encoded', 'is_alone', 'multiple_cabins']].corr()

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
    # Plot correlation matrix 1
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
    axes[0].set_title('Initial correlation Matrix')
    #Plot correlation matrix 2
    sns.heatmap(correlation_no_missing_age, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
    axes[1].set_title('Correlation Matrix without missing values for age_group_encoded column')
    #Plot correlation matrix 3
    sns.heatmap(correlation_imputed_age, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[2])
    axes[2].set_title('Correlation Matrix after imputing missing values for age_group_encoded column')
    plt.show()

    # impute missing values for embarked_encoded column by using IterativeImputer
    embarked_imputer = IterativeImputer(max_iter=10, random_state=42, missing_values=-1)
    X_train[['embarked_encoded', 'fare_per_person', 'deck_encoded', 'pclass']] = (
        embarked_imputer.fit_transform(X_train[['embarked_encoded', 'fare_per_person', 'deck_encoded', 'pclass']]))
    X_test[['embarked_encoded', 'fare_per_person', 'deck_encoded', 'pclass']] = (
        embarked_imputer.transform(X_test[['embarked_encoded', 'fare_per_person', 'deck_encoded', 'pclass']]))

    X_train['fare_per_person'].plot.hist(bins=10, color='skyblue', edgecolor='black')

    # create a robust scaller
    scaller = RobustScaler()
    X_train['fare_per_person_scaled'] = scaller.fit_transform(X_train[['fare_per_person']])
    # create bins
    bins = [-1, -0.25, -0.05, 0.1, 1.5, 4, np.inf]
    # create labels
    labels = np.linspace(0, 1, len(bins)-1)
    # create new column with scaled values
    X_train['fare_per_person_binned'] = pd.cut(X_train['fare_per_person_scaled'], bins=bins, labels=labels).astype(float)
    # plot histogram
    X_train['fare_per_person_binned'].plot.hist(bins=10, color='green', edgecolor='black')
    X_test['fare_per_person_scaled'] = scaller.transform(X_test[['fare_per_person']])
    X_test['fare_per_person_binned'] = pd.cut(X_test['fare_per_person_scaled'], bins=bins, labels=labels).astype(float)
    #X_train['fare_per_person_scaled'].plot.hist(bins=10, color='red', edgecolor='black')


    # Customize the plot (optional)
    plt.title('Histogram of fare_per_person')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    #plt.show()

    # check final correlation
    data_train = pd.concat([X_train, y_train], axis=1)
    #print(data_train.columns)
    correlation_matrix = data_train[['survived', 'pclass', 'sibsp', 'parch',
       'family_size', 'fare_per_person', 'is_alone', 'deck_encoded',
       'age_group_encoded', 'title_categorical_encoded', 'sex_encoded',
       'embarked_encoded', 'deck_encoded_no_missing', 'multiple_cabins', 'fare_per_person_binned']].corr()

    # Create plot
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Final correlation Matrix')
    plt.show()

    ### Classification

    ## Basic classifiers

    # create Random Forest Classifier
    rfc_clf = RandomForestClassifier(random_state=42)
    # create a Logistic Regression Classifier
    lrc_clf = LogisticRegression(random_state=42, max_iter=1000)
    # create a Decision Tree Classifier
    dtc_clf = DecisionTreeClassifier(random_state=42)
    # create a SVM Classifier
    svm_clf = SVC(random_state=42, probability=True)
    # create a KNN Classifier
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    # create a Naive Bayes Classifier
    nb_clf = GaussianNB()
    # create a XGBoost Classifier
    xgb_clf = xgb.XGBClassifier(random_state=42)
    # create a LightGBM Classifier
    lgbm_clf = lgb.LGBMClassifier(random_state=42, verbosity=0)
    # create a Voting Classifier
    voting_clf_h = VotingClassifier(estimators=[('rf', rfc_clf), ('lr', lrc_clf), ('dt', dtc_clf), ('svm', svm_clf),
                                              ('knn', knn_clf), ('nb', nb_clf), ('xgb', xgb_clf), ('lgbm', lgbm_clf)], voting='hard')
    voting_clf_s = VotingClassifier(estimators=[('rf', rfc_clf), ('lr', lrc_clf), ('dt', dtc_clf), ('svm', svm_clf),
                                              ('knn', knn_clf), ('nb', nb_clf), ('xgb', xgb_clf), ('lgbm', lgbm_clf)], voting='soft')

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    classifiers = {'Decision Tree Classifier': dtc_clf, 'Random Forest Classifier': rfc_clf, 'Logistic Regression Classifier': lrc_clf,
                   'SVM Classifier': svm_clf, 'KNN Classifier': knn_clf, 'Naive Bayes Classifier': nb_clf,
                   'XGBoost Classifier': xgb_clf, 'LightGBM Classifier': lgbm_clf, 'Voting Classifier Hard': voting_clf_h,
                   'Voting Classifier Soft': voting_clf_s}


    all_usable_features = ['pclass', 'sibsp', 'parch',
       'family_size', 'fare_per_person', 'is_alone', 'deck_encoded',
       'age_group_encoded', 'title_categorical_encoded', 'sex_encoded',
       'embarked_encoded', 'deck_encoded_no_missing', 'multiple_cabins', 'fare_per_person_binned', 'fare', 'fare_per_person_scaled']

    # Filter XGBoost warnings
    warnings.filterwarnings('once', module='xgboost')

    cross_val_dict = {}
    for key, value in classifiers.items():

        # do cross validation
        cv_scores = cross_val_score(value, X_train[all_usable_features], y_train, cv=10, n_jobs=-1).mean()
        cross_val_dict[key+" basic"] = cv_scores

        value.fit(X_train[all_usable_features], y_train)
        prediction = value.predict(X_test[all_usable_features])
        # calculate accuracy
        accuracy = accuracy_score(y_test, prediction)
        accuracy_dict[key+" basic"] = accuracy
        if key == 'XGBoost Classifier':
            # plot feature importance
            plot_importance(value)

    plt.show()

    ### check different features

    # create a list of all combinations
    all_combinations = []

    # Generate all combinations of all lengths
    for r in range(1, len(all_usable_features) + 1):
        combinations_object = combinations(all_usable_features, r)
        combinations_list = list(combinations_object)
        all_combinations += combinations_list

    np.random.seed(42)
    #print(len(all_combinations))
    num_of_combinations_to_check = 300
    combinations_indexes_to_check = np.random.choice(len(all_combinations), num_of_combinations_to_check, replace=False)

    print(combinations_indexes_to_check)


    accuracy_for_combinations = {}
    cross_val_for_combinations = {}
    # initialize a dictionary to store feature importance values
    feature_importance_values = {feature: [0, 0, 0] for feature in X_train.columns}

    for index, i in enumerate(combinations_indexes_to_check):
        curr_combination = list(all_combinations[i])
        # do cross validation
        cv_scores = cross_val_score(xgb_clf, X_train[curr_combination], y_train, cv=10, n_jobs=-1, verbose=0).mean()
        cross_val_for_combinations[tuple(curr_combination)] = [cv_scores, 'xgb']
        xgb_clf.fit(X_train[curr_combination], y_train)
        prediction = xgb_clf.predict(X_test[curr_combination])
        # calculate accuracy
        accuracy = accuracy_score(y_test, prediction)
        accuracy_for_combinations[tuple(curr_combination)] = [accuracy, 'xgb']

        for feature, importance in zip(curr_combination, xgb_clf.feature_importances_):
            feature_importance_values[feature][0] += importance
            feature_importance_values[feature][1] += 1
            feature_importance_values[feature][2] += cv_scores

        # do cross validation
        cv_scores = cross_val_score(rfc_clf, X_train[curr_combination], y_train, cv=10, n_jobs=-1).mean()
        cross_val_for_combinations[tuple(curr_combination)] = [cv_scores, 'rfc']
        rfc_clf.fit(X_train[curr_combination], y_train)
        prediction = rfc_clf.predict(X_test[curr_combination])
        # calculate accuracy
        accuracy = accuracy_score(y_test, prediction)
        accuracy_for_combinations[tuple(curr_combination)] = [accuracy, 'rfc']
        print("Finished cobinations: ", index, " out of ", num_of_combinations_to_check)

    # sort the accuracy_for_combinations dictionary and cross_val_for_combinations dictionary
    accuracy_for_combinations_sorted = dict(sorted(accuracy_for_combinations.items(), key=lambda item: item[1][0], reverse=True))
    cross_val_for_combinations_sorted = dict(sorted(cross_val_for_combinations.items(), key=lambda item: item[1][0], reverse=True))

    # calculate mean feature importance values
    for feature in feature_importance_values:
        if feature_importance_values[feature][1] != 0:
            feature_importance_values[feature][0] /= feature_importance_values[feature][1]
            feature_importance_values[feature][2] /= feature_importance_values[feature][1]

    # sort the feature_importance_values dictionary
    feature_importance_values_sorted = dict(sorted(feature_importance_values.items(), key=lambda item: item[1][0], reverse=True))

    print(accuracy_for_combinations_sorted)
    print(cross_val_for_combinations_sorted)
    print(feature_importance_values_sorted)

    # # print cross validation scores and accuracy
    # print(cross_val_dict)
    # print(accuracy_dict)
    # #print(best_estimators)

    ### hyperparameter tuning for different combinations of features

    # create a list of combinations to check
    combinations_to_check = []
    for i, key in enumerate(cross_val_for_combinations_sorted.items()):
        if i >= 20:
            break
        combinations_to_check.append(list(key[0]))

    # create a vector of 5 most important features
    most_important_features = [feature for feature in feature_importance_values_sorted][:5]
    for i, key in enumerate(feature_importance_values_sorted):
        if i >= 5:
            combinations_to_check.append(list(most_important_features))
            most_important_features.append(key)
        if i >= 10:
            break

    # create a parameter grid for Random Forest Classifier
    param_grid_rfc = {'n_estimators': randint(10, 1000), 'max_depth': randint(1, 100), 'min_samples_split': randint(2, 20),
                      'min_samples_leaf': randint(1, 20), 'max_features': uniform(0, 1)}
    # create a parameter grid for Logistic Regression Classifier
    param_grid_lrc = {'C': uniform(0, 10), 'penalty': ['l1', 'l2']}
    # create a parameter grid for Decision Tree Classifier
    param_grid_dtc = {'max_depth': randint(1, 100), 'min_samples_split': randint(2, 20), 'min_samples_leaf': randint(1, 20)}
    # create a parameter grid for SVM Classifier
    param_grid_svm = {'C': uniform(0, 10), 'gamma': ['scale', 'auto']}
    # create a parameter grid for KNN Classifier
    param_grid_knn = {'n_neighbors': randint(1, 20), 'weights': ['uniform', 'distance']}
    # create a parameter grid for Naive Bayes Classifier
    param_grid_nb = {}
    # create a parameter grid for XGBoost Classifier
    param_grid_xgb = {'n_estimators': randint(10, 1000), 'max_depth': randint(1, 100), 'learning_rate': uniform(0, 1),
                      'colsample_bytree': uniform(0.5, 0.5), 'subsample': uniform(0.5, 0.5), 'gamma': uniform(0, 5),
                      'scale_pos_weight': uniform(0.1, 10)}
    # create a parameter grid for LightGBM Classifier
    param_grid_lgbm = {'n_estimators': randint(10, 1000), 'max_depth': randint(1, 100), 'learning_rate': uniform(0, 1)}

    param_grids = {
        'Decision Tree Classifier': param_grid_dtc,
        'Random Forest Classifier': param_grid_rfc,
        'Logistic Regression Classifier': param_grid_lrc,
        'SVM Classifier': param_grid_svm,
        'KNN Classifier': param_grid_knn,
        'Naive Bayes Classifier': param_grid_nb,
        'XGBoost Classifier': param_grid_xgb,
        'LightGBM Classifier': param_grid_lgbm,
    }

    classifiers_for_tuning = {'Decision Tree Classifier': dtc_clf, 'Random Forest Classifier': rfc_clf, 'Logistic Regression Classifier': lrc_clf,
                   'SVM Classifier': svm_clf ,'KNN Classifier': knn_clf,
                   'XGBoost Classifier': xgb_clf, 'LightGBM Classifier': lgbm_clf, 'Voting Classifier Hard': voting_clf_h,
                   'Voting Classifier Soft': voting_clf_s}

    best_estimators = {}
    for key, value in classifiers_for_tuning.items():
        if key == 'Voting Classifier Soft' or key == 'Voting Classifier Hard':
            continue
        # create a RandomizedSearchCV
        rscv = RandomizedSearchCV(value, param_distributions=param_grids[key], n_iter=50, cv=10, random_state=42, verbose=2,
                                  n_jobs=-1, scoring='accuracy')
        rscv.fit(X_train[all_usable_features], y_train)
        best_estimators[key+" tuned "+str(all_usable_features)] = rscv.best_estimator_
        # predict values
        prediction = rscv.best_estimator_.predict(X_test[all_usable_features])
        # calculate accuracy
        accuracy = accuracy_score(y_test, prediction)
        accuracy_dict[key+" tuned "+str(all_usable_features)] = accuracy
        cross_val_dict[key+" tuned "+str(all_usable_features)] = rscv.best_score_
        print("Finished tuning for: ", key)

    # create a list of tuples of the best estimators for Voting Classifier
    best_estimators_list = [(key, value) for key, value in best_estimators.items() if key != 'Naive Bayes Classifier']

    # create a new Voting Classifier with best estimators
    voting_clf_tuned_h = VotingClassifier(estimators=best_estimators_list, voting='hard')
    voting_clf_tuned_s = VotingClassifier(estimators=best_estimators_list, voting='soft')

    # fit the Voting Classifiers
    voting_clf_tuned_h.fit(X_train[all_usable_features], y_train)
    voting_clf_tuned_s.fit(X_train[all_usable_features], y_train)
    # predict values
    prediction_h = voting_clf_tuned_h.predict(X_test[all_usable_features])
    prediction_s = voting_clf_tuned_s.predict(X_test[all_usable_features])
    # calculate accuracy
    accuracy_h = accuracy_score(y_test, prediction_h)
    accuracy_s = accuracy_score(y_test, prediction_s)
    accuracy_dict["Voting Classifier Hard tuned "+str(all_usable_features)] = accuracy_h
    accuracy_dict["Voting Classifier Soft tuned "+str(all_usable_features)] = accuracy_s
    cross_val_dict["Voting Classifier Hard tuned "+str(all_usable_features)] = voting_clf_tuned_h.score(X_train[all_usable_features], y_train)
    cross_val_dict["Voting Classifier Soft tuned "+str(all_usable_features)] = voting_clf_tuned_s.score(X_train[all_usable_features], y_train)

    # print cross validation scores and accuracy
    #print(cross_val_dict)
    #print(accuracy_dict)
    #print(best_estimators)


    # create a loop to check different combinations of features and use different models and tune their hyperparameters
    for features in combinations_to_check:
        curr_features = features
        print("Checking features: ", curr_features)
        for key, value in classifiers_for_tuning.items():
            if key == 'Voting Classifier Soft' or key == 'Voting Classifier Hard':
                continue
            # create a RandomizedSearchCV
            rscv = RandomizedSearchCV(value, param_distributions=param_grids[key], n_iter=50, cv=10, random_state=42,
                                      verbose=2,
                                      n_jobs=-1, scoring='accuracy')
            rscv.fit(X_train[curr_features], y_train)
            best_estimators[key+ " tuned "+str(curr_features)] = rscv.best_estimator_
            # predict values
            prediction = rscv.best_estimator_.predict(X_test[curr_features])
            # calculate accuracy
            accuracy = accuracy_score(y_test, prediction)
            accuracy_dict[key + " tuned "+str(curr_features)] = accuracy
            cross_val_dict[key + " tuned "+str(curr_features)] = rscv.best_score_
            print("Finished tuning for: ", key)

        # create a list of tuples of the best estimators for Voting Classifier
        best_estimators_list = [(key, value) for key, value in best_estimators.items() if
                                key != 'Naive Bayes Classifier']

        # create a new Voting Classifier with best estimators
        voting_clf_tuned_h = VotingClassifier(estimators=best_estimators_list, voting='hard')
        voting_clf_tuned_s = VotingClassifier(estimators=best_estimators_list, voting='soft')

        # fit the Voting Classifiers
        voting_clf_tuned_h.fit(X_train[curr_features], y_train)
        voting_clf_tuned_s.fit(X_train[curr_features], y_train)
        # predict values
        prediction_h = voting_clf_tuned_h.predict(X_test[curr_features])
        prediction_s = voting_clf_tuned_s.predict(X_test[curr_features])
        # calculate accuracy
        accuracy_h = accuracy_score(y_test, prediction_h)
        accuracy_s = accuracy_score(y_test, prediction_s)
        accuracy_dict['Voting Classifier Hard tuned '+str(curr_features)] = accuracy_h
        accuracy_dict['Voting Classifier Soft tuned '+str(curr_features)] = accuracy_s
        cross_val_dict['Voting Classifier Hard tuned '+str(curr_features)] = voting_clf_tuned_h.score(X_train[curr_features], y_train)
        cross_val_dict['Voting Classifier Soft tuned '+str(curr_features)] = voting_clf_tuned_s.score(X_train[curr_features], y_train)


    # sort cross_val_dict and accuracy_dict
    # cross_val_dict_sorted = dict(sorted(cross_val_dict.items(), key=lambda item: item[1], reverse=True))
    # accuracy_dict_sorted = dict(sorted(accuracy_dict.items(), key=lambda item: item[1], reverse=True))
    # print(cross_val_dict_sorted)
    # print(accuracy_dict_sorted)

    # # save the results to a file
    # results = {'cross_val_dict': cross_val_dict_sorted, 'accuracy_dict': accuracy_dict_sorted, 'best_estimators': best_estimators}
    # results_df = pd.DataFrame.from_dict(results)
    # results_df.to_csv('results_3.csv')
    # features_df = pd.DataFrame.from_dict(feature_importance_values_sorted)
    # features_df.to_csv('feature_importance_3.csv')

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # scale the data
    scaller = RobustScaler()
    X_train_scaled[all_usable_features] = scaller.fit_transform(X_train[all_usable_features])
    X_test_scaled[all_usable_features] = scaller.transform(X_test[all_usable_features])

    accuracy_for_combinations = {}
    cross_val_for_combinations = {}

    for index, i in enumerate(combinations_indexes_to_check):
        curr_combination = list(all_combinations[i])
        # do cross validation
        cv_scores = cross_val_score(xgb_clf, X_train_scaled[curr_combination], y_train, cv=10, n_jobs=-1,
                                    verbose=0).mean()
        cross_val_for_combinations[tuple(curr_combination)] = [cv_scores, 'xgb', 'scaled']
        xgb_clf.fit(X_train_scaled[curr_combination], y_train)
        prediction = xgb_clf.predict(X_test_scaled[curr_combination])
        # calculate accuracy
        accuracy = accuracy_score(y_test, prediction)
        accuracy_for_combinations[tuple(curr_combination)] = [accuracy, 'xgb', 'scaled']

        for feature, importance in zip(curr_combination, xgb_clf.feature_importances_):
            feature_importance_values[feature][0] += importance
            feature_importance_values[feature][1] += 1
            feature_importance_values[feature][2] += cv_scores

        # do cross validation
        cv_scores = cross_val_score(rfc_clf, X_train_scaled[curr_combination], y_train, cv=10, n_jobs=-1).mean()
        cross_val_for_combinations[tuple(curr_combination)] = [cv_scores, 'rfc', 'scaled']
        rfc_clf.fit(X_train_scaled[curr_combination], y_train)
        prediction = rfc_clf.predict(X_test_scaled[curr_combination])
        # calculate accuracy
        accuracy = accuracy_score(y_test, prediction)
        accuracy_for_combinations[tuple(curr_combination)] = [accuracy, 'rfc', 'scaled']
        print("Finished cobinations: ", index, " out of ", num_of_combinations_to_check)

    # sort the accuracy_for_combinations dictionary and cross_val_for_combinations dictionary
    accuracy_for_combinations_sorted = dict(
        sorted(accuracy_for_combinations.items(), key=lambda item: item[1][0], reverse=True))
    cross_val_for_combinations_sorted = dict(
        sorted(cross_val_for_combinations.items(), key=lambda item: item[1][0], reverse=True))

    # calculate mean feature importance values
    for feature in feature_importance_values:
        if feature_importance_values[feature][1] != 0:
            feature_importance_values[feature][0] /= feature_importance_values[feature][1]
            feature_importance_values[feature][2] /= feature_importance_values[feature][1]

    # sort the feature_importance_values dictionary
    feature_importance_values_sorted = dict(
        sorted(feature_importance_values.items(), key=lambda item: item[1][0], reverse=True))

    print(accuracy_for_combinations_sorted)
    print(cross_val_for_combinations_sorted)
    print(feature_importance_values_sorted)

    # # print cross validation scores and accuracy
    # print(cross_val_dict)
    # print(accuracy_dict)
    # #print(best_estimators)

    ### hyperparameter tuning for different combinations of features

    # create a list of combinations to check
    combinations_to_check = []
    for i, key in enumerate(cross_val_for_combinations_sorted.items()):
        if i >= 20:
            break
        combinations_to_check.append(list(key[0]))

    # create a vector of 5 most important features
    most_important_features = [feature for feature in feature_importance_values_sorted][:5]
    for i, key in enumerate(feature_importance_values_sorted):
        if i >= 5:
            combinations_to_check.append(list(most_important_features))
            most_important_features.append(key)
        if i >= 10:
            break

    for key, value in classifiers_for_tuning.items():
        if key == 'Voting Classifier Soft' or key == 'Voting Classifier Hard':
            continue
        # create a RandomizedSearchCV
        rscv = RandomizedSearchCV(value, param_distributions=param_grids[key], n_iter=50, cv=10, random_state=42,
                                  verbose=2,
                                  n_jobs=-1, scoring='accuracy')
        rscv.fit(X_train_scaled[all_usable_features], y_train)
        best_estimators[key + " scaled tuned " + str(all_usable_features)] = rscv.best_estimator_
        # predict values
        prediction = rscv.best_estimator_.predict(X_test_scaled[all_usable_features])
        # calculate accuracy
        accuracy = accuracy_score(y_test, prediction)
        accuracy_dict[key + " scaled tuned " + str(all_usable_features)] = accuracy
        cross_val_dict[key + " scaled tuned " + str(all_usable_features)] = rscv.best_score_
        print("Finished tuning for: ", key)

    # create a list of tuples of the best estimators for Voting Classifier
    best_estimators_list = [(key, value) for key, value in best_estimators.items() if key != 'Naive Bayes Classifier']

    # create a new Voting Classifier with best estimators
    voting_clf_tuned_h = VotingClassifier(estimators=best_estimators_list, voting='hard')
    voting_clf_tuned_s = VotingClassifier(estimators=best_estimators_list, voting='soft')

    # fit the Voting Classifiers
    voting_clf_tuned_h.fit(X_train_scaled[all_usable_features], y_train)
    voting_clf_tuned_s.fit(X_train_scaled[all_usable_features], y_train)
    # predict values
    prediction_h = voting_clf_tuned_h.predict(X_test_scaled[all_usable_features])
    prediction_s = voting_clf_tuned_s.predict(X_test_scaled[all_usable_features])
    # calculate accuracy
    accuracy_h = accuracy_score(y_test, prediction_h)
    accuracy_s = accuracy_score(y_test, prediction_s)
    accuracy_dict["Voting Classifier Hard scaled tuned " + str(all_usable_features)] = accuracy_h
    accuracy_dict["Voting Classifier Soft scaled tuned " + str(all_usable_features)] = accuracy_s
    cross_val_dict["Voting Classifier Hard scaled tuned " + str(all_usable_features)] = voting_clf_tuned_h.score(
        X_train_scaled[all_usable_features], y_train)
    cross_val_dict["Voting Classifier Soft scaled tuned " + str(all_usable_features)] = voting_clf_tuned_s.score(
        X_train_scaled[all_usable_features], y_train)

    # print cross validation scores and accuracy
    # print(cross_val_dict)
    # print(accuracy_dict)
    # print(best_estimators)

    # create a loop to check different combinations of features and use different models and tune their hyperparameters
    for features in combinations_to_check:
        curr_features = features
        print("Checking features: ", curr_features)
        for key, value in classifiers_for_tuning.items():
            if key == 'Voting Classifier Soft' or key == 'Voting Classifier Hard':
                continue
            # create a RandomizedSearchCV
            rscv = RandomizedSearchCV(value, param_distributions=param_grids[key], n_iter=50, cv=10, random_state=42,
                                      verbose=2,
                                      n_jobs=-1, scoring='accuracy')
            rscv.fit(X_train_scaled[curr_features], y_train)
            best_estimators[key + " scaled tuned " + str(curr_features)] = rscv.best_estimator_
            # predict values
            prediction = rscv.best_estimator_.predict(X_test_scaled[curr_features])
            # calculate accuracy
            accuracy = accuracy_score(y_test, prediction)
            accuracy_dict[key + " scaled tuned " + str(curr_features)] = accuracy
            cross_val_dict[key + " scaled tuned " + str(curr_features)] = rscv.best_score_
            print("Finished tuning for: ", key)

        # create a list of tuples of the best estimators for Voting Classifier
        best_estimators_list = [(key, value) for key, value in best_estimators.items() if
                                key != 'Naive Bayes Classifier']

        # create a new Voting Classifier with best estimators
        voting_clf_tuned_h = VotingClassifier(estimators=best_estimators_list, voting='hard')
        voting_clf_tuned_s = VotingClassifier(estimators=best_estimators_list, voting='soft')

        # fit the Voting Classifiers
        voting_clf_tuned_h.fit(X_train_scaled[curr_features], y_train)
        voting_clf_tuned_s.fit(X_train_scaled[curr_features], y_train)
        # predict values
        prediction_h = voting_clf_tuned_h.predict(X_test_scaled[curr_features])
        prediction_s = voting_clf_tuned_s.predict(X_test_scaled[curr_features])
        # calculate accuracy
        accuracy_h = accuracy_score(y_test, prediction_h)
        accuracy_s = accuracy_score(y_test, prediction_s)
        accuracy_dict['Voting Classifier Hard scaled tuned ' + str(curr_features)] = accuracy_h
        accuracy_dict['Voting Classifier Soft scaled tuned ' + str(curr_features)] = accuracy_s
        cross_val_dict['Voting Classifier Hard scaled tuned ' + str(curr_features)] = voting_clf_tuned_h.score(
            X_train_scaled[curr_features], y_train)
        cross_val_dict['Voting Classifier Soft scaled tuned ' + str(curr_features)] = voting_clf_tuned_s.score(
            X_train_scaled[curr_features], y_train)

    # sort cross_val_dict and accuracy_dict
    cross_val_dict_sorted = dict(sorted(cross_val_dict.items(), key=lambda item: item[1], reverse=True))
    accuracy_dict_sorted = dict(sorted(accuracy_dict.items(), key=lambda item: item[1], reverse=True))
    print(cross_val_dict_sorted)
    print(accuracy_dict_sorted)

    # save the results to a file
    results = {'cross_val_dict': cross_val_dict_sorted, 'accuracy_dict': accuracy_dict_sorted, 'best_estimators': best_estimators}
    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv('results.csv')
    features_df = pd.DataFrame.from_dict(feature_importance_values_sorted)
    features_df.to_csv('feature_importance.csv')

    # Write the dictionary to a JSON file
    file_path = "results.json"
    with open(file_path, "w") as json_file:
        json.dump(results, json_file)

if __name__ == "__main__":
    main()
    #killer_loop()