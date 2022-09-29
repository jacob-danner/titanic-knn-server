import pandas as pd
from sklearn.neighbors import *

def make_knn():

    train = pd.read_csv('./data/train.csv')

    rename = {
        'PassengerId': 'id',
        'Survived': 'survived',
        'Pclass': 'pass_class',
        'Name': 'name',
        'Sex': 'sex',
        'Age': 'age',
        'SibSp': 'sibsNspouses',
        'Parch': 'parentsNchildren',
        'Ticket': 'ticketnum',
        'Fare': 'fare',
        'Cabin': 'cabin',
        'Embarked': 'embarked'
    }

    train = train.rename(columns=rename)

    feature_cols = ['pass_class', 'sex', 'age', 'sibsNspouses', 'parentsNchildren', 'fare']
    features = train[feature_cols]

    sex_normal = features['sex'].apply(
        lambda x: 1 if (x == 'male') else 0
    )

    features['sex'] = sex_normal

    features['age'] = features['age'].fillna(-1)

    features['fare'] = features['fare'].fillna(-1)

    answers = train['survived']

    knn = KNeighborsClassifier()
    knn.fit(features, answers)

    return knn