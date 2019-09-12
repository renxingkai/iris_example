import sys
from joblib import dump

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb


class Data:
    def __init__(self, X, y, test_split: float = 0.2, random_state: int = 1):
        assert 1 >= test_split > 0, "Value must be a greater than 0 and less than or equal to 1"
        self.__X = X
        self.__y = y
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(X,
                                                                                        y,
                                                                                        test_size=test_split,
                                                                                        random_state=random_state)

    def train_data(self):
        return self.__X_train, self.__y_train

    def holdout_data(self):
        return self.__X_test, self.__y_test

    def all_data(self):
        return self.__X, self.__y


def perform_cv_training(model, params, data: Data = None, cv=5):
    if data is None:
        raise Exception("No data!!!")
    clf = GridSearchCV(model, params, cv=cv)
    X, y = data.train_data()
    clf.fit(X, y)
    return clf


def print_report(model, data: Data, final_eval: bool = False) -> None:
    if final_eval:
        print('Final evaluation')
        X, y = data.all_data()
    else:
        X, y = data.holdout_data()
    pred = model.predict(X)
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred))


def main():
    # Prepare data
    df = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/iris.csv')
    y = df['Species']
    X = df.drop('Species', axis=1)
    data = Data(X, y, 0.1, 1)

    # Define models
    models = [('knn', KNeighborsClassifier(n_jobs=4)),
              ('xgb', xgb.XGBClassifier(objective="multi:softprob")),
              ('lr', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=500))
              ]
    parameter_space = [
        {'knn__n_neighbors': range(3, 7), 'knn__algorithm': ['ball_tree', 'kd_tree'],
         'xgb__eta': [0.01, 0.1, 0.3], 'xgb__max_depth': range(3, 9), 'xgb__subsample': [0.1, 0.2, 0.3, 0.4, 0.5]}
    ]

    # Define, train, and save ensemble classifier
    clf = perform_cv_training(VotingClassifier(estimators=models, voting='soft'), parameter_space, data)
    print_report(clf, data)
    print_report(clf, data, final_eval=True)
    print('Saving results of classification')
    df['prediction'] = clf.predict(X)
    df.to_csv('results.csv', index=False)
    print('Serializing the classifier')
    dump(clf, 'iris_predictor.joblib')
    return 0


if __name__ == '__main__':
    sys.exit(main())
