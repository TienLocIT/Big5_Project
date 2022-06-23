import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
col_names = ['AgeGroup', 'Gender', 'Location', 'Question', 'ScoreOfOpenness', 'ScoreOfConscientiousness',
             'ScoreOfExtraversion', 'ScoreOfAgreeableness', 'ScoreOfNeuroticism', 'Outcome']
pima = pd.read_csv("data_gb5.csv", header=None, names=col_names)
pima.head()

# print(pima)
feature_cols = ['AgeGroup', 'Gender', 'Location', 'Question', 'ScoreOfOpenness',
                'ScoreOfConscientiousness', 'ScoreOfExtraversion', 'ScoreOfAgreeableness', 'ScoreOfNeuroticism']
X = pima[feature_cols]  # Features
y = pima.Outcome  # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)  # 70% training and 30% test


def run_model(X_train, y_train, X_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    score = clf.predict(X_test)

    return score
