from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, KFold
import pandas as pd
from DecisionTree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(name, ml_tool, X, y, cv):
    train_sizes, train_scores, test_scores = learning_curve(
        ml_tool, X=X, y=y, cv=cv, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 8))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Learning Curve of " + name)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


data = pd.read_csv("animals.csv")
diag_map = {'DEER': 1.0, 'CATTLE': -1.0}
data['class'] = data['class'].map(diag_map)

Y = data.loc[:, 'class']
X = data.drop("class", axis=1)

print(X.head())

score = 0
for i in range(10):

    dt = DecisionTreeClassifier(max_depth=20)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    m = dt.fit(X_train.to_numpy(), y_train.to_numpy())
    print(m)
    predictions = dt.predict(X_test.to_numpy())
    score = score + accuracy_score(y_test, predictions)

print("The accuracy is: ", score / 10.0)

from sklearn.tree import DecisionTreeClassifier as DTClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

dt = DTClassifier()
X = MinMaxScaler().fit_transform(X.values)

X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

dt.fit(X_train.to_numpy(), y_train.to_numpy())
predictions = dt.predict(X_test)

print("accuracy on library dt classifier: " + str(accuracy_score(y_test, predictions)))
plot_learning_curve("Learning Curve", dt, X, Y, KFold())