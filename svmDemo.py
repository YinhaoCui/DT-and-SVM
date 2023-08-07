import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn.svm import SVC

from svm import SVM

def confusion_matrix_plot_matplotlib(y_truth, y_predict, cmap=plt.cm.Blues):

    cm = confusion_matrix(y_truth, y_predict)
    plt.matshow(cm, cmap=cmap)
    plt.colorbar()

    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

data = pd.read_csv("animals.csv")
diag_map = {'DEER': 1.0, 'CATTLE': -1.0}
data['class'] = data['class'].map(diag_map)

label = data.loc[:, 'class']
feature = data.drop("class", axis=1)

# le1 = LabelEncoder()
# label = le1.fit_transform(label)

svm = SVM()

feature = MinMaxScaler().fit_transform(feature.values)

feature = pd.DataFrame(feature)
label = pd.DataFrame(label)

X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=42)

svm.fit(X_train.to_numpy(), y_train.to_numpy())

predictions = svm.predict(X_test)

print("accuracy on self implemented SVM classifier: " + str(accuracy_score(y_test, predictions)))

svm = SVC()
svm.fit(X_train.to_numpy(), y_train.to_numpy())

predictions = svm.predict(X_test)

print("accuracy on library SVM classifier: " + str(accuracy_score(y_test, predictions)))



