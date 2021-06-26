import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def train_eval(clf, X_train, y_train):
    n_folds = 10
    print(f"{n_folds}-fold cross-validation")
    scores = cross_validate(
        clf, X_train, y_train, cv=n_folds, scoring=('precision', 'recall', 'accuracy', 'f1')
    )
    print(f"CV Precision {sum(scores['test_precision']/n_folds)}")
    print(f"CV Recall {sum(scores['test_recall']/n_folds)}")
    print(f"CV Accuracy {sum(scores['test_accuracy']/n_folds)}")
    print(f"CV F1-measure {sum(scores['test_f1']/n_folds)}")
    print()


def test_eval(classifier, clf, X_test, y_test, y_pred):
    print(f"Test Precision {precision_score(y_test, y_pred, zero_division=0)}")
    print(f"Test Recall {recall_score(y_test, y_pred)}")
    print(f"Test Accuracy {accuracy_score(y_test, y_pred)}")
    print(f"Test F1-measure {f1_score(y_test, y_pred)}")
    plot_confusion_matrix(clf, X_test, y_test, values_format='d')
    plt.savefig(f"files/{classifier}.png")


# carregar noticias
dataset = pd.read_csv(open(f"datasets/features.tsv"), sep=",", header=0)
X, y = dataset.iloc[:, 6:], dataset["fake"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

classifiers = {
    "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=9),
    "RandomForestClassifier": RandomForestClassifier(),
    "SVC": SVC(),
}

for classifier, clf in classifiers.items():
    print(f"Classificador: {classifier}")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    train_eval(clf, X_train, y_train)
    test_eval(classifier, clf, X_test, y_test, y_pred)
