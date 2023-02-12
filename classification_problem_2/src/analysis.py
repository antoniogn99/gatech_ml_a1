import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from time import time

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.metrics import f1_score

from config import DATA_CSV_PATH


df = pd.read_csv(DATA_CSV_PATH)
x = df.drop(["churn"], axis=1).values
y = df["churn"].values


def tuning(estimator, param_grid, n_splits=5, n_repeats=2):
    gsearch = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0),
        verbose=0
    )
    gsearch.fit(x, y)
    return gsearch.best_params_, gsearch.best_score_

def find_best_hyperparameters():
    models = {
        "KNN": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {"n_neighbors": range(3, 21, 4)},
        },
        "DT": {
            "estimator": DecisionTreeClassifier(),
            "param_grid": {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [2, 4, 8, 16, 32, 64]
            }
        },
        "SVM": {
            "estimator": SVC(),
            "param_grid": {"kernel": ["linear", "poly", "rbf", "sigmoid"]},
        },
        "NN": {
            "estimator": MLPClassifier(random_state=0),
            "param_grid": {
                "hidden_layer_sizes": [(h, h) for h in [16, 32, 64]],
                "activation": ["logistic", "tanh", "relu"],
            },
        },
        "GBC": {
            "estimator": GradientBoostingClassifier(),
            "param_grid": {
                "n_estimators": [32, 64, 128, 256, 512],
                "max_depth": [3, 5, 7]
            },
        },
    }
    for name in ["NN"]:
        print("Tunning", name)
        estimator = models[name]["estimator"]
        param_grid = models[name]["param_grid"]
        best_params, best_score = tuning(estimator, param_grid)
        print("Best params:", best_params)
        print("Best score:", best_score)

def comparison():
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=11),
        "DT": DecisionTreeClassifier(criterion="gini", max_depth=4),
        "SVM": SVC(kernel="linear"),
        "NN": MLPClassifier(random_state=0, activation="logistic", hidden_layer_sizes=(32, 32)),
        "GBC": GradientBoostingClassifier(n_estimators=64, max_depth=3)
    }

    common_params = {
        "X": x,
        "y": y,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=5, test_size=0.2, random_state=0),
        "score_type": "both",
        "n_jobs": 1,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "f1_weighted",
    }

    for name in models.keys():
        estimator = models[name]
        t1 = time()
        LearningCurveDisplay.from_estimator(estimator, **common_params)
        t2 = time()
        print(name, t2-t1)
        plt.title(name)
        plt.ylabel("Weighted F1 Score")
        plt.xlim([0.8*0.1*len(y), 0.8*len(y)])
        plt.legend(labels=["Training", "Test"])
        plt.savefig("learning_curves_{}.png".format(name))

def plot_training_nn():
    train_scores = []
    test_scores = []
    max_iter_list = [4, 8, 16, 32, 64, 128]
    for max_iter in max_iter_list:
        model  = MLPClassifier(hidden_layer_sizes=(32, 32,), random_state=0, activation="logistic")
        _, test_score = tuning(model, {"max_iter": [max_iter]}, n_splits=5, n_repeats=1)
        test_scores.append(test_score)
        model  = MLPClassifier(hidden_layer_sizes=(32, 32,), random_state=0, activation="logistic", max_iter=max_iter)
        model.fit(x, y)
        train_scores.append(f1_score(y, model.predict(x), average="weighted"))
    
    plt.plot(max_iter_list, train_scores, "o-", label="Training")
    plt.plot(max_iter_list, test_scores, "o-", label="Test")
    plt.xticks(max_iter_list)
    plt.xlabel("Number of iterations")
    plt.ylabel("Weighted F1 Score")
    plt.legend()
    plt.title("NN")
    plt.savefig("nn_training.png")
    plt.clf()

def plot_training_gbc():
    train_scores = []
    test_scores = []
    n_estimators_list = [4, 8, 16, 32, 64, 128]
    for n_estimators in n_estimators_list:
        model  = GradientBoostingClassifier(max_depth=3)
        _, test_score = tuning(model, {"n_estimators": [n_estimators]}, n_splits=5, n_repeats=1)
        test_scores.append(test_score)
        model  = GradientBoostingClassifier(max_depth=3, n_estimators=n_estimators)
        model.fit(x, y)
        train_scores.append(f1_score(y, model.predict(x), average="weighted"))
    
    plt.plot(n_estimators_list, train_scores, "o-", label="Training")
    plt.plot(n_estimators_list, test_scores, "o-", label="Test")
    plt.xticks(n_estimators_list)
    plt.xlabel("Number of iterations")
    plt.ylabel("Weighted F1 Score")
    plt.legend()
    plt.title("GBC")
    plt.savefig("nn_gbc.png")
    plt.clf()

def plot_comparison():
    names = ["KNN", "DT", "SVM", "NN", "GBC"]
    scores = [0.765, 0.781, 0.771, 0.787, 0.795]
    times = [8.3, 0.42, 10**6, 30.45, 12.14]
    denominator = 60
    times = [t/denominator for t in times]
    x = np.array(range(len(names)))
    width = 0.2
    plt.bar(x+width/2, scores, width=width, label="Weighted F1 Score")
    for i, v in enumerate(scores):
        plt.text(i, v, "{:.2f}".format(v))
    plt.bar(x-width/2, times, width=width, label="Time to fit (mins)")
    plt.xticks(x, names)
    plt.ylim([0, 1.3])
    plt.legend()
    plt.savefig("comparison.png")

plot_comparison()
    
