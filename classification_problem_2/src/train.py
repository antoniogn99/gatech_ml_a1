import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from config import DATA_CSV_PATH


df = pd.read_csv(DATA_CSV_PATH)
x = df.drop(["status"], axis=1).values
y = df["status"].values


def tuning(estimator, param_grid, n_splits=5, n_repeats=2):
    gsearch = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
    )
    gsearch.fit(x, y)
    return gsearch.best_params_, gsearch.best_score_

def main():
    models = {
        "KNN": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {"n_neighbors": range(3, 21, 8)},
        },
        "DT": {
            "estimator": DecisionTreeClassifier(),
            "param_grid": {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [10, 100]
            }
        }
    }
    for name in models.keys():
        print("Tunning", name)
        estimator = models[name]["estimator"]
        param_grid = models[name]["param_grid"]
        best_params, best_score = tuning(estimator, param_grid)
        print("Best params:", best_params)
        print("Best score:", best_score)

main()
