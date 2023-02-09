import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from config import TRAIN_CSV_PATH, TEST_CSV_PATH
from data_generation import generate_random_points


df_train = pd.read_csv(TRAIN_CSV_PATH)
x_train = df_train.drop(["Y"], axis=1).values
y_train = df_train["Y"].values

df_test = pd.read_csv(TEST_CSV_PATH)
x_test = df_test.drop(["Y"], axis=1).values
y_test = df_test["Y"].values

def plot_points(points, groups):
    COLOR0 = "tab:orange"
    COLOR1 = "tab:blue"
    for point, group in zip(points, groups):
        color = COLOR1 if group == 1 else COLOR0
        plt.plot(point[0], point[1], marker=".", markersize=1, color=color)
    plt.xlim([0, 6])
    plt.ylim([0, 2])
    plt.xlabel(r"$X_1$")
    plt.ylabel(r"$X_2$")
    plt.scatter(-1, -1, marker="o", label="Y=1", color=COLOR1)
    plt.scatter(-1, -1, marker="o", label="Y=0", color=COLOR0)
    plt.legend(bbox_to_anchor=(1, 1), loc='lower right')
    plt.tight_layout()

def train_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred, accuracy_score(y_test, y_pred)

def plot_model(model, num_points=10**5):
    points = generate_random_points(num_points)
    groups = model.predict(points)
    plot_points(points, groups)

def analysis_knn():
    ns = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    ks = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    acc_matrix = []
    for n in ns:
        accs = []
        for k in ks:
            model = KNeighborsClassifier(n_neighbors=k)
            _, acc = train_model(model, x_train[:n], y_train[:n], x_test, y_test)
            accs.append(float(acc))
        acc_matrix.append(accs)
    acc_matrix = np.array(acc_matrix)
    sns.heatmap(acc_matrix, xticklabels=ks, yticklabels=ns, annot=True, cmap="Blues")
    plt.xlabel("Value of K")
    plt.ylabel("Number of training examples")
    plt.title("Accuracy with KNN")
    plt.savefig("knn_analysis.png")
    plt.clf()

    for k in [1, 9, 21]:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        plot_model(model)
        plt.title("KNN with k={}".format(k))
        plt.savefig("knn_k_{}.png".format(k))
        plt.clf()

def analysis_dt():
    ns = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    ds = [1, 3, 5, 7, 10, 15, 20, 40, 80, 160]
    acc_matrix = []
    for n in ns:
        accs = []
        for d in ds:
            model = DecisionTreeClassifier(max_depth=d, random_state=0)
            _, acc = train_model(model, x_train[:n], y_train[:n], x_test, y_test)
            accs.append(float(acc))
        acc_matrix.append(accs)
    acc_matrix = np.array(acc_matrix)
    sns.heatmap(acc_matrix, xticklabels=ds, yticklabels=ns, annot=True, cmap="Blues")
    plt.xlabel("Maximum depth")
    plt.ylabel("Number of training examples")
    plt.title("Accuracy with Decision Tree")
    plt.savefig("dt_analysis.png")
    plt.clf()

    for d in ds:
        model = DecisionTreeClassifier(max_depth=d, random_state=0)
        model.fit(x_train, y_train)
        plot_model(model)
        plt.title("Decision Tree with max_depth={}".format(d))
        plt.savefig("dt_d_{}.png".format(d))
        plt.clf()

def analysis_svm():
    ns = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    ds = [1, 2, 3, 4, 5]
    acc_matrix = []
    for n in ns:
        accs = []
        for d in ds:
            model = SVC(kernel="poly", degree=d)
            _, acc = train_model(model, x_train[:n], y_train[:n], x_test, y_test)
            accs.append(float(acc))
        acc_matrix.append(accs)
    acc_matrix = np.array(acc_matrix)
    sns.heatmap(acc_matrix, xticklabels=ds, yticklabels=ns, annot=True, cmap="Blues")
    plt.xlabel("Degree of the polynomial kernel function")
    plt.ylabel("Number of training examples")
    plt.title("Accuracy with SVM (polynomial kernel)")
    plt.savefig("svm_poly_analysis.png")
    plt.clf()

    ks = ["linear", "poly", "rbf", "sigmoid"]
    acc_matrix = []
    for n in ns:
        accs = []
        for k in ks:
            model = SVC(kernel=k)
            _, acc = train_model(model, x_train[:n], y_train[:n], x_test, y_test)
            accs.append(float(acc))
        acc_matrix.append(accs)
    acc_matrix = np.array(acc_matrix)
    sns.heatmap(acc_matrix, xticklabels=ks, yticklabels=ns, annot=True, cmap="Blues")
    plt.xlabel("Kernel")
    plt.ylabel("Number of training examples")
    plt.title("Accuracy with SVM")
    plt.savefig("svm_analysis.png")
    plt.clf()

    for k in ks:
        model = SVC(kernel=k)
        model.fit(x_train, y_train)
        plot_model(model)
        plt.title("SVM with kernel={}".format(k))
        plt.savefig("svm_k_{}.png".format(k))
        plt.clf()

analysis_svm()