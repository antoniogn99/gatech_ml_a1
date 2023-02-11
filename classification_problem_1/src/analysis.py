import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    return accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)

def plot_model(model, num_points=10**5):
    points = generate_random_points(num_points)
    groups = model.predict(points)
    plot_points(points, groups)

def analysis_knn():
    ns = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    ks = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    acc_matrix = []
    train_acc_matrix = []
    for n in ns:
        accs = []
        train_accs = []
        for k in ks:
            model = KNeighborsClassifier(n_neighbors=k)
            train_acc, acc = train_model(model, x_train[-n:], y_train[-n:], x_test, y_test)
            accs.append(float(acc))
            train_accs.append(train_acc)
        acc_matrix.append(accs)
        train_acc_matrix.append(train_accs)

    acc_matrix = np.array(acc_matrix)
    sns.heatmap(acc_matrix, xticklabels=ks, yticklabels=ns, annot=True, cmap="Blues")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Number of training examples")
    plt.title("Test accuracy with KNN")
    plt.savefig("knn_analysis.png")
    plt.clf()

    train_acc_matrix = np.array(train_acc_matrix)
    sns.heatmap(train_acc_matrix, xticklabels=ks, yticklabels=ns, annot=True, cmap="Blues")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Number of training examples")
    plt.title("Train accuracy with KNN")
    plt.savefig("knn_analysis_train.png")
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
    train_acc_matrix = []
    for n in ns:
        accs = []
        train_accs = []
        for d in ds:
            model = DecisionTreeClassifier(max_depth=d, random_state=0)
            train_acc, acc = train_model(model, x_train[-n:], y_train[-n:], x_test, y_test)
            accs.append(float(acc))
            train_accs.append(train_acc)
        acc_matrix.append(accs)
        train_acc_matrix.append(train_accs)
    
    acc_matrix = np.array(acc_matrix)
    sns.heatmap(acc_matrix, xticklabels=ds, yticklabels=ns, annot=True, cmap="Blues")
    plt.xlabel("Maximum depth")
    plt.ylabel("Number of training examples")
    plt.title("Test accuracy with Decision Tree")
    plt.savefig("dt_analysis.png")
    plt.clf()

    train_acc_matrix = np.array(train_acc_matrix)
    sns.heatmap(train_acc_matrix, xticklabels=ds, yticklabels=ns, annot=True, cmap="Blues")
    plt.xlabel("Maximum depth")
    plt.ylabel("Number of training examples")
    plt.title("Train accuracy with Decision Tree")
    plt.savefig("dt_analysis_train.png")
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
    train_acc_matrix = []
    for n in ns:
        accs = []
        train_accs = []
        for k in ks:
            model = SVC(kernel=k)
            train_acc, acc = train_model(model, x_train[-n:], y_train[-n:], x_test, y_test)
            accs.append(float(acc))
            train_accs.append(train_acc)
        acc_matrix.append(accs)
        train_acc_matrix.append(train_accs)

    acc_matrix = np.array(acc_matrix)
    sns.heatmap(acc_matrix, xticklabels=ks, yticklabels=ns, annot=True, cmap="Blues")
    plt.xlabel("Kernel")
    plt.ylabel("Number of training examples")
    plt.title("Test accuracy with SVM")
    plt.savefig("svm_analysis.png")
    plt.clf()

    train_acc_matrix = np.array(train_acc_matrix)
    sns.heatmap(train_acc_matrix, xticklabels=ks, yticklabels=ns, annot=True, cmap="Blues")
    plt.xlabel("Kernel")
    plt.ylabel("Number of training examples")
    plt.title("Train accuracy with SVM")
    plt.savefig("svm_analysis_train.png")
    plt.clf()

    for k in ks:
        model = SVC(kernel=k)
        model.fit(x_train, y_train)
        plot_model(model)
        plt.title("SVM with kernel={}".format(k))
        plt.savefig("svm_k_{}.png".format(k))
        plt.clf()

def analysis_nn():
    ns = [2000, 3000, 4000, 5000]
    hs = [8, 16, 32, 64, 128]
    acc_matrix = []
    train_acc_matrix = []
    for n in ns:
        accs = []
        train_accs = []
        for h in hs:
            model = MLPClassifier(hidden_layer_sizes=(h, h, ), max_iter=10**5, random_state=0)
            train_acc, acc = train_model(model, x_train[-n:], y_train[-n:], x_test, y_test)
            accs.append(float(acc))
            train_accs.append(train_acc)
        acc_matrix.append(accs)
        train_acc_matrix.append(train_accs)
    acc_matrix = np.array(acc_matrix)
    sns.heatmap(acc_matrix, xticklabels=hs, yticklabels=ns, annot=True, cmap="Blues")
    plt.xlabel("Hidden layers' size")
    plt.ylabel("Number of training examples")
    plt.title("Test accuracy with Neural Network with 2 layers")
    plt.savefig("nn_analysis.png")
    plt.clf()

    train_acc_matrix = np.array(train_acc_matrix)
    sns.heatmap(train_acc_matrix, xticklabels=hs, yticklabels=ns, annot=True, cmap="Blues")
    plt.xlabel("Hidden layers' size")
    plt.ylabel("Number of training examples")
    plt.title("Train accuracy with Neural Network with 2 layers")
    plt.savefig("nn_analysis_train.png")
    plt.clf()

    for h in hs:
        model = MLPClassifier(hidden_layer_sizes=(h, h, ), max_iter=10**5, random_state=0)
        model.fit(x_train, y_train)
        plot_model(model)
        plt.title("Neural Network with 2 layers of size {}".format(h))
        plt.savefig("nn_h_{}.png".format(h))
        plt.clf()

# Source: https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-mlpclassifier
def analysis_nn_2():
    mlp = MLPClassifier(hidden_layer_sizes=(128, 128,), max_iter=10**5, random_state=0)
    num_samples = x_train.shape[0]
    num_epochs = 4000
    batch_size = 500
    classes = np.unique(y_train)
    iterations_per_print = 10

    epochs = []
    scores_train = []
    scores_test = []
    for epoch in range(num_epochs):
        random_perm = np.random.permutation(num_samples)
        mini_batch_index = 0
        while True:
            indices = random_perm[mini_batch_index: mini_batch_index + batch_size]
            mlp.partial_fit(x_train[indices], y_train[indices], classes=classes)
            mini_batch_index += batch_size
            if mini_batch_index >= num_samples:
                break

        if epoch % iterations_per_print == 0:
            train_score = mlp.score(x_train, y_train)
            test_score = mlp.score(x_test, y_test)
            scores_train.append(train_score)
            scores_test.append(test_score)
            epochs.append(epoch)
            print("Iteration {}. Acc in Train: {}. Acc in Test: {}".format(epoch, train_score, test_score))

    plt.plot(epochs, scores_train, label="Test")
    plt.plot(epochs, scores_test, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Neural Network with 2 layers of size 128")
    plt.savefig("nn_analysis_2.png")
    plt.clf()

    plot_model(mlp)
    plt.title("Neural Network with 2 layers of size 256 (4000 epochs)")
    plt.savefig("nn_h_256_long.png")
    plt.clf()

    print(accuracy_score(y_test, mlp.predict(x_test)))

def analysis_gbc():
    ns = [2000, 3000, 4000, 5000]
    ks = [32, 64, 128, 256, 512, 1024, 2048, 5096, 10192]
    acc_matrix = []
    train_acc_matrix = []
    for n in ns:
        accs = []
        train_accs = []
        for k in ks:
            model = GradientBoostingClassifier(n_estimators=k)
            train_acc, acc = train_model(model, x_train[-n:], y_train[-n:], x_test, y_test)
            accs.append(float(acc))
            train_accs.append(train_acc)
        acc_matrix.append(accs)
        train_acc_matrix.append(train_accs)
    acc_matrix = np.array(acc_matrix)
    sns.heatmap(acc_matrix, xticklabels=ks, yticklabels=ns, annot=True, cmap="Blues")
    plt.xlabel("Number of trees")
    plt.ylabel("Number of training examples")
    plt.title("Test accuracy with Gradient Boosting Classifier")
    plt.savefig("gbc_analysis.png")
    plt.clf()

    train_acc_matrix = np.array(train_acc_matrix)
    sns.heatmap(train_acc_matrix, xticklabels=ks, yticklabels=ns, annot=True, cmap="Blues")
    plt.xlabel("Number of trees")
    plt.ylabel("Number of training examples")
    plt.title("Train accuracy with Gradient Boosting Classifier")
    plt.savefig("gbc_analysis_train.png")
    plt.clf()

    for k in ks:
        model = GradientBoostingClassifier(n_estimators=k)
        model.fit(x_train, y_train)
        plot_model(model)
        plt.title("Gradient Boosting Classifier with {} trees".format(k))
        plt.savefig("gbc_k_{}.png".format(k))
        plt.clf()

def analysis_gbc_2():
    n_estimators = [32, 64, 128, 256, 512, 1024, 2048]
    scores_train = []
    scores_test = []
    for n in n_estimators:
        model = GradientBoostingClassifier(n_estimators=n)
        model.fit(x_train, y_train)
        scores_train.append(accuracy_score(y_train, model.predict(x_train)))
        scores_test.append(accuracy_score(y_test, model.predict(x_test)))

    plt.plot(n_estimators, scores_train, label="Train")
    plt.plot(n_estimators, scores_test, label="Test")
    plt.xlabel("Iteration")
    plt.xticks(n_estimators, ["", "", "", 256, 512, 1024, 2048])
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Gradient Boosting Classifier")
    plt.savefig("gbc_analysis_2.png")
    plt.clf()

def comparison():
    models={
        "DT": DecisionTreeClassifier(max_depth=40),
        "KNN": KNeighborsClassifier(n_neighbors=9),
        "SVM": SVC(kernel="rbf"),
        "NN": MLPClassifier(hidden_layer_sizes=(128, 128), batch_size=500, learning_rate_init=0.01, max_iter=10**8),
        "GB": GradientBoostingClassifier(n_estimators=2048)
    }
    model_names = list(models.keys())
    scores_train = []
    scores_test = []
    for name in model_names:
        print("Training", name)
        model = models[name]
        model.fit(x_train, y_train)
        scores_train.append(accuracy_score(y_train, model.predict(x_train)))
        scores_test.append(accuracy_score(y_test, model.predict(x_test)))
    
    x = np.array(range(len(model_names)))
    width = 0.2
    plt.bar(x-width/2, scores_train, width=width, label="Train accuracy")
    plt.bar(x+width/2, scores_test, width=width, label="Test accuracy")
    for i, v in enumerate(scores_test):
        plt.text(i, v, "{:.2f}".format(v))
    for i, v in enumerate(scores_train):
        plt.text(i-0.3, v, "{:.2f}".format(v))
    plt.xticks(x, model_names)
    plt.ylim([0, 1.3])
    plt.legend()
    plt.savefig("comparison.png")
