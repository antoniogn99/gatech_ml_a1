import matplotlib.pyplot as plt
import random

import pandas as pd


def generate_random_points(num_points=5000):
    x1 = [random.random()*6 for _ in range(num_points)]
    x2 = [random.random()*2 for _ in range(num_points)]
    points = [(x1[i], x2[i]) for i in range(num_points)]
    groups = []
    for p in points:
        center = (int(p[0])+0.5, 1)
        if int(p[0]) % 2 == 0:
            if (p[0]-center[0])**2 + (p[1]-center[1])**2 < 0.25**2:
                groups.append(1)
            elif p[1] < 1 or (p[0]-center[0])**2 + (p[1]-center[1])**2 < 0.5**2:
                groups.append(0)
            else:
                groups.append(1)
        else:
            if (p[0]-center[0])**2 + (p[1]-center[1])**2 < 0.25**2:
                groups.append(0)
            elif p[1] > 1 or (p[0]-center[0])**2 + (p[1]-center[1])**2 < 0.5**2:
                groups.append(1)
            else:
                groups.append(0)
    return points, groups


def generate_train_test_data():
    points, groups = generate_random_points(5000)
    df_train = pd.DataFrame(points, columns=["X1", "X2"])
    df_train["Y"] = groups
    df_train.to_csv("c1_train.csv")
    points, groups = generate_random_points(1000)
    df_test = pd.DataFrame(points, columns=["X1", "X2"])
    df_test["Y"] = groups
    df_test.to_csv("c1_test.csv")



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
    plt.show()

generate_train_test_data()