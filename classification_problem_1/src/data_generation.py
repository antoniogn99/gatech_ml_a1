import random

import pandas as pd


def generate_random_points(num_points):
    x1 = [random.random()*6 for _ in range(num_points)]
    x2 = [random.random()*2 for _ in range(num_points)]
    points = [(x1[i], x2[i]) for i in range(num_points)]
    return points

def generate_data(num_points):
    points = generate_random_points(num_points)
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

def generate_train_test_data(train_size=5000, test_size=1000, num_erros=100):
    points, groups = generate_data(train_size)
    for _ in range(num_erros):
        i = random.randint(0, train_size-1)
        groups[i] = 1-groups[i]
    df_train = pd.DataFrame(points, columns=["X1", "X2"])
    df_train["Y"] = groups
    df_train.to_csv("c1_train.csv", index=False)
    points, groups = generate_data(test_size)
    df_test = pd.DataFrame(points, columns=["X1", "X2"])
    df_test["Y"] = groups
    df_test.to_csv("c1_test.csv", index=False)


if __name__ == "__main__":
    from analysis import plot_points
    import matplotlib.pyplot as plt
    points, groups = generate_data(10**6)
    plot_points(points, groups)
    plt.title("Target function")
    plt.savefig("target_function.png")
