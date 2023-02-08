import matplotlib.pyplot as plt
import random


def generate_data():
    num_points = 10000
    x1 = [random.random()*2 for _ in range(num_points)]
    x2 = [random.random()*6 for _ in range(num_points)]
    points = [(x1[i], x2[i]) for i in range(num_points)]
    groups = []
    for p in points:
        center = (1, int(p[1])+0.5)
        if int(p[1]) % 2 == 0:
            if (p[0]-center[0])**2 + (p[1]-center[1])**2 < 0.25**2:
                groups.append(1)
            elif p[0] < 1 or (p[0]-center[0])**2 + (p[1]-center[1])**2 < 0.5**2:
                groups.append(0)
            else:
                groups.append(1)
        else:
            if (p[0]-center[0])**2 + (p[1]-center[1])**2 < 0.25**2:
                groups.append(0)
            elif p[0] > 1 or (p[0]-center[0])**2 + (p[1]-center[1])**2 < 0.5**2:
                groups.append(1)
            else:
                groups.append(0)
    plot_points(points, groups)



def plot_points(points, groups):
    for point, group in zip(points, groups):
        color = "tab:blue" if group == 1 else "tab:orange"
        plt.plot([point[0]], [point[1]], ".", color=color)
    plt.show()

generate_data()