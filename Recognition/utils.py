import numpy as np
from matplotlib import pyplot as plt


def normalize(array, maxValue=1):
    min_ = array.min()
    max_ = array.max()
    range_ = max_ - min_
    img = (array - min_) / range_
    return img * maxValue


def show_bar(x_data, title):
    num = 10
    num_bin = np.linspace(0, 1, num)
    plt.figure()
    plt.hist(x_data, num_bin)
    plt.xticks(num_bin)
    plt.xlabel(f'{title}')
    plt.ylabel('num')
    plt.title(f'{title}')
    plt.grid(alpha=0.4)
    plt.savefig(rf'../visualization/{title}.png')
    plt.close()