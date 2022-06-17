import numpy as np
from dataloder import getAnnotation


def calculateDcol(cols):
    num = len(cols)
    D_col = np.zeros(num - 1)
    for i in range(num - 1):
        D_col[i] = cols[i + 1] - cols[i]
    return D_col


def calculateLength(cols):
    num = len(cols)
    length = np.zeros(num)
    for i in range(num):
        length[i] = cols[i][1] - cols[i][0]
    return length


def delete_(length):
    num = len(length)
    mean_ = length.mean()
    std_ = length.std()
    for i in range(num):
        if length[i] < (mean_ - std_ * 3) or length[i] > (mean_ + std_ * 3):
            np.delete(length, i)
    return length


def _median(cols):
    X = [item[0] for item in cols]
    spacing = np.zeros(len(X) - 1)
    for i in range(len(X) - 1):
        spacing[i] = X[i + 1] - X[i]
    spacing_med = int(np.median(spacing))
    return spacing_med


def estimateNum(cols):
    spacing_med = _median(cols)
    num = round((cols[-1][0] - cols[0][0]) / spacing_med) + 1
    return num


def correctionCol(cols, num):
    # W = np.zeros(len(cols))
    # for i in range(len(cols)):
    #     W[i] = cols[i][1] - cols[i][0]
    # W_med = W.mean()

    spacing_med = _median(cols)
    new_cols = [cols[0]]
    for i in range(num - 1):
        item = (int(new_cols[i][0] + spacing_med), int(new_cols[i][1] + spacing_med))
        new_cols.append(item)
    return new_cols