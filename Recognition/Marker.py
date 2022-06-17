import cv2
import numpy as np
from matplotlib import pyplot as plt


Agarose1_param = [0.168, 0.139, 0.105, 0.146, 0.089, 0.117, 0.146, 0.09]
Agarose2_param = [0.062, 0.081, 0.075, 0.138, 0.086, 0.144, 0.216, 0.198]
DL10000_param = [0.066, 0.158, 0.218, 0.224, 0.197, 0.137]


class Marker:
    def __init__(self, origin_tape, binary_tape):
        self.origin_tape = origin_tape
        self.binary_tape = binary_tape
        self.iscomplete = False
        self.labels, self.stats, self.num, self.size, self.Y = self._getComponents()
        self.Yd = self._getYd()
        self.By, self.By_index = self._getBrightest()
        self.Mtype = self.judge()
        if not self.iscomplete and self.Mtype:
            self.new_Y = self.getNewY()
            self.new_binary = self._padding()

    def _getComponents(self):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(self.binary_tape, connectivity=8)
        stats = stats[1:]
        num = len(stats)
        W = int(stats[:, 2].mean())
        H = int(stats[:, 3].mean())
        Y = stats[:, 1] + stats[:, 3]
        return labels, stats, num, (W, H), Y

    def _getBrightest(self):
        brightness = np.zeros(self.num)
        for l in range(len(self.stats)):
            brightness[l] = self.origin_tape[self.labels == l + 1].mean()
        brightness = brightness * self.num / brightness.sum()
        b_index = brightness.argmax()

        # DL15000
        # if brightness[b_index] > 1.5:
        #     return self.Y[b_index]
        # else:
        #     return False

        return self.Y[b_index], b_index

    def _getYd(self):
        num = len(self.Y) - 1
        Yd = []
        for i in range(num):
            Yd.append(self.Y[i + 1] - self.Y[i])
        Yd = np.array(Yd) / np.array(Yd).sum()
        return Yd

    def TAE(self, array, list_):
        tae = 0
        for i in range(len(list_)):
            tae += np.abs(list_[i] - array[i])
        return tae

    def judgebybri(self):
        tae1 = self.TAE(self.Yd[:self.By_index] * (self.num - 1), [i * 8 for i in Agarose1_param[int(4 - self.By_index):4]]) + \
               self.TAE(self.Yd[self.By_index:] * (self.num - 1), [i * 8 for i in Agarose1_param[4:int(self.num - self.By_index + 3)]])
        tae2 = self.TAE(self.Yd[:self.By_index] * (self.num - 1), [i * 8 for i in Agarose2_param[int(4 - self.By_index):4]]) + \
               self.TAE(self.Yd[self.By_index:] * (self.num - 1), [i * 8 for i in Agarose2_param[4:int(self.num - self.By_index + 3)]])
        tae3 = self.TAE(self.Yd[:self.By_index] * (self.num - 1), [i * 6 for i in DL10000_param[int(3 - self.By_index):3]]) + \
               self.TAE(self.Yd[self.By_index:] * (self.num - 1), [i * 6 for i in DL10000_param[3:int(self.num - self.By_index + 2)]])
        M = ['Agarose1', 'Agarose2', 'DL10000']
        m_ind = np.argmin(np.array([tae1, tae2, tae3]))

        # if self.By_index == 4:
        #     tae1 = self.TAE(self.Yd[:4] * (self.num - 1), [i * 8 for i in Agarose1_param[:4]])
        #     tae2 = self.TAE(self.Yd[:4] * (self.num - 1), [i * 8 for i in Agarose2_param[:4]])
        #     if tae1 < tae2:
        #         Mtype = 'Agarose1'
        #     else:
        #         Mtype = 'Agarose2'
        # elif self.By_index == 3:
        #     Mtype = 'DL10000'
        # else:
        #     Mtype = None
        return M[m_ind]

    def judge(self):
        if self.num == 9:
            if self.TAE(self.Yd, Agarose1_param) <= 0.1:
                Mtype = 'Agarose1'
                self.iscomplete = True
            elif self.TAE(self.Yd, Agarose2_param) <= 0.1:
                Mtype = 'Agarose2'
                self.iscomplete = True
            else:
                Mtype = self.judgebybri()
        elif self.num == 7:
            if self.TAE(self.Yd, DL10000_param) <= 0.1:
                Mtype = 'DL10000'
                self.iscomplete = True
            else:
                Mtype = self.judgebybri()
        else:
            Mtype = self.judgebybri()
        return Mtype

    def getNewY(self):
        if self.Mtype == 'Agarose1':
            new_Y = np.zeros(9)
            new_Y[4] = self.Y[self.By_index]
            new_Y[3] = self.Y[self.By_index - 1]
            for i in [2, 1, 0]:
                new_Y[i] = new_Y[i + 1] - int(
                    (self.By - self.Y[self.By_index - 1]) * Agarose1_param[i] / Agarose1_param[3])
            for i in [5, 6, 7, 8]:
                new_Y[i] = new_Y[i - 1] + int(
                    (self.By - self.Y[self.By_index - 1]) * Agarose1_param[i - 1] / Agarose1_param[3])
        elif self.Mtype == 'Agarose2':
            new_Y = np.zeros(9)
            new_Y[4] = self.Y[self.By_index]
            new_Y[3] = self.Y[self.By_index - 1]
            for i in [2, 1, 0]:
                new_Y[i] = new_Y[i + 1] - int(
                    (self.By - self.Y[self.By_index - 1]) * Agarose2_param[i] / Agarose2_param[3])
            for i in [5, 6, 7, 8]:
                new_Y[i] = new_Y[i - 1] + int(
                    (self.By - self.Y[self.By_index - 1]) * Agarose2_param[i - 1] / Agarose2_param[3])
        elif self.Mtype == 'DL10000':
            new_Y = np.zeros(7)
            new_Y[3] = self.Y[self.By_index]
            new_Y[2] = self.Y[self.By_index - 1]
            for i in [1, 0]:
                new_Y[i] = new_Y[i + 1] - int(
                    (self.By - self.Y[self.By_index - 1]) * DL10000_param[i] / DL10000_param[3])
            for i in [4, 5, 6]:
                new_Y[i] = new_Y[i - 1] + int(
                    (self.By - self.Y[self.By_index - 1]) * DL10000_param[i - 1] / DL10000_param[3])
        else:
            raise
        return new_Y.astype('uint16')

    def _padding(self):
        new_binary = np.zeros_like(self.binary_tape).astype('uint16')
        Wd = len(new_binary[0]) - self.size[0]
        for y in self.new_Y:
            for row in range(y - self.size[1], y):
                try:
                    new_binary[row, int(Wd / 2):-int(Wd / 2)] = 1
                except:
                    raise
        return new_binary