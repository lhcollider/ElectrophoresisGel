import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from Recognition.Marker import Marker


class Labeldata():
    def Annotate_cols(self, image, cols):
        array = np.array(image)
        row = len(array)
        for i in range(len(cols)):
            col = (int(np.array(cols[i]).mean()))
            cv2.putText(image,
                        text=f'S{i}',
                        org=(int(col - 30), int(row - 30)),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.5,
                        color=(220, 20, 60)
                        )
        return image

    def Annotate_rows(self, image, rows, cols):
        range_ = [50, 1000]
        fit_p = np.polyfit(np.array([rows[0][-1][1], rows[0][0][1]]), np.array(range_), 1)
        for i in range(len(rows)):
            for j in range(len(rows[i])):
                row = rows[i][j][1]
                if i == 0:
                    place = (int(cols[i][0] - 30), int(row))
                else:
                    place = (int(cols[i][0] + 30), int(row))
                y = np.polyval(fit_p, row)
                cv2.putText(image,
                            text=f'{int(y)}',
                            org=place,
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=0.5,
                            color=(220, 20, 60)
                            )
        # cv2.imwrite('Annotation.png', image)
        return image


class splittape(Labeldata):
    def __init__(self, blockSize=25, connectivity=8):
        self.blockSize = blockSize
        self.connectivity = connectivity

    def getX(self, image):
        array = np.array(image)
        flip = 0
        cols = []
        for i in range(len(array[0])):
            sum_ = sum(array[:, i])
            if flip == 0:
                if sum_ != 0:
                    start = i
                    flip = 1 - flip
            else:
                if sum_ == 0:
                    cols.append((start, i))
                    flip = 1 - flip
        return cols

    def get_segments(self, image, cols):
        imgs = []
        array = np.array(image)
        for item in cols:
            start, end = item
            imgs.append(array[:, start:end])
        return imgs

    def gettape(self, image, cols):
        array = np.array(image)
        imgs = []
        for item in cols:
            start, end = item
            imgs.append(array[:, start:end])
        return imgs

    def BinaryandFilter(self, image):
        img = cv2.adaptiveThreshold(image,
                                    maxValue=1,
                                    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    thresholdType=cv2.THRESH_BINARY,
                                    blockSize=self.blockSize,
                                    C=0)
        return self._FilterSmallAreas(img)

    def ErodeandFilter(self, image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        img = cv2.erode(image, kernel)
        img = self._BandpassFilter(img)
        img = cv2.dilate(img, kernel)
        return img

    def _FilterSmallAreas(self, image, threshold=500):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=self.connectivity)
        for l in range(len(stats)):
            if stats[l][4] < threshold:
                labels[labels == l] = 0
        labels[labels != 0] = 1
        image = cv2.merge([labels.astype(np.uint8)])
        return image

    def _FilterBigAreas(self, image, threshold=5000):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=self.connectivity)
        for l in range(len(stats)):
            if stats[l][4] > threshold:
                labels[labels == l] = 0
        labels[labels != 0] = 1
        image = cv2.merge([labels.astype(np.uint8)])
        return image

    def _BandpassFilter(self, image, threshold1=500, threshold2=5000):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=self.connectivity)
        for l in range(len(stats)):
            if stats[l][4] < threshold1 or stats[l][4] > threshold2:
                labels[labels == l] = 0
        labels[labels != 0] = 1
        image = cv2.merge([labels.astype(np.uint8)])
        return image

    def getY(self, image, cols):
        images = self.get_segments(image, cols)
        flip = 0
        rows = []
        for image in images:
            row = []
            for i in range(len(image[:, 0])):
                sum_ = sum(image[i, :])
                if flip == 0:
                    if sum_ != 0:
                        start = i
                        flip = 1 - flip
                else:
                    if sum_ == 0:
                        row.append((start, i))
                        flip = 1 - flip
            rows.append(row)
        return rows

    def reconstruct(self, image):
        img_new = np.zeros_like(image)
        contours, _ = cv2.findContours(image.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            rect = cv2.minAreaRect(contours[i])
            if self.isTape(rect):
                for col, row in contours[i][:, 0, :]:
                    img_new[row, col] = 255
        return img_new

    def isTape(self, rect):
        (H, W), theta = rect[1:]
        if (H / W) > 2 and theta < 10:
            return True
        else:
            return False

    def Removeborder(self, image, top_rate=4, bottom_rate=8, left_rate=32, right_rate=32):
        n_row = len(image)
        n_col = len(image[0])
        image[:, :int(n_col / left_rate)] = 0
        image[:int(n_row/top_rate), :] = 0
        image[int(n_row * (1 - 1 / bottom_rate)):, :] = 0
        image[:, int(n_col * (1 - 1 / right_rate)):] = 0
        return image

    def process(self, image):
        img = self.BinaryandFilter(image)
        img = self.ErodeandFilter(img)
        img = self.Removeborder(img)
        # img = self.reconstruct(img)

        cols = self.getX(img)

        # 分割
        splited_imgs = self.gettape(image, cols)
        marker = splited_imgs[0]
        marker_binary = self.gettape(img, cols)[0]
        _, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        stats = stats[1:]
        if len(stats) != 0:
            mk = Marker(marker, marker_binary)
            if not mk.iscomplete and mk.Mtype:
                print(mk.Mtype)
                img[:, cols[0][0]:cols[0][1]] = mk.new_binary

        rows = self.getY(img, cols)

        out_img = self.Annotate_rows(self.Annotate_cols(image, cols), rows, cols)

        return img, out_img