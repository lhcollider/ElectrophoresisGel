import cv2
import numpy as np
from matplotlib import pyplot as plt
from split_tape import splittape


class Preprocess():
    def __init__(self, blockSize=25):
        self.blockSize = blockSize

    def DeNoise(self, image, ksize=9, std=0):
        img = cv2.GaussianBlur(image,
                               ksize=(ksize, ksize),
                               sigmaX=std)
        img = cv2.medianBlur(img,
                             ksize=ksize)
        return img

    def getAngle(self, box):
        left_point_x = np.min(box[:, 0])
        right_point_x = np.max(box[:, 0])
        top_point_y = np.min(box[:, 1])
        bottom_point_y = np.max(box[:, 1])
        left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
        right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
        top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
        bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]

        tan_1 = abs(bottom_point_y - right_point_y) / abs(right_point_x - bottom_point_x)
        angle1 = np.arctan(tan_1) / np.pi * 180

        tan_2 = abs(bottom_point_y - left_point_y) / abs(bottom_point_x - left_point_x)
        angle2 = np.arctan(tan_2) / np.pi * 180

        if angle1 < angle2:
            return angle1
        else:
            return 360 - angle2

    def _getContours(self, image):
        img = cv2.adaptiveThreshold(image,
                                    maxValue=1,
                                    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    thresholdType=cv2.THRESH_BINARY,
                                    blockSize=self.blockSize,
                                    C=0)
        contours, _ = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        for i in range(len(contours)):
            area = (cv2.contourArea(contours[i]))
            if area > max_area:
                max_ind = i
                max_area = area
        rect = cv2.minAreaRect(contours[max_ind])
        box = cv2.boxPoints(rect)
        return box

    def Rotation(self, image, angle):
        (H, W) = image.shape[:2]
        (cX, cY) = (int(W // 2), int(H // 2))
        M = cv2.getRotationMatrix2D(center=(cX, cY),
                                    angle=-angle,
                                    scale=1)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((H * sin) + (W * cos))
        nH = int((H * cos) + (W * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH))

    def resizemap(self, image, box):
        left_point_x = int(np.min(box[:, 0]))
        right_point_x = int(np.max(box[:, 0]))
        top_point_y = int(np.min(box[:, 1]))
        bottom_point_y = int(np.max(box[:, 1]))
        return image[top_point_y:bottom_point_y, left_point_x:right_point_x]

    def process(self, image):
        # 旋转
        img = self.DeNoise(image)
        box = self._getContours(img)
        angle = self.getAngle(box)
        img = self.Rotation(image, angle)

        # 裁剪
        img = self.DeNoise(img)
        box_new = self._getContours(img)
        img = self.resizemap(img, box_new)

        # img_edge = cv2.Canny(img, 40, 120)
        # img_edge = cv2.Sobel(img,
        #                      ddepth=cv2.CV_64F,
        #                      dx=0,
        #                      dy=1)

        return img