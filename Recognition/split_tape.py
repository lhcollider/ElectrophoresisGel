import numpy as np
import json
import os
import cv2
from matplotlib import pyplot as plt
from Marker import Marker
from interval import correctionCol, estimateNum


Agarose1_param = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000]
Agarose2_param = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000]
DL10000_param = [250, 500, 1000, 2000, 4000, 7000, 10000]


class Labeldata_js():
    def __init__(self):
        self.json_dict = []

    def Annotate_cols(self, cols, M_index):
        j = 1
        for i in range(len(cols)):
            col = (int(np.array(cols[i]).mean()))
            if i == M_index:
                dict_ = {"x":int(col),"y":100,"data":'M'}
                # df.append([int(col - 30), 100, 'M'])
                # cv2.putText(image,
                #             text='M',
                #             org=(int(col - 30), 100),
                #             fontFace=cv2.FONT_HERSHEY_COMPLEX,
                #             fontScale=1,
                #             color=(255, 0, 0)
                #             )
            else:
                dict_ = {"x":int(col),"y":100,"data":f'S{j}'}
                # df.append([int(col - 30), 100, f'S{j}'])
                # cv2.putText(image,
                #             text=f'S{j}',
                #             org=(int(col - 30), 100),
                #             fontFace=cv2.FONT_HERSHEY_COMPLEX,
                #             fontScale=1,
                #             color=(255, 0, 0)
                #             )
                # bp = anno[anno['sample_name'] == f'sam{j}']['pre_length'].item()
                # cv2.putText(image,
                #             text=bp,
                #             org=(int(col - 30), 150),
                #             fontFace=cv2.FONT_HERSHEY_COMPLEX,
                #             fontScale=1,
                #             color=(255, 0, 0))
                j += 1
            self.json_dict.append(dict_)
        # with open(r'../test.json', 'w', encoding='utf-8') as jf:
        #     json.dump(json_dict, jf)
        # df = pd.DataFrame(df, columns=['x', 'y', 'value'])
        # df.to_csv(r'../visualization/cols.csv')

    def Annotate_rows(self, rows, cols, M_index, param, iscomplete=False):
        # Y = rows[M_index]
        # df = []
        # 标记Marker
        # for i in range(len(rows[M_index])):
        #     place = (int(cols[M_index][0] + 30), int(rows[M_index][i]))
        #     cv2.putText(image,
        #                 text=f'{int(param[i])}',
        #                 org=place,
        #                 fontFace=cv2.FONT_HERSHEY_COMPLEX,
        #                 fontScale=0.5,
        #                 color=(255, 0, 0)
        #                 )
        # k = 1
        # 判断样本所在区间
        for i in range(len(rows)):
            for j in range(len(rows[i])):
                if i == M_index:
                    if not iscomplete:
                        dict_ = {"x":int(cols[M_index][0]),"y":int(rows[M_index][j]),"data":f'{int(param[len(param) - j - 1])}'}
                    else:
                        dict_ = {"x":int(cols[M_index][0]),"y":int(rows[M_index][j][0]),"data":f'{int(param[len(param) - j - 1])}'}
                    self.json_dict.append(dict_)
                    # df.append([int(cols[M_index][0] + 30), int(rows[M_index][j]), f'{int(param[len(param) - j - 1])}'])
                    # place = (int(cols[M_index][0] + 30), int(rows[M_index][j]))
                    # cv2.putText(image,
                    #             text=f'{int(param[len(param) - j - 1])}',
                    #             org=place,
                    #             fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    #             fontScale=1,
                    #             color=(255, 0, 0)
                    #             )
                # else:
                #     print(f'sam{k}: ')
                #     bp = int(anno[anno['sample_name'] == f'sam{k}']['pre_length'])
                #     y = rows[i][j][1]
                #     y_index = self.getinterval(y, Y)
                #     bp_index = self.getinterval(bp, param)
                #     if y_index == len(param) - bp_index:
                #         print('success')
                #     else:
                #         print('fail')
                #     k += 1
        # self.add_json(json_dict)
        # df = pd.DataFrame(df)
        # df.to_csv(r'../visualization/rows.csv')

    def getinterval(self, value, interval):
        for i in range(len(interval) + 1):
            if i == 0:
                if value < interval[0]:
                    range_index = 0
            elif i == len(interval) + 1:
                if value >= interval[-1]:
                    range_index = len(interval) + 1
            else:
                if value >= interval[i - 1] and value < interval[i]:
                    range_index = i
        return range_index

    def add_json(self, obj):
        item_list = []
        with open(r'../test.json', 'r', encoding='utf-8') as jf:
            load_dict = json.load(jf)
            num_item = len(load_dict)
            for i in range(num_item):
                x = load_dict[i]["x"]
                y = load_dict[i]["y"]
                data = load_dict[i]["data"]
                item_dict = {"x":x,"y":y,"data":data}
                item_list.append(item_dict)
        item_list.append(obj)
        with open(r'../test.json', 'w', encoding='utf-8') as jf2:
            json.dump(item_list, jf2, ensure_ascii=False)


class Labeldata():
    def Annotate_cols(self, image, cols, M_index):
        j = 1
        for i in range(len(cols)):
            col = (int(np.array(cols[i]).mean()))
            if i == M_index:
                cv2.putText(image,
                            text='M',
                            org=(int(col - 30), 100),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=1,
                            color=(255, 0, 0)
                            )
            else:
                cv2.putText(image,
                            text=f'S{j}',
                            org=(int(col - 30), 100),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=1,
                            color=(255, 0, 0)
                            )
                # bp = anno[anno['sample_name'] == f'sam{j}']['pre_length'].item()
                # cv2.putText(image,
                #             text=bp,
                #             org=(int(col - 30), 150),
                #             fontFace=cv2.FONT_HERSHEY_COMPLEX,
                #             fontScale=1,
                #             color=(255, 0, 0))
                j += 1
        return image

    def Annotate_rows(self, image, rows, cols, M_index, param, iscomplete=False):
        Y = rows[M_index]

        # 标记Marker
        # for i in range(len(rows[M_index])):
        #     place = (int(cols[M_index][0] + 30), int(rows[M_index][i]))
        #     cv2.putText(image,
        #                 text=f'{int(param[i])}',
        #                 org=place,
        #                 fontFace=cv2.FONT_HERSHEY_COMPLEX,
        #                 fontScale=0.5,
        #                 color=(255, 0, 0)
        #                 )
        k = 1
        # 判断样本所在区间
        for i in range(len(rows)):
            for j in range(len(rows[i])):
                if i == M_index:
                    if not iscomplete:
                        place = (int(cols[M_index][0] + 30), int(rows[M_index][j]))
                    else:
                        place = (int(cols[M_index][0] + 30), int(rows[M_index][j][0]))
                    cv2.putText(image,
                                text=f'{int(param[len(param) - j - 1])}',
                                org=place,
                                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                fontScale=1,
                                color=(255, 0, 0)
                                )
                # else:
                #     print(f'sam{k}: ')
                #     bp = int(anno[anno['sample_name'] == f'sam{k}']['pre_length'])
                #     y = rows[i][j][1]
                #     y_index = self.getinterval(y, Y)
                #     bp_index = self.getinterval(bp, param)
                #     if y_index == len(param) - bp_index:
                #         print('success')
                #     else:
                #         print('fail')
                #     k += 1
        return image

    def getinterval(self, value, interval):
        for i in range(len(interval) + 1):
            if i == 0:
                if value < interval[0]:
                    range_index = 0
            elif i == len(interval) + 1:
                if value >= interval[-1]:
                    range_index = len(interval) + 1
            else:
                if value >= interval[i - 1] and value < interval[i]:
                    range_index = i
        return range_index


class splittape():
    def __init__(self, blockSize=25, connectivity=8):
        self.blockSize = blockSize
        self.connectivity = connectivity
        self.W = 0
        self.H = 0

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

    def getMarker(self, rows):
        m_ = 0
        index = 0
        for i in range(len(rows)):
            if len(rows[i]) > m_:
                index = i
                m_ = len(rows[i])
        return index

    def get_segments(self, image, cols):
        imgs = []
        array = np.array(image)
        for item in cols:
            start, end = item
            imgs.append(array[:, start:end])
        return imgs

    # def gettape(self, image, cols):
    #     array = np.array(image)
    #     imgs = []
    #     for item in cols:
    #         start, end = item
    #         imgs.append(array[:, start:end])
    #     return imgs

    def BinaryandFilter(self, image):
        img = cv2.adaptiveThreshold(image,
                                    maxValue=1,
                                    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    thresholdType=cv2.THRESH_BINARY,
                                    blockSize=self.blockSize,
                                    C=0)
        # return self._FilterSmallAreas(img)
        return img

    def ErodeandFilter(self, image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        img = cv2.erode(image, kernel)
        img = self._BandpassFilter(img)
        img = cv2.dilate(img, kernel)
        return img

    def getpoint(self, stat):
        x = stat[0]
        y = stat[1]
        h = stat[2]
        w = stat[3]
        X = []
        Y = []
        for i in range(h):
            for j in range(w):
                X.append(x + i)
                Y.append(y + j)
        return X, Y

    def _FilterSmallAreas(self, image, threshold=300):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=self.connectivity)
        stats = stats[stats[:, 2] < (self.W / 2)]
        stats = stats[stats[:, 3] < (self.H / 2)]

        # index = []
        # for ind, i in enumerate(stats[:, 4] < threshold):
        #     if i == True:
        #         index.append(ind)
        # labels[labels == index] = 0
        X = []
        Y = []
        for l in range(len(stats)):
            if stats[l][4] < threshold:
                x, y = self.getpoint(stats[l])
                X.extend(x)
                Y.extend(y)
        labels[Y, X] = 0
        labels[labels != 0] = 1
        image = cv2.merge([labels.astype(np.uint8)])
        return image

    def _FilterBigAreas(self, image, threshold=5000):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=self.connectivity)
        stats = stats[stats[:, 2] < (self.W / 2)]
        stats = stats[stats[:, 3] < (self.H / 2)]

        X = []
        Y = []
        for l in range(len(stats)):
            if stats[l][4] < threshold:
                x, y = self.getpoint(stats[l])
                X.extend(x)
                Y.extend(y)
        labels[Y, X] = 0
        labels[labels != 0] = 1
        image = cv2.merge([labels.astype(np.uint8)])
        return image

    def _BandpassFilter(self, image, threshold1=200, threshold2=3000):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=self.connectivity)
        stats = stats[stats[:, 2] < (self.W / 2)]
        stats = stats[stats[:, 3] < (self.H / 2)]

        X = []
        Y = []
        for l in range(len(stats)):
            if stats[l][4] < threshold1 or stats[l][4] > threshold2:
                x, y = self.getpoint(stats[l])
                X.extend(x)
                Y.extend(y)
        labels[Y, X] = 0

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

    def Removeborder(self, image, top_rate=4, bottom_rate=8, left_rate=32, right_rate=16):
        image[:, :int(self.W / left_rate)] = 0
        image[:int(self.H / top_rate), :] = 0
        image[int(self.H * (1 - 1 / bottom_rate)):, :] = 0
        image[:, int(self.W * (1 - 1 / right_rate)):] = 0
        return image

    def process(self, image):
        self.H, self.W = image.shape
        # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.BinaryandFilter(image)
        img = self.ErodeandFilter(img)
        img = self.Removeborder(img)
        # img = self.reconstruct(img)

        cols = self.getX(img)
        num = estimateNum(cols)
        if len(cols) != num:
            cols = correctionCol(cols, num)

        # 分割
        splited_imgs = self.get_segments(image, cols)

        rows = self.getY(img, cols)
        M_index = self.getMarker(rows)
        marker = splited_imgs[M_index]
        marker_binary = self.get_segments(img, cols)[M_index]
        _, labels, stats, _ = cv2.connectedComponentsWithStats(marker_binary, connectivity=8)
        stats = stats[1:]
        if len(stats) != 0:
            mk = Marker(marker, marker_binary)
            if not mk.iscomplete and mk.Mtype:
                # print(mk.Mtype)
                img[:, cols[M_index][0]:cols[M_index][1]] = mk.new_binary
                rows[M_index] = mk.new_Y
            if mk.Mtype:
                if mk.Mtype == 'Agarose1':
                    param = Agarose1_param
                elif mk.Mtype == 'Agarpse2':
                    param = Agarose2_param
                elif mk.Mtype == 'DL10000':
                    param = DL10000_param
                else:
                    param = Agarose1_param


        # ld = Labeldata()
        # image = ld.Annotate_cols(image, cols, M_index)
        # out_img = ld.Annotate_rows(image, rows, cols, M_index, param, mk.iscomplete)
        # return img

        lj = Labeldata_js()
        lj.Annotate_cols(cols, M_index)
        lj.Annotate_rows(rows, cols, M_index, param, iscomplete=mk.iscomplete)
        return json.dumps(lj.json_dict)