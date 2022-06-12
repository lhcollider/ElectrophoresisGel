import os
import cv2
from skimage import io as io
from dataloder import getAnnotation
from utils.utils import normalize, show_bar
from matplotlib import pyplot as plt

preddir = r'D:\python\ElectrophoresisGel\data\test\binary'
gtdir = r'D:\python\ElectrophoresisGel\data\annotation'


def getComponents(image):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
    bbox = []
    for i in range(len(stats) - 1):
        xmin = stats[i+1, 0]
        ymin = stats[i+1, 1]
        xmax = stats[i+1, 0] + stats[i+1, 2]
        ymax = stats[i+1, 1] + stats[i+1, 3]
        bbox.append([xmin, ymin, xmax, ymax])
    return bbox


def isintersect(bbox1, bbox2):
    # 判断两个矩形是否相交
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = bbox1
    x11, y11, x12, y12 = bbox2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False


def calculateIOU(bbox1, bbox2):
    # bbox=(xA,yA,xB,yB)
    # 计算两个矩形框的重合度
    if isintersect(bbox1, bbox2) == True:
        x01, y01, x02, y02 = bbox1
        x11, y11, x12, y12 = bbox2
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        intersection = col * row
        area1 = (x02 - x01) * (y02 - y01)
        area2 = (x12 - x11) * (y12 - y11)
        coincide = intersection / (area1 + area2 - intersection)
        return coincide
    else:
        return 0


if __name__ == '__main__':
    annotation = getAnnotation(gtdir)
    total = {
        'P': 0,
        'TP': 0,
        'FP': 0,
        'precision': [],
        'recall': []
    }
    for file in os.listdir(preddir):
        alone = {
            'P': 0,
            'TP': 0,
            'FP': 0
        }
        filename = file.replace('.png', '')
        try:
            anno = annotation[filename]
        except:
            continue
        if len(anno['objects']) == 0:
            continue
        else:
            alone['P'] = len(anno['objects'])
        filepath = os.path.join(preddir, file)
        image = io.imread(filepath, as_gray=True)
        img = normalize(image, maxValue=255).astype('uint8')
        bbox_ps = getComponents(img)
        if len(bbox_ps) != 0:
            for bbox_p in bbox_ps:
                iouMax = 0
                for bbox_t in anno['objects']:
                    iou = calculateIOU(bbox_p, anno['objects'][bbox_t])
                    if iou > iouMax:
                        iouMax = iou
                if iouMax > 0.5:
                    alone['TP'] += 1
                else:
                    alone['FP'] += 1
            total['P'] += alone['P']
            total['TP'] += alone['TP']
            total['FP'] += alone['FP']
            total['precision'].append(alone['TP'] / (alone['TP'] + alone['FP']))
            total['recall'].append(alone['TP'] / alone['P'])
    precision = total['TP'] / (total['TP'] + total['FP'])
    recall = total['TP'] / total['P']
    print(f'total precision: {precision}, total recall: {recall}')
    print('precision list: {}, recall list: {}'.format(total['precision'], total['recall']))
    show_bar(total['precision'], 'precision')
    show_bar(total['recall'], 'recall')