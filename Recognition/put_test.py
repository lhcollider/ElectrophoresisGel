import os
import sys
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt


def cv2ImgAddText(img, text, left, top, textColor=(255, 0, 0), textSize=30):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    input_ = sys.argv[1]
    image_path = input_.split('~~~')[0]
    data = input_.split('~~~')[1]

    # image_path = r'D:\python\ElectrophoresisGel\data\a5a491c0b9c1722b0b4c067b8a9600ad__1657262918.9838579.png'
    # data = '[{"x": 69, "y": 100, "data": "S1"}, {"x": 191, "y": 100, "data": "S2"}, {"x": 313, "y": 100, "data": "S3"}, {"x": 435, "y": 100, "data": "S4"}, {"x": 557, "y": 100, "data": "S5"}, {"x": 679, "y": 100, "data": "S6"}, {"x": 801, "y": 100, "data": "S7"}, {"x": 923, "y": 100, "data": "S8"}, {"x": 1045, "y": 100, "data": "S9"}, {"x": 1167, "y": 100, "data": "S10"}, {"x": 1289, "y": 100, "data": "S11"}, {"x": 1411, "y": 100, "data": "M"}, {"x": 1533, "y": 100, "data": "S12"}, {"x": 1400, "y": 452, "data": "5000"}, {"x": 1400, "y": 489, "data": "3000"}, {"x": 1400, "y": 520, "data": "2000"}, {"x": 1400, "y": 543, "data": "1500"}, {"x": 1400, "y": 576, "data": "1000"}, {"x": 1400, "y": 596, "data": "750"}, {"x": 1400, "y": 622, "data": "500"}, {"x": 1400, "y": 655, "data": "250"}, {"x": 1400, "y": 675, "data": "100"}]'

    # image = cv2.imread(image_path, 1)
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output_path = image_path.replace('.png', '_anno.png')

    js = json.loads(data)

    for point in js:
        cv2.putText(image,
                    text=point['data'],
                    org=(point['x'], point['y']),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.7,
                    color=(0, 0, 255)
                    )
        # image = cv2ImgAddText(image, point.get('data'), point.get('x'), point.get('y'))

    cv2.imencode('.png', image)[1].tofile(output_path)

    file_stats = os.stat(output_path)
    print(output_path + '|' + str(file_stats.st_size))