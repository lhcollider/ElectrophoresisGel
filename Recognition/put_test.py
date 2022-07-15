import os
import sys
import json
import cv2
import time
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

    # image_path = r'D:\python\ElectrophoresisGel\data\test\test.png'
    # data = '[{"x":174,"y":130,"data":"123"}]'

    # image = cv2.imread(image_path, 1)
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output_path = os.path.join(os.path.dirname(image_path), str(time.time()).replace('.', '')) + '.png'

    js = json.loads(data)

    for point in js:
        cv2.putText(image,
                    text=point['data'],
                    org=(point['x'], int(point['y'] + 23)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.6,
                    color=(0, 0, 255)
                    )
        # image = cv2ImgAddText(image, point.get('data'), point.get('x'), point.get('y'))

    cv2.imencode('.png', image)[1].tofile(output_path)

    file_stats = os.stat(output_path)
    print(output_path + '|' + str(file_stats.st_size))