import os
import time
import cv2
import sys
from skimage import io as io
from preprocess import Preprocess
from utils import normalize


if __name__ == '__main__':
    file_path = sys.argv[1]
    # file_path = r'D:\python\ElectrophoresisGel\data\a5a491c0b9c1722b0b4c067b8a9600ad_.jpg'
    if file_path.endswith('.jpg'):
        output_path = file_path.replace('.jpg', f'_{time.time()}.png')
    elif file_path.endswith('.jpeg'):
        output_path = file_path.replace('.jpeg', f'_{time.time()}.png')
    elif file_path.endswith('.png'):
        output_path = file_path.replace('.png', f'_{time.time()}.png')
    elif file_path.endswith('.tif'):
        output_path = file_path.replace('.tif', f'_{time.time()}.png')
    else:
        raise TypeError('file type needs jpg, jpeg, png, tif')

    at = Preprocess()
    image = io.imread(file_path, as_gray=True)
    image = normalize(image, maxValue=255).astype('uint8')
    try:
        img = at.process(image)
        H, W = img.shape
        # cv2.imwrite(output_path, img)
        cv2.imencode('.png', img)[1].tofile(output_path)
    except:
        H, W = image.shape

    file_stats = os.stat(output_path)
    print(output_path + '|' + str(file_stats.st_size) + '|' + str(H) + '|' + str(W))