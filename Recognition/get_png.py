import time
import cv2
import sys
import os
from skimage import io as io
from utils import normalize


if __name__ == '__main__':
    file_path = sys.argv[1]
    output_path = os.path.join(os.path.dirname(file_path), f'origin_{time.time()}.png')
    image = io.imread(file_path, as_gray=True)
    image = normalize(image, maxValue=255).astype('uint8')
    cv2.imencode('.png', image)[1].tofile(output_path)
    H, W = image.shape
    file_stats = os.stat(output_path)
    print(output_path + '|' + str(file_stats.st_size) + '|' + str(H) + '|' + str(W))