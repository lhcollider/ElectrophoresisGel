import os
import time
from matplotlib import pyplot as plt
from skimage import io as io
from preprocess import Preprocess
from utils.utils import normalize


if __name__ == '__main__':
    raw_path = r'../data/raw'
    output_path = r'../data/preprocess'
    os.makedirs(output_path, exist_ok=True)
    at = Preprocess()
    start = time.time()
    for file in os.listdir(raw_path):
        print(file)
        file_path = os.path.join(raw_path, file)
        if file.endswith('.tif'):
            image = io.imread(file_path, as_gray=True)
            image = normalize(image, maxValue=255).astype('uint8')
            img = at.process(image)
            plt.figure()
            plt.axis('off')
            plt.imshow(img, cmap='gray')
            plt.savefig(os.path.join(output_path, file.replace('.tif', '.png')),
                        bbox_inches='tight',
                        dpi=600,
                        pad_inches=0.0)
            plt.close()
    end = time.time()
    print(f'time cost: {end - start}')