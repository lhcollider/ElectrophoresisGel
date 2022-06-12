import os
from matplotlib import pyplot as plt
from skimage import io as io
from split_tape import splittape
from preprocess import Preprocess
from utils.utils import normalize
import shutil
import argparse


if __name__ == '__main__':
    path = r'D:\python\ElectrophoresisGel\data\preprocess\262 263.png'
    preprocess_path = r'./test/preprocess'
    binary_path = r'./test/binary'
    output_path = r'./test/annotation'
    os.makedirs(preprocess_path, exist_ok=True)
    os.makedirs(binary_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    file = path.split('/')[-1]
    image = io.imread(path, as_gray=True)
    img = normalize(image, maxValue=255).astype('uint8')

    at = Preprocess()
    img = at.process(img)
    plt.figure()
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.savefig(os.path.join(preprocess_path, file.replace('.tif', '.png')),
                bbox_inches='tight',
                dpi=600,
                pad_inches=0.0)
    plt.close()

    st = splittape()
    img, out_img = st.process(img)

    plt.figure()
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.savefig(os.path.join(binary_path, file),
                bbox_inches='tight',
                dpi=600,
                pad_inches=0.0)
    plt.close()

    plt.figure()
    plt.axis('off')
    plt.imshow(out_img, cmap='gray')
    plt.savefig(os.path.join(output_path, file),
                bbox_inches='tight',
                dpi=600,
                pad_inches=0.0)
    plt.close()