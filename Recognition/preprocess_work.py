import os
from skimage import io as io
from preprocess import Preprocess
from utils import normalize
import cv2


if __name__ == '__main__':
    raw_path = r'../data/raw'
    output_path = r'../data/preprocess'
    os.makedirs(output_path, exist_ok=True)
    at = Preprocess()
    # start = time.time()
    for file in os.listdir(raw_path):
        # if file not in ['PCR screen MG14.tif']:
        #     continue
        print(file)
        file_path = os.path.join(raw_path, file)
        if file.endswith('.tif'):
            image = io.imread(file_path, as_gray=True)
            image = normalize(image, maxValue=255).astype('uint8')
            img = at.process(image)
            H, W = img.shape
            # cv2.imwrite(os.path.join(output_path, file.replace('.tif', '.png')), img)
            cv2.imencode('.png', img)[1].tofile(os.path.join(output_path, file.replace('.tif', '.png')))
            # plt.figure(figsize=(W / 100, H / 100))
            # plt.axis('off')
            # plt.imshow(img, cmap='gray')
            # plt.savefig(os.path.join(output_path, file.replace('.tif', '.png')),
            #             bbox_inches='tight',
            #             # dpi=my_dpi,
            #             pad_inches=0.0)
            # plt.close()
            print(str(H) + '|' + str(W))
    # end = time.time()
    # print(f'time cost: {end - start}')