import os
import numpy as np
from matplotlib import pyplot as plt
from skimage import io as io
from preprocess import Preprocess
from split_tape import splittape
from utils.utils import normalize


if __name__ == '__main__':
    preprocessed_path = r'../data/preprocess'
    binary_path = r'../data/test/binary'
    output_path = r'../data/test/annotation'
    # split_path = r'../data/test/split'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(binary_path, exist_ok=True)
    st = splittape()
    for file in os.listdir(preprocessed_path):
        print(file)
        file_path = os.path.join(preprocessed_path, file)
        if file.endswith('.png'):
            fname = file.replace('.png', '')
            image = io.imread(file_path)
            img = normalize(image, maxValue=255).astype('uint8')
            img, out_img = st.process(img)

            plt.figure()
            plt.axis('off')
            plt.imshow(img)
            plt.savefig(os.path.join(binary_path, file),
                        bbox_inches='tight',
                        dpi=600,
                        pad_inches=0.0)
            plt.close()

            plt.figure()
            plt.axis('off')
            plt.imshow(out_img)
            plt.savefig(os.path.join(output_path, file),
                        bbox_inches='tight',
                        dpi=600,
                        pad_inches=0.0)
            plt.close()

            # for i in range(len(splited_imgs)):
            #     plt.figure()
            #     savepath = os.path.join(split_path, f'{fname}')
            #     os.makedirs(savepath, exist_ok=True)
            #     plt.axis('off')
            #     plt.imshow(splited_imgs[i], cmap='gray')
            #     plt.savefig(os.path.join(savepath, f'{i+1}.png'),
            #                 bbox_inches='tight',
            #                 dpi=600,
            #                 pad_inches=0.0)
            #     plt.close()
    # plt.imshow(img, cmap='gray')
    # plt.show()
    # st = splittape()
    # img = st.process(img)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    # cols = st.seg_map()
    # st.gettape()