import os
from matplotlib import pyplot as plt
from skimage import io as io
from split_tape import splittape
from Recognition.utils import normalize

if __name__ == '__main__':
    dir = r'D:/python/ElectrophoresisGel/data/test/preprocess'
    # anno_dir = r'../data/test/结果测评-ljh'
    binary_path = r'../data/test/binary'
    output_path = r'../data/test/annotation'
    os.makedirs(binary_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # at = Preprocess()
    # img = at.process(img)
    # plt.figure()
    # plt.axis('off')
    # plt.imshow(img, cmap='gray')
    # plt.savefig(os.path.join(preprocess_path, file.replace('.tif', '.png')),
    #             bbox_inches='tight',
    #             dpi=600,
    #             pad_inches=0.0)
    # plt.close()

    for file in os.listdir(dir):
        filename = file.replace('.png', '')
        if filename not in ['2021.12.24-1']:
            continue
        print(filename)
        path = os.path.join(dir, file)
        image = io.imread(path)[:, :, :3]
        img = normalize(image, maxValue=255).astype('uint8')
        st = splittape()
        try:
            img, out_img = st.process(img)
        except:
            continue

        plt.figure()
        plt.axis('off')
        plt.imshow(img, cmap='gray')
        plt.savefig(os.path.join(binary_path, file),
                    bbox_inches='tight',
                    dpi=600,
                    pad_inches=0.0)
        # plt.close()

        plt.figure()
        plt.axis('off')
        plt.imshow(out_img)
        plt.savefig(os.path.join(output_path, file),
                    bbox_inches='tight',
                    dpi=600,
                    pad_inches=0.0)
        # plt.close()