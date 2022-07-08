import os
import cv2
from skimage import io as io
from split_tape import splittape
from utils import normalize
import json


if __name__ == '__main__':
    preprocessed_path = r'../data/preprocess'
    # binary_path = r'../data/test/binary'
    # output_path = r'../data/test/annotation'
    output_path = r'D:\python\ElectrophoresisGel\data\debug\annotation'
    # split_path = r'../data/test/split'
    os.makedirs(output_path, exist_ok=True)
    # os.makedirs(binary_path, exist_ok=True)
    st = splittape()
    for file in os.listdir(preprocessed_path):
        if file in ['678、CEN菌落PCR.png', 'lm-d-2.png', 'M13 酶切.png', 'PCR screen MG14.png', 'YWM678 MG1-8 YPD.png',
                    'ywm678（MG1)  5-FOA  hifi组装pcr.png', 'ywm678（MG1)  PB 5个 第二次验证.png']:
            continue
        print(file)
        file_path = os.path.join(preprocessed_path, file)
        if file.endswith('.png'):
            fname = file.replace('.png', '')
            image = io.imread(file_path, as_gray=True)
            img = normalize(image, maxValue=255).astype('uint8')
            anno = st.process(img)

            js = json.loads(anno)

            for point in js:
                cv2.putText(image,
                            text=point['data'],
                            org=(point['x'], point['y']),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=0.7,
                            color=(0, 0, 255)
                            )

            outfile_path = os.path.join(output_path, file)
            cv2.imencode('.png', image)[1].tofile(outfile_path)
            # print(anno)
            # cv2.imwrite(os.path.join(binary_path, file), img)
            # cv2.imencode('.png', img)[1].tofile(os.path.join(binary_path, file))
            # plt.figure()
            # plt.axis('off')
            # plt.imshow(img)
            # plt.savefig(os.path.join(binary_path, file),
            #             bbox_inches='tight',
            #             dpi=600,
            #             pad_inches=0.0)
            # plt.close()

            # plt.figure()
            # plt.axis('off')
            # plt.imshow(out_img)
            # plt.savefig(os.path.join(output_path, file),
            #             bbox_inches='tight',
            #             dpi=600,
            #             pad_inches=0.0)
            # plt.close()

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