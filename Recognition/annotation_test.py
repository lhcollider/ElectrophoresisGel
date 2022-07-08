import sys
from skimage import io as io
from split_tape import splittape
from utils import normalize


if __name__ == '__main__':
    file_path = sys.argv[1]
    # file_path = r'D:\python\ElectrophoresisGel\data\a5a491c0b9c1722b0b4c067b8a9600ad__1657262918.9838579.png'

    st = splittape()
    image = io.imread(file_path, as_gray=True)
    img = normalize(image, maxValue=255).astype('uint8')
    anno_data = st.process(img)
    print(anno_data)