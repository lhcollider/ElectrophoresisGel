import os
from xml.dom.minidom import parse


def getAnnotation(dirpath):
    annotation = {}
    for file in os.listdir(dirpath):
        filename = file.replace('.xml', '')
        filepath = os.path.join(dirpath, file)
        dom = parse(filepath)
        root = dom.documentElement
        sizes = root.getElementsByTagName('size')[0]
        W = int(sizes.getElementsByTagName('width')[0].childNodes[0].data)
        H = int(sizes.getElementsByTagName('height')[0].childNodes[0].data)
        C = int(sizes.getElementsByTagName('depth')[0].childNodes[0].data)
        size = [W, H, C]
        annotation.update({filename: {'size': size,
                                      'objects': {}}})
        for index, object_ in enumerate(root.getElementsByTagName('object')):
            bbox = object_.getElementsByTagName('bndbox')[0]
            xmin = int(bbox.getElementsByTagName('xmin')[0].childNodes[0].data)
            ymin = int(bbox.getElementsByTagName('ymin')[0].childNodes[0].data)
            xmax = int(bbox.getElementsByTagName('xmax')[0].childNodes[0].data)
            ymax = int(bbox.getElementsByTagName('ymax')[0].childNodes[0].data)
            annotation[filename]['objects'].update({f'bbox{index}': [xmin, ymin, xmax, ymax]})
    return annotation