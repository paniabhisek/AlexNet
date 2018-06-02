#/usr/bin/python3
# -*- coding: utf-8 -*-

# external library modules
from PIL import Image
import numpy as np

def image2nparray(image, size=None):
    """
    Converts an image to numpy data.
    If it is greyscale image, then convert it to RGB first
    and then change to numpy array

    :param image: path to the image file
    :param size: If given reshape the image to this size.
    """
    img = Image.open(image)

    if img.mode == 'L':
        img = img.convert('RGB')
    if size:
        img = img.resize(size)
    img.load()

    return np.asarray(img, dtype = "int32")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar = 'image-path',
                        help = 'ImageNet dataset path')
    args = parser.parse_args()
    im = image2nparray(args.image_path)
    print("Without reshaping, size:", im.shape)
    im = image2nparray(args.image_path, (127, 127))
    print("After reshaping, size:", im.shape)
