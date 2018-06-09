#!/usr/bin/python
# -*- coding: utf-8 -*-

# library modules
from random import randint
import logging

# External library modules
from PIL import Image
from PIL import ImageOps

# local modules
from utils import imgs2np

logger = logging.getLogger('AlexNet.data_augment')

def mirror(image):
    """
    Return the horizontal mirror of the given image

    :type image: Pillow Image object
    """
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def rotate(image):
    """
    Rotate the given image to 90 and 270 degrees
    to produce to different images.

    :type image: Pillow Image object
    """
    # 90 degree
    img1 = image.rotate(90)

    # 270 degree
    img2 = image.rotate(270)

    return [img1, img2]

def mirror_and_rotate(image):
    """
    Mirror the image and then rotate(90 and 270
    degrees separately).

    :type image: Pillow Image object
    """
    return rotate(mirror(image))

def invert(image):
    """
    Invert the pixel intensities to produce
    different looking image.

    :type image: Pillow Image object
    """
    return ImageOps.invert(image)

def random_crop(image, times):
    """
    Randomly crop the given image for `:py:times:` many times.

    :type image: Pillow Image object
    :param times: How many times to crop the image
    """
    # random candidate
    random_cand = [image.size[0] - 227,
                   image.size[1] - 227]

    if random_cand[0] < 0:
        random_cand[0] = 0
        logger.warning("Image size: %d x %d", image.size[0],
                                image.size[1])
    if random_cand[1] < 0:
        random_cand[1] = 0
        logger.warning("Image size: %d x %d", image.size[0],
                                image.size[1])

    final_images = []
    for i in range(times):
        area = [None] * 4
        area[0] = randint(0, random_cand[0])
        area[1] = randint(0, random_cand[1])
        area[2] = area[0] + 227
        area[3] = area[1] + 227
        final_images.append(image.crop(area))

    return final_images

@imgs2np
def augment(image, size, times=10):
    """
    Augment data using different type of methods like
      - Mirroring
      - Rotation
      - Invertion
      - Cropping
    """
    augmented_imgs = [image.resize(size)]
    augmented_imgs.append(mirror(augmented_imgs[0]))
    augmented_imgs.extend(rotate(augmented_imgs[0]))
    augmented_imgs.extend(mirror_and_rotate(augmented_imgs[0]))
    augmented_imgs.append(invert(augmented_imgs[0]))

    random_crps = random_crop(image, times)

    for img_crp in random_crps:
        augmented_imgs.append(img_crp)
        #augmented_imgs = [img_crp]
        augmented_imgs.append(mirror(img_crp))
        augmented_imgs.extend(rotate(img_crp))
        augmented_imgs.extend(mirror_and_rotate(img_crp))
        augmented_imgs.append(invert(img_crp))

    return augmented_imgs
