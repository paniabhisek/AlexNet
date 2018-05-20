#/usr/bin/python3
# -*- coding: utf-8 -*-

# library modules
import os
import random

# External library modules
import numpy as np

# local modules
from utils import image2nparray
import logs

class LSVRC2010:
    """
    Read the train data of ILSVRC2010.
    """

    def __init__(self, path):
        """
        Find which folder has what kind of images

        :param path: The directory path for the ILSVRC2010 training data
        """
        self.logger = logs.get_logger()
        self.path = path

        self.categories = {}
        self.read_categories()

        self.folders = []
        self.read_folders()
        
        self.image_names = []
        self.gen_image_names()
        random.shuffle(self.image_names)

    def read_categories(self):
        """
        For a folder name store which categories
        of images are present

        e.g. `self.categories['n00129435'] = ['roller']` means
        inside the folder 'n00129435', every image is a 'roller'
        """
        category_file = os.path.join(self.path, 'words.txt')
        with open(category_file) as c:
            for line in c:
                folder_name, _categories = line.split('\t')
                categories = list(map(lambda x: x.strip(), _categories.split(',')))
                self.categories[folder_name] = categories
        self.logger.info("There are %d categories in total", len(self.categories))

    def read_folders(self):
        """
        Find what all folder name's are present in the
        dataset.
        """
        wnids_file = os.path.join(self.path, 'wnids.txt')
        with open(wnids_file) as w:
            self.folders = list(map(lambda x: x.strip(), w.readlines()))
        self.logger.info("There are %d categories present in the dataset", len(self.folders))

    @property
    def num_images_in_a_folder(self):
        """
        Return number of images present in one folder
        of training set.
        """
        image_path = os.path.join(self.path, 'train',
                                  self.folders[0], 'images')
        return len(os.listdir(image_path))

    def gen_image_names(self):
        """
        If there are 1000 images in folder `f`, then all
        images inside `f` are  `f_0.JPEG`, `f_1.JPEG`, ..., `f_999.JPEG`.
        """
        # Number of images in a single training folder
        num_images = self.num_images_in_a_folder
        self.logger.info("The training dataset has %d images in each category",
                         num_images)

        for folder in self.folders:
            for image in range(num_images):
                # self.image_names.append(f_{idx}.JPEG)
                self.image_names.append(folder + '_' + str(image) + '.JPEG')

        self.logger.info("An example training file name: %s", self.image_names[0])

    def get_full_image_path(self, image_name):
        """
        Return full image path
        e.g. ../tiny-imagenet-200/train/n03854065/images/n03854065_297.JPEG

        :param image_name: The name of the image. e.g. n03854065_297.JPEG
        """
        folder_name = image_name.split('_')[0]
        return os.path.join(self.path, 'train', folder_name,
                            'images', image_name)

    def get_images_for_1_batch(self, batch_size):
        """
        A generator which returns `batch_size` of images in
        a numpy array

        :param batch_size: size of each batch
        """
        image_path = self.get_full_image_path(self.image_names[0])
        self.logger.info("image dimension: %s",
                         image2nparray(image_path).shape)

        start = 0
        end = batch_size
        num_images = len(self.image_names)

        while start < num_images:
            images = []
            # Careful for the end index
            if end > num_images:
                end = num_images

            # Convert to numpy array for all the images in current batch
            for i in range(start, end):
                image_path = self.get_full_image_path(self.image_names[i])
                images.append(image2nparray(image_path))
                if images[0].shape != images[-1].shape:
                    print(image_path, images[-1].shape)
            yield np.array(images)

            # Prepare for next batch
            start += batch_size
            end += batch_size

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar = 'image-path',
                        help = 'ImageNet dataset path')
    args = parser.parse_args()

    data = LSVRC2010(args.image_path)

    image_cur_batch = data.get_images_for_1_batch(128)
    data.logger.info("The first batch shape: %s", next(image_cur_batch).shape)
