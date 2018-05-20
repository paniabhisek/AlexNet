#/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import random

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

    def get_image_for_1_batch(self, batch_size):
        """
        """
        pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar = 'image-path',
                        help = 'ImageNet dataset path')
    args = parser.parse_args()

    data = LSVRC2010(args.image_path)
