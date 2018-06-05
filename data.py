#/usr/bin/python3
# -*- coding: utf-8 -*-

# library modules
import os
import random
import logging

# External library modules
import numpy as np

# local modules
from utils import image2nparray

class LSVRC2010:
    """
    Read the train data of ILSVRC2010.
    """

    def __init__(self, path):
        """
        Find which folder has what kind of images

        :param path: The directory path for the ILSVRC2010 training data
        """
        self.logger = logging.getLogger('AlexNet.LSVRC2010')
        self.path = path

        self.categories = {}
        self.read_categories()

        self.folders = []
        self.read_folders()
        # Use this to create one-hot encoding
        self.folder_indices = self.get_folder_indices()
        
        self.image_names = []
        self.gen_image_names()

    def get_folder_indices(self):
        """
        Indices of the folders in the sorted folder names.
        This will be helpful while creating one-hot encodings.

        :Example:
        >>> self.folders = ['hi', 'Alex', 'deep']
        >>> self.get_folder_indices()
        {'deep': 1, 'hi': 2, 'Alex': 0}
        """
        return dict((folder, i) for i, folder in enumerate(sorted(self.folders)))

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

        train_path = os.path.join(self.path, 'train')
        for folder in self.folders:
            for image in os.listdir(os.path.join(train_path, folder, 'images')):
                # self.image_names.append(f_{idx}.JPEG)
                self.image_names.append(image)

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

    def get_folder_indices_for_cur_batch(self, start, end):
        """
        Get the indices of the folders in the current batch.

        :param start: start index in `self.image_names`
        :param end: end index in `self.image_names`. `end` won't be included.

        :Example:
        >>> self.folder_indices = {'A': 1, 'B': 0, 'C': 2,
                'D': 4, 'E', 3}
        >>> start == 3, end == 5
        >>> self.image_names[3] == 'B_3.JPEG'
        >>> self.image_names[4] == 'D_1.JPEG'
        >>> self.get_folder_indices_for_cur_batch(3, 5)
        np.array([0, 4])
        """
        findices = []
        for i in range(start, end):
            folder = self.image_names[i].split('_')[0]
            findices.append(self.folder_indices[folder])

        return np.array(findices)

    def one_hot(self, start, end):
        """
        Get the one hot encoding of current
        batch of images.

        The size of the output encoding matrix
        has to be (batch size x no of categories).

        :param start: start index in `self.image_names`
        :param end: end index in `self.image_names`. `end` won't be included.
        """
        batch_size = end - start

        y_hat = np.zeros((batch_size, len(self.folders)))
        folder_indices = self.get_folder_indices_for_cur_batch(start, end)
        y_hat[np.arange(batch_size), folder_indices] = 1

        return y_hat

    def get_images_for_cur_batch(self, start, end, img_size):
        """
        Convert to numpy array for all the images in current batch

        :param start: start index in `self.image_names`
        :param end: end index in `self.image_names`. `end` won't be included.
        """
        images = []

        for i in range(start, end):
            image_path = self.get_full_image_path(self.image_names[i])
            images.append(image2nparray(image_path, img_size))

            if images[0].shape != images[-1].shape:
                self.logger.error("Image path: %s, shape: %s",
                                  image_path, images[-1].shape)

        return np.array(images)

    def get_images_for_1_batch(self, batch_size, img_size):
        """
        A generator which returns `batch_size` of images in
        a numpy array

        :param batch_size: size of each batch
        """
        random.shuffle(self.image_names)

        image_path = self.get_full_image_path(self.image_names[0])
        self.logger.info("image dimension: %s",
                         image2nparray(image_path).shape)

        start = 0
        end = batch_size
        num_images = len(self.image_names)

        while start < num_images:
            # Careful for the end index
            if end > num_images:
                end = num_images

            images = self.get_images_for_cur_batch(start, end, img_size)
            y_hat = self.one_hot(start, end)

            yield images, y_hat

            # Prepare for next batch
            start += batch_size
            end += batch_size

        self.logger.warning("Start idx: %d, Total no images: %d",
                            start, num_images)
        raise StopIteration

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar = 'image-path',
                        help = 'ImageNet dataset path')
    args = parser.parse_args()

    data = LSVRC2010(args.image_path)

    image_cur_batch = data.get_images_for_1_batch(128, (227, 227))
    first_batch = next(image_cur_batch)
    data.logger.info("The first batch shape: %s", first_batch[0].shape)
    data.logger.info("The first one hot vector shape: %s", first_batch[1].shape)
