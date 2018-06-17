#/usr/bin/python3
# -*- coding: utf-8 -*-

# library modules
import os
import random
import logging

# External library modules
import numpy as np
from scipy.io import loadmat

# local modules
from utils import image2PIL, image2nparray
from data_augment import augment

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

        self.val_images = []
        self.image_to_folder_val = {}

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
        But not necessarily as `0, 1, 2, ...`(increasing order from 0).
        So better to read what files are present in `f` rather than just
        assuming that all files are present in increasing order.
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

    def get_full_image_path_val(self, image_name):
        """
        Return full image path
        e.g. ../tiny-imagenet-200/train/n03854065_303.JPEG

        :param image_name: The name of the image. e.g. n03854065_303.JPEG
        """
        return os.path.join(self.path, 'val', image_name)

    def get_folder_indices_for_cur_batch(self, start, end, times):
        """
        Get the indices of the folders in the current batch.
        For each image you'll have `:py:times:` number of total
        images after doing *data augmentation*.

        :param start: start index in `self.image_names`
        :param end: end index in `self.image_names`. `end` won't be included.
        :param times: Total number of images for each image after
                      data augmentation.

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
            findices.extend([self.folder_indices[folder]] * times)

        return np.array(findices)

    def get_folder_indices_for_cur_batch_val(self, start, end):
        """
        Get the indices of the folders in the current batch.

        :param start: start index in `self.val_images`
        :param end: end index in `self.val_images`. `end` won't be included.

        :Example:
        >>> self.folder_indices = {'A': 1, 'B': 0, 'C': 2,
                'D': 4, 'E', 3}
        >>> start == 3, end == 5
        >>> self.val_images[3] == 'B_3.JPEG'
        >>> self.val_images[4] == 'D_1.JPEG'
        >>> self.get_folder_indices_for_cur_batch(3, 5)
        np.array([0, 4])
        """
        findices = []
        for i in range(start, end):
            folder = self.image_to_folder_val[self.val_images[i]]
            findices.append(self.folder_indices[folder])

        return np.array(findices)

    def one_hot(self, start, end, times):
        """
        Get the one hot encoding of current
        batch of images. For each image reserve total `:py:times:`
        space due to data augmentation.

        The size of the output encoding matrix
        has to be (batch size x no of categories).

        :param start: start index in `self.image_names`
        :param end: end index in `self.image_names`. `end` won't be included.
        :param times: Total number of one hot for each image after
                      data augmentation.
        """
        batch_size = end - start

        y_hat = np.zeros((batch_size * times, len(self.folders)))
        folder_indices = self.get_folder_indices_for_cur_batch(start, end, times)
        y_hat[np.repeat(np.arange(batch_size), times), folder_indices] = 1

        return y_hat

    def one_hot_val(self, start, end):
        """
        Get the one hot encoding of current

        The size of the output encoding matrix
        has to be (batch size x no of categories).

        :param start: start index in `self.val_images`
        :param end: end index in `self.val_images`. `end` won't be included.
        """
        batch_size = end - start

        y_hat = np.zeros((batch_size, len(self.folders)))
        folder_indices = self.get_folder_indices_for_cur_batch_val(start, end)
        y_hat[np.arange(batch_size), folder_indices] = 1

        return y_hat

    def get_images_for_cur_batch(self, start, end, img_size):
        """
        Convert to numpy array for all the images in current batch.
        For each image do data augmentation.

        :param start: start index in `self.image_names`
        :param end: end index in `self.image_names`. `end` won't be included.
        """
        images = []

        for i in range(start, end):
            image_path = self.get_full_image_path(self.image_names[i])
            image = image2PIL(image_path)
            images.extend(augment(image, img_size))

            if images[0].shape != images[-1].shape:
                self.logger.error("Image path: %s, shape: %s",
                                  image_path, images[-1].shape)

        return np.array(images)

    def get_images_for_cur_batch_val(self, start, end, img_size):
        """
        Convert to numpy array for all the images in current batch.
        For each image do data augmentation.

        :param start: start index in `self.image_names`
        :param end: end index in `self.image_names`. `end` won't be included.
        """
        images = []

        for i in range(start, end):
            image_path = self.get_full_image_path_val(self.val_images[i])
            image = image2nparray(image_path, img_size)
            images.append(image)

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
        # self.logger.info("image dimension: %s",
        #                 image2nparray(image_path).shape)

        start = 0
        end = batch_size
        num_images = len(self.image_names)

        while start < num_images:
            # Careful for the end index
            if end > num_images:
                end = num_images

            images = self.get_images_for_cur_batch(start, end, img_size)
            y_hat = self.one_hot(start, end,
                                 len(images) // (end - start))

            yield images, y_hat

            # Prepare for next batch
            start += batch_size
            end += batch_size

        self.logger.warning("Start idx: %d, Total no images: %d",
                            start, num_images)
        raise StopIteration

    def gen_validation_images(self):
        """
        Generate image names for the validation dataset.
        Only generate the names not the whole path
        """

        val_path = os.path.join(self.path, 'val')
        for image in os.listdir(val_path):
            # self.image_names.append(f_{idx}.JPEG)
            self.val_images.append(image)

        self.val_images.sort()

        self.logger.info("The validation dataset has %d images",
                         len(self.val_images))

    def find_synset_data_val(self):
        """
        Read validation ground truth file to find out
        which image belongs to which category.
        Store the image to folder_name mappings in
        `:py:self.image_to_folder_val:`
        """
        # Ground truth file
        g_file = os.path.join(self.path, 'devkit-1.0', 'data',
                              'ILSVRC2010_validation_ground_truth.txt')

        meta_file = os.path.join(self.path, 'devkit-1.0', 'data',
                                 'meta.mat')
        mat = loadmat(meta_file)
        synsets = mat['synsets']
        with open(g_file) as f:
            for i, file_name in zip(f, self.val_images):
                self.image_to_folder_val[file_name] = synsets[int(i) - 1][0][1][0]

    def get_images_for_1_batch_val(self, batch_size, img_size):
        """
        A generator which returns `batch_size` of images in
        a numpy array

        :param batch_size: size of each batch
        """
        self.gen_validation_images()
        self.find_synset_data_val()

        random.shuffle(self.val_images)

        start = 0
        end = batch_size
        num_images = len(self.val_images)

        while start < num_images:
            # Careful for the end index
            if end > num_images:
                end = num_images

            images = self.get_images_for_cur_batch_val(start, end, img_size)
            y_hat = self.one_hot_val(start, end)

            yield images, y_hat

            # Prepare for next batch
            start += batch_size
            end += batch_size

        self.logger.warning("Total no images: %d",
                            num_images)
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

    image_cur_batch = data.get_images_for_1_batch_val(2, (227, 227))
    first_batch = next(image_cur_batch)
    data.logger.info("The first batch shape: %s", first_batch[0].shape)
    data.logger.info("The first one hot vector shape: %s", first_batch[1].shape)
