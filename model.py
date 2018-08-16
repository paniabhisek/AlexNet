#!/usr/bin/python3
# -*- coding: utf-8 -*-

# library modules
from math import ceil
import json
import time
import os
import threading

# External library modules
import tensorflow as tf
import numpy as np

# local modules
from data import LSVRC2010
import logs

class AlexNet:
    """
    A tensorflow implementation of the paper:
    `AlexNet <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_
    """

    def __init__(self, path, batch_size, resume):
        """
        Build the AlexNet model
        """
        self.logger = logs.get_logger()

        self.resume = resume
        self.path = path
        self.batch_size = batch_size
        self.lsvrc2010 = LSVRC2010(self.path, batch_size)
        self.num_classes = len(self.lsvrc2010.wnid2label)

        self.learning_rate = 0.001
        self.momentum = 0.9
        self.lambd = tf.constant(0.0005, name='lambda')
        self.input_shape = (None, 227, 227, 3)
        self.output_shape = (None, self.num_classes)

        self.logger.info("Creating placeholders for graph...")
        self.create_tf_placeholders()

        self.logger.info("Creating variables for graph...")
        self.create_tf_variables()

        self.logger.info("Initialize hyper parameters...")
        self.hyper_param = {}
        self.init_hyper_param()

    def create_tf_placeholders(self):
        """
        Create placeholders for the graph.
        The input for these will be given while training or testing.
        """
        self.input_image = tf.placeholder(tf.float32, shape=self.input_shape,
                                          name='input_image')
        self.labels = tf.placeholder(tf.int32, shape=self.output_shape,
                                     name='output')

    def create_tf_variables(self):
        """
        Create variables for epoch, batch and global step
        """
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.cur_epoch = tf.Variable(0, name='epoch', trainable=False)
        self.cur_batch = tf.Variable(0, name='batch', trainable=False)

        self.increment_epoch_op = tf.assign(self.cur_epoch, self.cur_epoch+1)
        self.increment_batch_op = tf.assign(self.cur_batch, self.cur_batch+1)
        self.init_batch_op = tf.assign(self.cur_batch, 0)

    def init_hyper_param(self):
        """
        Store the hyper parameters.
        For each layer store number of filters(kernels)
        and filter size.
        If it's a fully connected layer then store the number of neurons.
        """
        with open('hparam.json') as f:
            self.hyper_param = json.load(f)

    def get_filter(self, layer_num, layer_name):
        """
        :param layer_num: Indicates the layer number in the graph
        :type layer_num: int
        :param layer_name: Name of the filter
        """
        layer = 'L' + str(layer_num)

        filter_height, filter_width, in_channels = self.hyper_param[layer]['filter_size']
        out_channels = self.hyper_param[layer]['filters']

        return tf.Variable(tf.truncated_normal(
            [filter_height, filter_width, in_channels, out_channels],
            dtype = tf.float32, stddev = 1e-2), name = layer_name)

    def get_strides(self, layer_num):
        """
        :param layer_num: Indicates the layer number in the graph
        :type layer_num: int
        """
        layer = 'L' + str(layer_num)

        stride = self.hyper_param[layer]['stride']
        strides = [1, stride, stride, 1]

        return strides

    def get_bias(self, layer_num, value=0.0):
        """
        Get the bias variable for current layer

        :param layer_num: Indicates the layer number in the graph
        :type layer_num: int
        """
        layer = 'L' + str(layer_num)
        initial = tf.constant(value,
                              shape=[self.hyper_param[layer]['filters']],
                              name='C' + str(layer_num))
        return tf.Variable(initial, name='B' + str(layer_num))

    @property
    def l2_loss(self):
        """
        Compute the l2 loss for all the weights
        """
        conv_bias_names = ['B' + str(i) for i in range(1, 6)]
        weights = []
        for v in tf.trainable_variables():
            if 'biases' in v.name: continue
            if v.name.split(':')[0] in conv_bias_names: continue
            weights.append(v)

        return self.lambd * sum(tf.nn.l2_loss(weight) for weight in weights)

    def build_graph(self):
        """
        Build the tensorflow graph for AlexNet.

        First 5 layers are Convolutional layers. Out of which
        first 2 and last layer will be followed by *max pooling*
        layers.

        Next 2 layers are fully connected layers.

        L1_conv -> L1_MP -> L2_conv -> L2_MP -> L3_conv
        -> L4_conv -> L5_conv -> L5_MP -> L6_FC -> L7_FC

        Where L1_conv -> Convolutional layer 1
              L5_MP -> Max pooling layer 5
              L7_FC -> Fully Connected layer 7

        Use `tf.nn.conv2d` to initialize the filters so
        as to reduce training time and `tf.layers.max_pooling2d`
        as we don't need to initialize in the pooling layer.
        """
        # Layer 1 Convolutional layer
        filter1 = self.get_filter(1, 'L1_filter')
        l1_conv = tf.nn.conv2d(self.input_image, filter1,
                               self.get_strides(1),
                               padding = self.hyper_param['L1']['padding'],
                               name='L1_conv')
        l1_conv = tf.add(l1_conv, self.get_bias(1))
        l1_conv = tf.nn.local_response_normalization(l1_conv,
                                                     depth_radius=5,
                                                     bias=2,
                                                     alpha=1e-4,
                                                     beta=.75)
        l1_conv = tf.nn.relu(l1_conv)

        # Layer 1 Max Pooling layer
        l1_MP = tf.layers.max_pooling2d(l1_conv,
                                        self.hyper_param['L1_MP']['filter_size'],
                                        self.hyper_param['L1_MP']['stride'],
                                        name='L1_MP')

        # Layer 2 Convolutional layer
        filter2 = self.get_filter(2, 'L2_filter')
        l2_conv = tf.nn.conv2d(l1_MP, filter2,
                               self.get_strides(2),
                               padding = self.hyper_param['L2']['padding'],
                               name='L2_conv')
        l2_conv = tf.add(l2_conv, self.get_bias(2, 1.0))
        l2_conv = tf.nn.local_response_normalization(l2_conv,
                                                     depth_radius=5,
                                                     bias=2,
                                                     alpha=1e-4,
                                                     beta=.75)
        l2_conv = tf.nn.relu(l2_conv)

        # Layer 2 Max Pooling layer
        l2_MP = tf.layers.max_pooling2d(l2_conv,
                                        self.hyper_param['L2_MP']['filter_size'],
                                        self.hyper_param['L2_MP']['stride'],
                                        name='L2_MP')

        # Layer 3 Convolutional layer
        filter3 = self.get_filter(3, 'L3_filter')
        l3_conv = tf.nn.conv2d(l2_MP, filter3,
                               self.get_strides(3),
                               padding = self.hyper_param['L3']['padding'],
                               name='L3_conv')
        l3_conv = tf.add(l3_conv, self.get_bias(3))
        l3_conv = tf.nn.relu(l3_conv)

        # Layer 4 Convolutional layer
        filter4 = self.get_filter(4, 'L4_filter')
        l4_conv = tf.nn.conv2d(l3_conv, filter4,
                               self.get_strides(4),
                               padding = self.hyper_param['L4']['padding'],
                               name='L4_conv')
        l4_conv = tf.add(l4_conv, self.get_bias(4, 1.0))
        l4_conv = tf.nn.relu(l4_conv)

        # Layer 5 Convolutional layer
        filter5 = self.get_filter(5, 'L5_filter')
        l5_conv = tf.nn.conv2d(l4_conv, filter5,
                               self.get_strides(5),
                               padding = self.hyper_param['L5']['padding'],
                               name='L5_conv')
        l5_conv = tf.add(l5_conv, self.get_bias(5, 1.0))
        l5_conv = tf.nn.relu(l5_conv)

        # Layer 5 Max Pooling layer
        l5_MP = tf.layers.max_pooling2d(l5_conv,
                                        self.hyper_param['L5_MP']['filter_size'],
                                        self.hyper_param['L5_MP']['stride'],
                                        name='L5_MP')

        flatten = tf.layers.flatten(l5_MP)

        # Layer 6 Fully connected layer
        l6_FC = tf.contrib.layers.fully_connected(flatten,
                                                  self.hyper_param['FC6'],
                                                  biases_initializer=tf.ones_initializer())

        # Dropout layer
        l6_keep_prob = tf.constant(0.5, tf.float32)
        l6_dropout = tf.nn.dropout(l6_FC, l6_keep_prob,
                                   name='l6_dropout')

        # Layer 7 Fully connected layer
        l7_FC = tf.contrib.layers.fully_connected(l6_dropout,
                                                  self.hyper_param['FC7'],
                                                  biases_initializer=tf.ones_initializer())

        # Dropout layer
        l7_keep_prob = tf.constant(0.5, tf.float32)
        l7_dropout = tf.nn.dropout(l7_FC, l7_keep_prob,
                                   name='l7_dropout')

        # final layer before softmax
        self.logits = tf.contrib.layers.fully_connected(l7_dropout,
                                                        self.num_classes)

        # loss function
        loss_function = tf.nn.softmax_cross_entropy_with_logits(
            logits = self.logits,
            labels = self.labels
        )

        # total loss
        self.loss = tf.reduce_mean(loss_function) + self.l2_loss

        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=self.momentum)\
                                 .minimize(self.loss, global_step=self.global_step)

        correct = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        self.top5_correct = tf.nn.in_top_k(self.logits, tf.argmax(self.labels, 1), 5)
        self.top5_accuracy = tf.reduce_mean(tf.cast(self.top5_correct, tf.float32))

        self.add_summaries()

    def add_summaries(self):
        """
        Add summaries for loss, top1 and top5 accuracies

        Add loss, top1 and top5 accuracies to summary files
        in order to visualize in tensorboard
        """
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('Top-1-Acc', self.accuracy)
        tf.summary.scalar('Top-5-Acc', self.top5_accuracy)

        self.merged = tf.summary.merge_all()

    def save_model(self, sess, saver):
        """
        Save the current model

        :param sess: Session object
        :param saver: Saver object responsible to store
        """
        model_base_path = os.path.join(os.getcwd(), 'model')
        if not os.path.exists(model_base_path):
            os.mkdir(model_base_path)
        model_save_path = os.path.join(os.getcwd(), 'model', 'model.ckpt')
        save_path = saver.save(sess, model_save_path)
        self.logger.info("Model saved in path: %s", save_path)

    def restore_model(self, sess, saver):
        """
        Restore previously saved model

        :param sess: Session object
        :param saver: Saver object responsible to store
        """
        model_base_path = os.path.join(os.getcwd(), 'model')
        model_restore_path = os.path.join(os.getcwd(), 'model', 'model.ckpt')
        saver.restore(sess, model_restore_path)
        self.logger.info("Model Restored from path: %s",
                         model_restore_path)

    def get_summary_writer(self, sess):
        """
        Get summary writer for training and validation

        Responsible for creating summary writer so it can
        write summaries to a file so it can be read by
        tensorboard later.
        """
        if not os.path.exists(os.path.join('summary', 'train')):
            os.makedirs(os.path.join('summary', 'train'))
        if not os.path.exists(os.path.join('summary', 'val')):
            os.makedirs(os.path.join('summary', 'val'))
        return (tf.summary.FileWriter(os.path.join(os.getcwd(),
                                                  'summary', 'train'),
                                      sess.graph),
                tf.summary.FileWriter(os.path.join(os.getcwd(),
                                                   'summary', 'val'),
                                      sess.graph))

    def train(self, epochs, thread='false'):
        """
        Train AlexNet.
        """
        batch_step, val_step = 10, 500

        self.logger.info("Building the graph...")
        self.build_graph()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            (summary_writer_train,
             summary_writer_val) = self.get_summary_writer(sess)
            if self.resume and os.path.exists(os.path.join(os.getcwd(),
                                                           'model')):
                self.restore_model(sess, saver)
            else:
                sess.run(init)

            resume_batch = True
            best_loss = float('inf')
            while sess.run(self.cur_epoch) < epochs:
                losses = []
                accuracies = []

                epoch = sess.run(self.cur_epoch)
                if not self.resume or (
                        self.resume and not resume_batch):
                    sess.run(self.init_batch_op)
                resume_batch = False
                start = time.time()
                gen_batch = self.lsvrc2010.gen_batch
                for images, labels in gen_batch:
                    batch_i = sess.run(self.cur_batch)
                    # If it's resumed from stored model,
                    # this will save from messing up the batch number
                    # in subsequent epoch
                    if batch_i >= ceil(len(self.lsvrc2010.image_names) / self.batch_size):
                        break
                    (_, loss, acc, top5_acc, global_step,
                     _) = sess.run([self.optimizer, self.loss,
                                    self.accuracy, self.top5_accuracy,
                                    self.global_step, self.increment_batch_op],
                                   feed_dict = {
                                       self.input_image: images,
                                       self.labels: labels
                                   })

                    losses.append(loss)
                    accuracies.append(acc)
                    if batch_i % batch_step == 0:
                        (summary, logits,
                         _top5) = sess.run([self.merged,
                                            self.logits, self.top5_correct],
                                           feed_dict = {
                                               self.input_image: images,
                                               self.labels: labels
                                           })
                        summary_writer_train.add_summary(summary, global_step)
                        summary_writer_train.flush()
                        end = time.time()
                        try:
                            true_idx = np.where(_top5[0]==True)[0][0]
                            self.logger.debug("logit at %d: %s", true_idx,
                                              str(logits[true_idx]))
                        except IndexError as ie:
                            self.logger.debug(ie)
                        self.logger.info("Time: %f Epoch: %d Batch: %d Loss: %f "
                                         "Avg loss: %f Accuracy: %f Avg Accuracy: %f "
                                         "Top 5 Accuracy: %f",
                                         end - start, epoch, batch_i,
                                         loss, sum(losses) / len(losses),
                                         acc, sum(accuracies) / len(accuracies),
                                         top5_acc)
                        start = time.time()

                    if batch_i % val_step == 0:
                        images_val, labels_val = self.lsvrc2010.get_batch_val
                        (summary,
                         acc, top5_acc) = sess.run([self.merged,
                                                    self.accuracy,
                                                    self.top5_accuracy],
                                                   feed_dict = {
                                                       self.input_image: images_val,
                                                       self.labels: labels_val
                                                   })
                        summary_writer_val.add_summary(summary, global_step)
                        summary_writer_val.flush()
                        self.logger.info("Validation - Accuracy: %f Top 5 Accuracy: %f",
                                         acc, top5_acc)

                        cur_loss = sum(losses) / len(losses)
                        if cur_loss < best_loss:
                            best_loss = cur_loss
                            self.save_model(sess, saver)

                # Increase epoch number
                sess.run(self.increment_epoch_op)

    def test(self):
        raise NotImplementedError

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar = 'image-path',
                        help = 'ImageNet dataset path')
    parser.add_argument('--resume', metavar='resume',
                        type=lambda x: x != 'False', default=True,
                        required=False,
                        help='Resume training (True or False)')
    args = parser.parse_args()

    alexnet = AlexNet(args.image_path, 128, resume=args.resume)
    alexnet.train(50)

