AlexNet
=======

This is the `tensorflow` implementation of `this paper <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_. For a more efficient implementation for GPU, head over to `here <http://code.google.com/p/cuda-convnet/>`_.

Performance
===========

With the current model. It looks like the model is overfitting substentially.

Some of the logs:

.. code::

    AlexNet - INFO - Time: 0.643425 Epoch: 0 Batch: 7821 Loss: 331154216.161753 Accuracy: 0.002953
    AlexNet - INFO - Time: 0.090631 Epoch: 1 Batch: 7821 Loss: 6.830358 Accuracy: 0.002981
    AlexNet - INFO - Time: 0.087481 Epoch: 2 Batch: 7821 Loss: 6.830358 Accuracy: 0.002981
    AlexNet - INFO - Time: 0.089649 Epoch: 3 Batch: 7821 Loss: 6.830358 Accuracy: 0.002981
