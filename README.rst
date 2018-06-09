AlexNet
=======

This is the ``tensorflow`` implementation of `this paper <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_. For a more efficient implementation for GPU, head over to `here <http://code.google.com/p/cuda-convnet/>`_.

Performance
===========

With the model at the commit ``69ef36bccd2e4956f9e1371f453dfd84a9ae2829``, it looks like the model is overfitting substentially.

Some of the logs:

.. code::

    AlexNet - INFO - Time: 0.643425 Epoch: 0 Batch: 7821 Loss: 331154216.161753 Accuracy: 0.002953
    AlexNet - INFO - Time: 0.090631 Epoch: 1 Batch: 7821 Loss: 6.830358 Accuracy: 0.002981
    AlexNet - INFO - Time: 0.087481 Epoch: 2 Batch: 7821 Loss: 6.830358 Accuracy: 0.002981
    AlexNet - INFO - Time: 0.089649 Epoch: 3 Batch: 7821 Loss: 6.830358 Accuracy: 0.002981

So the next task is to add dropout layers and/or data augmentation methods. Data augmentation method will increase the training time substantially as compared to adding dropout layers. So let's add dropout layers in the last two fully connected layers.
The model still overfits even if dropout layers has been added and the accuracies are almost similar to the previous one.

Some of the techniques used for data augmentation methods are:

- Mirror
- Rotation
- Inversion (pixel wise)
- Random cropping

After adding the above data augmentation methods, the training accuracy is looking better.

.. code::

   AlexNet - INFO - Time: 7.628146 Epoch: 0 Batch: 62572 Loss: 60264.496469 Accuracy: 0.986872

**Note**: To train the model after adding data augmentation methods I had to use lesser original batch size(``16``, which increases the batch size to ``1232`` after data augmentation). Otherwise it throws ``Tensorflow-ResourceExhaustedError``
