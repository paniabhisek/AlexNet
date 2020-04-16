**Final Edit**: tensorflow version: 1.7.0. The following text is written as per the reference as I was not able to reproduce the result. Key link in the following text: `bias of 1 in fully connected layers introduced dying relu problem <https://datascience.stackexchange.com/questions/37314/bias-of-1-in-fully-connected-layers-introduced-dying-relu-problem>`_. Key suggestion from `here <https://github.com/dontfollowmeimcrazy/imagenet/blob/master/tf/models/alexnet.py>`_

AlexNet
=======

This is the ``tensorflow`` implementation of `this paper <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_. For a more efficient implementation for GPU, head over to `here <http://code.google.com/p/cuda-convnet/>`_.

Dataset:

Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) **ImageNet Large Scale Visual Recognition Challenge**. arXiv:1409.0575, 2014. `paper <http://arxiv.org/abs/1409.0575>`_ | `bibtex <http://ai.stanford.edu/~olga/bibtex/ILSVRCarxiv14.bib>`_

Dataset info:

- Link: `ILSVRC2010 <http://www.image-net.org/challenges/LSVRC/2010/download-all-nonpub>`_
- Training size: *1261406 images*
- Validation size: *50000 images*
- Test size: *150000 images*
- Dataset size: *124 GB*
- GPU: *8 GB* GPU
- GPU Device: Quadro P4000

To save up time:

I got one corrupted image: ``n02487347_1956.JPEG``. The error read: ``Can not identify image file '/path/to/image/n02487347_1956.JPEG n02487347_1956.JPEG``. This happened when I read the image using ``PIL``. Before using this code, please make sure you can open ``n02487347_1956.JPEG`` using ``PIL``. If not delete the image.

How to Run
==========

- To train from scratch: ``python model.py <path-to-training-data> --resume False --train true``
- To resume training: ``python model.py <path-to-training-data> --resume True --train true`` or ``python model.py <path-to-training-data> --train true``
- To test: ``python model.py <path-to-training-data> --test true``

Performance
===========

With the model at the commit ``69ef36bccd2e4956f9e1371f453dfd84a9ae2829``, it looks like the model is overfitting substentially.

Some of the logs:

.. code::

    AlexNet - INFO - Time: 0.643425 Epoch: 0 Batch: 7821 Loss: 331154216.161753 Accuracy: 0.002953
    AlexNet - INFO - Time: 0.090631 Epoch: 1 Batch: 7821 Loss: 6.830358 Accuracy: 0.002981
    AlexNet - INFO - Time: 0.087481 Epoch: 2 Batch: 7821 Loss: 6.830358 Accuracy: 0.002981
    AlexNet - INFO - Time: 0.089649 Epoch: 3 Batch: 7821 Loss: 6.830358 Accuracy: 0.002981

Addition of dropout layer and/or data augmentation:

The model still overfits even if dropout layers has been added and the accuracies are almost similar to the previous one. After adding data augmentation method: sometime it goes to 100% and sometime it stays at 0% in the first epoch itself.

By mistakenly I have added ``tf.nn.conv2d`` which doesn't have any activation function by default as in the case for ``tf.contrib.layers.fully_connected`` (default is ``relu``). So it makes sense after 3 epochs there is no improvement in the accuracy.

Once ``relu`` has been added, the model was looking good. In the first epoch, few batch accuracies were 0.00781, 0.0156 with lot of other batches were 0s. In the second epoch the number of 0s decreased.

After changing the learning rate to ``0.001``:

.. code::

    2018-08-12 09:33:43,218 - AlexNet.LSVRC2010 - INFO - There are 1000 categories in total
    2018-08-12 09:34:18,758 - AlexNet.LSVRC2010 - INFO - There are 1261405 total training images in the dataset
    2018-08-12 09:34:19,778 - AlexNet - INFO - Creating placeholders for graph...
    2018-08-12 09:34:19,806 - AlexNet - INFO - Creating variables for graph...
    2018-08-12 09:34:19,821 - AlexNet - INFO - Initialize hyper parameters...
    2018-08-12 09:34:19,823 - AlexNet - INFO - Building the graph...
    2018-08-12 09:34:55,227 - AlexNet - INFO - Time: 18.011845 Epoch: 0 Batch: 0 Loss: 111833.523438 Avg loss: 111833.523438 Accuracy: 0.000000 Avg Accuracy: 0.000000 Top 5 Accuracy: 0.003906
    2018-08-12 09:35:01,769 - AlexNet - INFO - ===================Validation===================
    2018-08-12 09:35:01,769 - AlexNet - INFO - Loss: 111833.523438 Accuracy: 0.003906 Top 5 Accuracy: 0.007812
    2018-08-12 09:35:08,077 - AlexNet - INFO - Time: 12.849736 Epoch: 0 Batch: 10 Loss: 207.790985 Avg loss: 29811.891185 Accuracy: 0.003906 Avg Accuracy: 0.001420 Top 5 Accuracy: 0.011719
    2018-08-12 09:35:12,964 - AlexNet - INFO - Time: 4.886815 Epoch: 0 Batch: 20 Loss: 37.401054 Avg loss: 15659.049957 Accuracy: 0.000000 Avg Accuracy: 0.000930 Top 5 Accuracy: 0.000000
    2018-08-12 09:35:18,125 - AlexNet - INFO - Time: 5.160061 Epoch: 0 Batch: 30 Loss: 8.535903 Avg loss: 10612.695999 Accuracy: 0.003906 Avg Accuracy: 0.001134 Top 5 Accuracy: 0.011719
    2018-08-12 09:35:27,981 - AlexNet - INFO - Time: 9.856055 Epoch: 0 Batch: 40 Loss: 7.088428 Avg loss: 8026.056906 Accuracy: 0.000000 Avg Accuracy: 0.001048 Top 5 Accuracy: 0.035156
    2018-08-12 09:35:35,951 - AlexNet - INFO - Time: 7.969535 Epoch: 0 Batch: 50 Loss: 6.946403 Avg loss: 6453.687260 Accuracy: 0.000000 Avg Accuracy: 0.000996 Top 5 Accuracy: 0.785156
    2018-08-12 09:35:44,153 - AlexNet - INFO - Time: 8.200076 Epoch: 0 Batch: 60 Loss: 6.922817 Avg loss: 5396.842000 Accuracy: 0.000000 Avg Accuracy: 0.001153 Top 5 Accuracy: 0.960938
    2018-08-12 09:35:52,891 - AlexNet - INFO - Time: 8.737850 Epoch: 0 Batch: 70 Loss: 6.912984 Avg loss: 4637.697923 Accuracy: 0.000000 Avg Accuracy: 0.001045 Top 5 Accuracy: 0.988281
    2018-08-12 09:36:01,211 - AlexNet - INFO - Time: 8.319336 Epoch: 0 Batch: 80 Loss: 6.910833 Avg loss: 4065.996093 Accuracy: 0.003906 Avg Accuracy: 0.001061 Top 5 Accuracy: 0.984375
    2018-08-12 09:36:09,668 - AlexNet - INFO - Time: 8.457077 Epoch: 0 Batch: 90 Loss: 6.911587 Avg loss: 3619.943563 Accuracy: 0.000000 Avg Accuracy: 0.000944 Top 5 Accuracy: 0.996094
    2018-08-12 09:36:17,721 - AlexNet - INFO - Time: 8.052173 Epoch: 0 Batch: 100 Loss: 6.911614 Avg loss: 3262.217633 Accuracy: 0.000000 Avg Accuracy: 0.000928 Top 5 Accuracy: 1.000000
    2018-08-12 09:36:25,930 - AlexNet - INFO - Time: 8.208531 Epoch: 0 Batch: 110 Loss: 6.921659 Avg loss: 2968.946785 Accuracy: 0.000000 Avg Accuracy: 0.001056 Top 5 Accuracy: 0.996094
    2018-08-12 09:36:33,839 - AlexNet - INFO - Time: 7.908030 Epoch: 0 Batch: 120 Loss: 6.910044 Avg loss: 2724.150479 Accuracy: 0.000000 Avg Accuracy: 0.001065 Top 5 Accuracy: 0.996094
    2018-08-12 09:36:41,737 - AlexNet - INFO - Time: 7.898355 Epoch: 0 Batch: 130 Loss: 6.896086 Avg loss: 2516.727494 Accuracy: 0.007812 Avg Accuracy: 0.001133 Top 5 Accuracy: 1.000000
    2018-08-12 09:36:49,676 - AlexNet - INFO - Time: 7.937427 Epoch: 0 Batch: 140 Loss: 6.914582 Avg loss: 2338.726179 Accuracy: 0.000000 Avg Accuracy: 0.001108 Top 5 Accuracy: 1.000000
    2018-08-12 09:36:57,978 - AlexNet - INFO - Time: 8.301199 Epoch: 0 Batch: 150 Loss: 6.911684 Avg loss: 2184.301310 Accuracy: 0.000000 Avg Accuracy: 0.001061 Top 5 Accuracy: 1.000000
    2018-08-12 09:37:05,975 - AlexNet - INFO - Time: 7.986927 Epoch: 0 Batch: 160 Loss: 6.908568 Avg loss: 2049.059589 Accuracy: 0.000000 Avg Accuracy: 0.001043 Top 5 Accuracy: 1.000000
    2018-08-12 09:37:14,373 - AlexNet - INFO - Time: 8.396514 Epoch: 0 Batch: 170 Loss: 6.909007 Avg loss: 1929.635595 Accuracy: 0.000000 Avg Accuracy: 0.001051 Top 5 Accuracy: 1.000000

**The accuracy for current batch is ``0.000`` while the top 5 accuracy is ``1.000``**. That made me check my code for any implementation error (again!). The graph looked fine in ``tensorboard``. I didn't found any error.

The output of final layer: out of 1000 numbers for a single training example, all are 0s except few (3 or 4).

The ``relu`` activation function will make any negative numbers to zero. if the final layer produces 997 of them 0s and 3 non 0s, then ``tf.nn.in_top_k`` will think that this example's output is in top5 as all 997 of them are in 4th position. So there is nothing wrong in there, but one problem though, the training will be substantially slow or it might not converge at all. If we would have got considerable amount of non 0s then it would be faster then other known (``tanh``, ``signmoid``) activation function.

The output layer is producing lot of 0s which means it is producing lot of negative numbers before ``relu`` is applied.

A couple things can be done:

1. Reduce standard deviation to 0.01(currently 0.1), which will make the weights closer to 0 and maybe it will produce some more positive values
2. Apply local response normalization(not applying currently) and make standard deviation to 0.01
3. Use L2 regularization methods to penalize the weights for the way they are, in the hope they will be positive, and make standard deviation to 0.01.

Turns out none of them worked:

The next thing I could think of is to change the **Optimzer**. I was using ``tf.train.AdamOptimizer`` (as it is more recent and it's faster) but the paper is using **Gradient Descent with Momentum**. After changing the optimizer to ``tf.train.MomentumOptimizer`` *only* didn't improve anything.

But when I changed the optimizer to ``tf.train.MomentumOptimizer`` *along with* standard deviation to ``0.01``, things started to change. The top 5 accuracy was no longer ``1.000`` in the initial phase of training when top 1 accuracy was ``0.000``. A lot of positive values can also be seen in the output layer.

.. code::

    2018-08-15 07:48:16,518 - AlexNet.LSVRC2010 - INFO - There are 1000 categories in total
    2018-08-15 07:48:19,122 - AlexNet.LSVRC2010 - INFO - There are 1261405 total training images in the dataset
    2018-08-15 07:48:19,276 - AlexNet - INFO - Creating placeholders for graph...
    2018-08-15 07:48:19,286 - AlexNet - INFO - Creating variables for graph...
    2018-08-15 07:48:19,294 - AlexNet - INFO - Initialize hyper parameters...
    2018-08-15 07:48:19,294 - AlexNet - INFO - Building the graph...
    2018-08-15 07:48:31,054 - AlexNet - INFO - Time: 7.554070 Epoch: 0 Batch: 0 Loss: 12.694657 Avg loss: 12.694657 Accuracy: 0.000000 Avg Accuracy: 0.000000 Top 5 Accuracy: 0.000000
    2018-08-15 07:48:33,664 - AlexNet - INFO - Validation - Accuracy: 0.007812 Top 5 Accuracy: 0.007812
    2018-08-15 07:48:36,993 - AlexNet - INFO - Time: 5.938657 Epoch: 0 Batch: 10 Loss: 7.352790 Avg loss: 8.957169 Accuracy: 0.000000 Avg Accuracy: 0.003551 Top 5 Accuracy: 0.000000
    2018-08-15 07:48:39,417 - AlexNet - INFO - Time: 2.423536 Epoch: 0 Batch: 20 Loss: 6.988025 Avg loss: 8.058247 Accuracy: 0.000000 Avg Accuracy: 0.001860 Top 5 Accuracy: 0.000000
    2018-08-15 07:48:41,863 - AlexNet - INFO - Time: 2.445175 Epoch: 0 Batch: 30 Loss: 6.949255 Avg loss: 7.712968 Accuracy: 0.000000 Avg Accuracy: 0.001764 Top 5 Accuracy: 0.000000
    2018-08-15 07:48:44,271 - AlexNet - INFO - Time: 2.406801 Epoch: 0 Batch: 40 Loss: 6.983654 Avg loss: 7.531209 Accuracy: 0.000000 Avg Accuracy: 0.001715 Top 5 Accuracy: 0.000000
    2018-08-15 07:48:46,737 - AlexNet - INFO - Time: 2.465197 Epoch: 0 Batch: 50 Loss: 6.917971 Avg loss: 7.413875 Accuracy: 0.000000 Avg Accuracy: 0.001838 Top 5 Accuracy: 0.000000
    2018-08-15 07:48:49,130 - AlexNet - INFO - Time: 2.392022 Epoch: 0 Batch: 60 Loss: 6.905321 Avg loss: 7.335342 Accuracy: 0.000000 Avg Accuracy: 0.001665 Top 5 Accuracy: 0.007812
    2018-08-15 07:48:51,844 - AlexNet - INFO - Time: 2.713004 Epoch: 0 Batch: 70 Loss: 6.933993 Avg loss: 7.278179 Accuracy: 0.000000 Avg Accuracy: 0.001430 Top 5 Accuracy: 0.000000
    2018-08-15 07:48:54,833 - AlexNet - INFO - Time: 2.988007 Epoch: 0 Batch: 80 Loss: 6.945042 Avg loss: 7.234285 Accuracy: 0.000000 Avg Accuracy: 0.001640 Top 5 Accuracy: 0.000000
    2018-08-15 07:48:57,737 - AlexNet - INFO - Time: 2.903596 Epoch: 0 Batch: 90 Loss: 6.928125 Avg loss: 7.199531 Accuracy: 0.000000 Avg Accuracy: 0.001717 Top 5 Accuracy: 0.000000
    2018-08-15 07:49:00,525 - AlexNet - INFO - Time: 2.787234 Epoch: 0 Batch: 100 Loss: 6.927566 Avg loss: 7.171717 Accuracy: 0.000000 Avg Accuracy: 0.001702 Top 5 Accuracy: 0.000000
    2018-08-15 07:49:03,372 - AlexNet - INFO - Time: 2.845992 Epoch: 0 Batch: 110 Loss: 6.921791 Avg loss: 7.148882 Accuracy: 0.000000 Avg Accuracy: 0.001548 Top 5 Accuracy: 0.000000
    2018-08-15 07:49:06,260 - AlexNet - INFO - Time: 2.888164 Epoch: 0 Batch: 120 Loss: 6.902064 Avg loss: 7.129549 Accuracy: 0.000000 Avg Accuracy: 0.001550 Top 5 Accuracy: 0.023438
    2018-08-15 07:49:09,457 - AlexNet - INFO - Time: 3.196037 Epoch: 0 Batch: 130 Loss: 6.892216 Avg loss: 7.112829 Accuracy: 0.000000 Avg Accuracy: 0.001610 Top 5 Accuracy: 0.023438
    2018-08-15 07:49:12,286 - AlexNet - INFO - Time: 2.828465 Epoch: 0 Batch: 140 Loss: 6.919292 Avg loss: 7.098849 Accuracy: 0.007812 Avg Accuracy: 0.001662 Top 5 Accuracy: 0.007812
    2018-08-15 07:49:15,333 - AlexNet - INFO - Time: 3.046322 Epoch: 0 Batch: 150 Loss: 6.913494 Avg loss: 7.086305 Accuracy: 0.000000 Avg Accuracy: 0.001604 Top 5 Accuracy: 0.000000
    2018-08-15 07:49:18,165 - AlexNet - INFO - Time: 2.831317 Epoch: 0 Batch: 160 Loss: 6.919824 Avg loss: 7.075751 Accuracy: 0.000000 Avg Accuracy: 0.001553 Top 5 Accuracy: 0.000000
    2018-08-15 07:49:20,772 - AlexNet - INFO - Time: 2.606696 Epoch: 0 Batch: 170 Loss: 6.919248 Avg loss: 7.066110 Accuracy: 0.000000 Avg Accuracy: 0.001553 Top 5 Accuracy: 0.000000
    2018-08-15 07:49:23,477 - AlexNet - INFO - Time: 2.704031 Epoch: 0 Batch: 180 Loss: 6.897551 Avg loss: 7.057617 Accuracy: 0.007812 Avg Accuracy: 0.001511 Top 5 Accuracy: 0.015625
    2018-08-15 07:49:26,396 - AlexNet - INFO - Time: 2.918349 Epoch: 0 Batch: 190 Loss: 6.902122 Avg loss: 7.049929 Accuracy: 0.000000 Avg Accuracy: 0.001513 Top 5 Accuracy: 0.007812
    2018-08-15 07:49:29,334 - AlexNet - INFO - Time: 2.930083 Epoch: 0 Batch: 200 Loss: 6.909646 Avg loss: 7.043022 Accuracy: 0.007812 Avg Accuracy: 0.001594 Top 5 Accuracy: 0.007812
    2018-08-15 07:49:32,254 - AlexNet - INFO - Time: 2.918482 Epoch: 0 Batch: 210 Loss: 6.912971 Avg loss: 7.036663 Accuracy: 0.000000 Avg Accuracy: 0.001555 Top 5 Accuracy: 0.000000
    2018-08-15 07:49:35,127 - AlexNet - INFO - Time: 2.871966 Epoch: 0 Batch: 220 Loss: 6.914743 Avg loss: 7.030835 Accuracy: 0.000000 Avg Accuracy: 0.001555 Top 5 Accuracy: 0.000000
    2018-08-15 07:49:37,802 - AlexNet - INFO - Time: 2.674033 Epoch: 0 Batch: 230 Loss: 6.911674 Avg loss: 7.025807 Accuracy: 0.000000 Avg Accuracy: 0.001488 Top 5 Accuracy: 0.000000
    2018-08-15 07:49:40,728 - AlexNet - INFO - Time: 2.912393 Epoch: 0 Batch: 240 Loss: 6.921112 Avg loss: 7.021023 Accuracy: 0.000000 Avg Accuracy: 0.001491 Top 5 Accuracy: 0.000000
    2018-08-15 07:49:43,599 - AlexNet - INFO - Time: 2.869925 Epoch: 0 Batch: 250 Loss: 6.916319 Avg loss: 7.016570 Accuracy: 0.000000 Avg Accuracy: 0.001463 Top 5 Accuracy: 0.000000
    2018-08-15 07:49:46,381 - AlexNet - INFO - Time: 2.781311 Epoch: 0 Batch: 260 Loss: 6.907039 Avg loss: 7.012589 Accuracy: 0.007812 Avg Accuracy: 0.001437 Top 5 Accuracy: 0.007812
    2018-08-15 07:49:49,391 - AlexNet - INFO - Time: 3.009089 Epoch: 0 Batch: 270 Loss: 6.902983 Avg loss: 7.008793 Accuracy: 0.000000 Avg Accuracy: 0.001413 Top 5 Accuracy: 0.007812
    2018-08-15 07:49:52,207 - AlexNet - INFO - Time: 2.815411 Epoch: 0 Batch: 280 Loss: 6.912594 Avg loss: 7.005259 Accuracy: 0.000000 Avg Accuracy: 0.001390 Top 5 Accuracy: 0.000000
    2018-08-15 07:49:55,066 - AlexNet - INFO - Time: 2.843428 Epoch: 0 Batch: 290 Loss: 6.891138 Avg loss: 7.001918 Accuracy: 0.007812 Avg Accuracy: 0.001369 Top 5 Accuracy: 0.023438
    2018-08-15 07:49:58,272 - AlexNet - INFO - Time: 3.205470 Epoch: 0 Batch: 300 Loss: 6.914747 Avg loss: 6.998840 Accuracy: 0.000000 Avg Accuracy: 0.001402 Top 5 Accuracy: 0.000000
    2018-08-15 07:50:01,100 - AlexNet - INFO - Time: 2.827327 Epoch: 0 Batch: 310 Loss: 6.906322 Avg loss: 6.995992 Accuracy: 0.000000 Avg Accuracy: 0.001382 Top 5 Accuracy: 0.015625
    2018-08-15 07:50:03,924 - AlexNet - INFO - Time: 2.823234 Epoch: 0 Batch: 320 Loss: 6.911921 Avg loss: 6.993308 Accuracy: 0.000000 Avg Accuracy: 0.001387 Top 5 Accuracy: 0.000000
    2018-08-15 07:50:06,715 - AlexNet - INFO - Time: 2.790976 Epoch: 0 Batch: 330 Loss: 6.913865 Avg loss: 6.990783 Accuracy: 0.000000 Avg Accuracy: 0.001369 Top 5 Accuracy: 0.000000
    2018-08-15 07:50:09,480 - AlexNet - INFO - Time: 2.763432 Epoch: 0 Batch: 340 Loss: 6.913737 Avg loss: 6.988495 Accuracy: 0.000000 Avg Accuracy: 0.001352 Top 5 Accuracy: 0.007812
    2018-08-15 07:50:12,447 - AlexNet - INFO - Time: 2.967160 Epoch: 0 Batch: 350 Loss: 6.911568 Avg loss: 6.986181 Accuracy: 0.000000 Avg Accuracy: 0.001358 Top 5 Accuracy: 0.007812
    2018-08-15 07:50:15,123 - AlexNet - INFO - Time: 2.675277 Epoch: 0 Batch: 360 Loss: 6.916988 Avg loss: 6.984106 Accuracy: 0.000000 Avg Accuracy: 0.001320 Top 5 Accuracy: 0.000000
    2018-08-15 07:50:18,022 - AlexNet - INFO - Time: 2.899030 Epoch: 0 Batch: 370 Loss: 6.907708 Avg loss: 6.982115 Accuracy: 0.000000 Avg Accuracy: 0.001306 Top 5 Accuracy: 0.007812
    2018-08-15 07:50:21,034 - AlexNet - INFO - Time: 3.009900 Epoch: 0 Batch: 380 Loss: 6.915299 Avg loss: 6.980155 Accuracy: 0.000000 Avg Accuracy: 0.001271 Top 5 Accuracy: 0.000000
    2018-08-15 07:50:23,871 - AlexNet - INFO - Time: 2.835065 Epoch: 0 Batch: 390 Loss: 6.914540 Avg loss: 6.978238 Accuracy: 0.000000 Avg Accuracy: 0.001299 Top 5 Accuracy: 0.000000
    2018-08-15 07:50:26,741 - AlexNet - INFO - Time: 2.867900 Epoch: 0 Batch: 400 Loss: 6.922895 Avg loss: 6.976529 Accuracy: 0.000000 Avg Accuracy: 0.001364 Top 5 Accuracy: 0.000000
    2018-08-15 07:50:29,574 - AlexNet - INFO - Time: 2.832481 Epoch: 0 Batch: 410 Loss: 6.916379 Avg loss: 6.974939 Accuracy: 0.000000 Avg Accuracy: 0.001331 Top 5 Accuracy: 0.000000
    2018-08-15 07:50:32,332 - AlexNet - INFO - Time: 2.748183 Epoch: 0 Batch: 420 Loss: 6.914469 Avg loss: 6.973401 Accuracy: 0.000000 Avg Accuracy: 0.001299 Top 5 Accuracy: 0.000000
    2018-08-15 07:50:35,288 - AlexNet - INFO - Time: 2.954412 Epoch: 0 Batch: 430 Loss: 6.912920 Avg loss: 6.971925 Accuracy: 0.000000 Avg Accuracy: 0.001269 Top 5 Accuracy: 0.000000
    2018-08-15 07:50:38,041 - AlexNet - INFO - Time: 2.752243 Epoch: 0 Batch: 440 Loss: 6.905376 Avg loss: 6.970463 Accuracy: 0.000000 Avg Accuracy: 0.001276 Top 5 Accuracy: 0.015625
    2018-08-15 07:50:41,128 - AlexNet - INFO - Time: 3.087069 Epoch: 0 Batch: 450 Loss: 6.909246 Avg loss: 6.969112 Accuracy: 0.000000 Avg Accuracy: 0.001265 Top 5 Accuracy: 0.007812
    2018-08-15 07:50:44,073 - AlexNet - INFO - Time: 2.942974 Epoch: 0 Batch: 460 Loss: 6.904259 Avg loss: 6.967809 Accuracy: 0.000000 Avg Accuracy: 0.001271 Top 5 Accuracy: 0.015625
    2018-08-15 07:50:47,071 - AlexNet - INFO - Time: 2.997020 Epoch: 0 Batch: 470 Loss: 6.907288 Avg loss: 6.966543 Accuracy: 0.000000 Avg Accuracy: 0.001310 Top 5 Accuracy: 0.007812
    2018-08-15 07:50:49,881 - AlexNet - INFO - Time: 2.809317 Epoch: 0 Batch: 480 Loss: 6.911692 Avg loss: 6.965313 Accuracy: 0.000000 Avg Accuracy: 0.001299 Top 5 Accuracy: 0.000000
    2018-08-15 07:50:53,481 - AlexNet - INFO - Time: 3.600028 Epoch: 0 Batch: 490 Loss: 6.915403 Avg loss: 6.964301 Accuracy: 0.000000 Avg Accuracy: 0.001289 Top 5 Accuracy: 0.000000
    2018-08-15 07:50:56,337 - AlexNet - INFO - Time: 2.855357 Epoch: 0 Batch: 500 Loss: 6.901047 Avg loss: 6.963102 Accuracy: 0.000000 Avg Accuracy: 0.001325 Top 5 Accuracy: 0.031250
    2018-08-15 07:50:59,348 - AlexNet - INFO - Validation - Accuracy: 0.007812 Top 5 Accuracy: 0.015625

Atleast this will ensure training will not be slower.

Turns out changing the *optimizer* didn't improve the model, instead it only slowed down training. Near the end of epoch 1, the top 5 accuracy again went to 1.0000.

Final thing that I searched was `his <https://github.com/dontfollowmeimcrazy/imagenet/blob/master/tf/models/alexnet.py>`_  setting of bias, where he was using ``0`` as bias for fully connected layers. But the paper has strictly mentionied to use 1 as biases in fully connected layers. The model didn't overfit, it didn't create lot of 0s after the end of graph, loss started decreasing really well, accuracies were looking nice!! I don't fully understand at the moment why the bias in fully connected layers caused the problem. I've created a question on `datascience.stackexchange.com <https://datascience.stackexchange.com/questions/37314/bias-of-1-in-fully-connected-layers-introduced-dying-relu-problem>`_. If anyone knows how the bias helped the network to learn nicely, please comment or post your answer there! It'll surely help me and other folks who are struggling on the same problem.

The model has been trained for nearly 2 days. The top5 accuracy for validation were fluctuating between nearly 75% to 80% and top1 accuracy were fluctuating between nearly 50% to 55% at which point I stopped training.

For the commit ``d0cfd566157d7c12a1e75c102fff2a80b4dc3706``:

- screenlog.0: The log file after running ``python model.py <path-to-training-data> --train true`` in `screen <http://man7.org/linux/man-pages/man1/screen.1.html>`_
- model and logs: `google drive <https://drive.google.com/drive/folders/14olIl-cxSpXovDkhxZglcx9jDVEcVP2E>`_

Here are the graphs:

- *red line*: training
- *blue line*: validation

**top1 accuracy**:

.. image:: pictures/top1-acc.png

**top5 accuracy**:

.. image:: pictures/top5-acc.png

**loss**:

.. image:: pictures/loss.png

*Incase the above graphs are not visible clearly in terms of numbers on Github, please download it to your local computer, it should be clear there.*

**Note**: Near global step no 300k, I stopped it mistakenly. At that point it was 29 epochs and some hundered batches. But when I started again it started from epoch no 29 and batch no 0(as there wasn't any improvement for the few hundered batches). That's why the graph got little messed up.

With the current setting I've got the following accuracies for test dataset:

- Top1 accuracy: **47.9513%**
- Top5 accuracy: **71.8840%**

**Note**: To increase test accuracy, train the model for more epochs with lowering the learning rate when validation accuracy doesn't improve.
