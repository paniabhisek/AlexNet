AlexNet
=======

This is the ``tensorflow`` implementation of `this paper <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_. For a more efficient implementation for GPU, head over to `here <http://code.google.com/p/cuda-convnet/>`_.

Dataset can be found `here <http://www.image-net.org/challenges/LSVRC/2010/>`_ (But you need your university email id to get it).

Dataset info:

- Training size: *1261406 images*
- Validation size: *50000 images*
- Dataset size: *124 GB*

To save up time:

I got one corrupted image (``n02487347_1956.JPEG``). The error read: ``Can not identify image file '/path/to/image/n02487347_1956.JPEG n02487347_1956.JPEG``. This happened when I read the image using ``PIL``. Before using this code, please make sure you can open ``n02487347_1956.JPEG`` using ``PIL``. If not delete the image, you won't loose anything if you delete 1 image out of 1 million.

So I trained on ``1261405`` images using *8 GB* GPU.

Performance
===========

With the model at the commit ``69ef36bccd2e4956f9e1371f453dfd84a9ae2829``, it looks like the model is overfitting substentially.

Some of the logs:

.. code::

    AlexNet - INFO - Time: 0.643425 Epoch: 0 Batch: 7821 Loss: 331154216.161753 Accuracy: 0.002953
    AlexNet - INFO - Time: 0.090631 Epoch: 1 Batch: 7821 Loss: 6.830358 Accuracy: 0.002981
    AlexNet - INFO - Time: 0.087481 Epoch: 2 Batch: 7821 Loss: 6.830358 Accuracy: 0.002981
    AlexNet - INFO - Time: 0.089649 Epoch: 3 Batch: 7821 Loss: 6.830358 Accuracy: 0.002981

So the next task is to add dropout layers and/or data augmentation methods. Data augmentation method will increase the training time substantially as compared to adding dropout layers. So let's first add dropout layers in the last two fully connected layers.

*One eternity later...*

The model still overfits even if dropout layers has been added and the accuracies are almost similar to the previous one. Adding data augmentation method wasn't a rescue either: Sometime it goes to 100% and sometime it stays at 0% in the first epoch itself.

After digging deeper it turns out that ``tf.nn.conv2d`` layer doesn't have any activation function by default as in the case for ``tf.contrib.layers.fully_connected`` (default is ``relu``). So basically I had 1 linear function and 3 non linear functions in the entire architecture as opposed to 8 non linear functions (as given in the paper). So it makes sense after 3 epochs there is no improvement in the accuracy.

Once ``relu`` has been added, the model was looking good. In the first epoch, few batch accuracies were 0.00781, 0.0156 with lot of other batches were 0s. In the second epoch the number of 0s decreased. So maybe the model will behave properly with other learning rates.

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

Which is kind of scary: **The accuracy for current batch is ``0.000`` while the top 5 accuracy is ``1.000``**. That made me check my code for any implementation error (again!). The graph looked fine in ``tensorboard``. After checking the code for couple of hours intensely, I didn't found any error. So one thing for sure!. I'm not able to find bug in my own code or something serious is going on. Now it was time to check the output of the final layer.

Guess what! Out of 1000 numbers for a single training example, all are 0s except few (3 or 4). Ahh! there you are!

What a relief!! At least this time there was no implementation error.

The ``relu`` activation function will make any negative numbers to zero. if the final layer produces 997 of them 0s and 3 non 0s, then ``tf.nn.in_top_k`` will think that this example's output is in top5 as all 997 of them are in 4th position. So there is nothing wrong in there, but one problem though, the training will be substantially slow or it might not converge at all. It will be slow as the derivative in the 0 case is 0 where learning is difficult. If we would have got considerable amount of non 0s then it would be faster then other known (``tanh``, ``signmoid``) activation function.

The output layer is producing lot of 0s which means it is producing lot of negative numbers before ``relu`` is applied.

A couple things can be done:

1. Reduce standard deviation to 0.01(currently 0.1), which will make the weights closer to 0 and maybe it will produce some more positive values
2. Apply local response normalization(not applying currently) and make standard deviation 0.01
3. Use L2 regularization methods to penalize the weights for the way they are, in the hope they will be positive, and make standard deviation 0.01.

Turns out none of them worked:

**1. Reduce standard deviation to 0.01(currently 0.1), which will make the weights closer to 0 and maybe it will produce some more positive values**:

.. code::

    2018-08-15 06:52:47,983 - AlexNet.LSVRC2010 - INFO - There are 1000 categories in total
    2018-08-15 06:53:29,007 - AlexNet.LSVRC2010 - INFO - There are 1261405 total training images in the dataset
    2018-08-15 06:53:30,106 - AlexNet - INFO - Creating placeholders for graph...
    2018-08-15 06:53:30,123 - AlexNet - INFO - Creating variables for graph...
    2018-08-15 06:53:30,132 - AlexNet - INFO - Initialize hyper parameters...
    2018-08-15 06:53:30,133 - AlexNet - INFO - Building the graph...
    2018-08-15 06:53:54,648 - AlexNet - INFO - Time: 10.361611 Epoch: 0 Batch: 0 Loss: 13.413406 Avg loss: 13.413406 Accuracy: 0.000000 Avg Accuracy: 0.000000 Top 5 Accuracy: 0.007812
    2018-08-15 06:53:57,136 - AlexNet - INFO - Validation - Accuracy: 0.000000 Top 5 Accuracy: 0.000000
    2018-08-15 06:54:00,965 - AlexNet - INFO - Time: 6.316998 Epoch: 0 Batch: 10 Loss: 6.907755 Avg loss: 16.692112 Accuracy: 0.000000 Avg Accuracy: 0.000710 Top 5 Accuracy: 1.000000
    2018-08-15 06:54:03,357 - AlexNet - INFO - Time: 2.391446 Epoch: 0 Batch: 20 Loss: 6.907755 Avg loss: 12.032895 Accuracy: 0.000000 Avg Accuracy: 0.001488 Top 5 Accuracy: 1.000000
    2018-08-15 06:54:05,807 - AlexNet - INFO - Time: 2.448325 Epoch: 0 Batch: 30 Loss: 6.907755 Avg loss: 10.379624 Accuracy: 0.000000 Avg Accuracy: 0.001008 Top 5 Accuracy: 1.000000
    2018-08-15 06:54:08,460 - AlexNet - INFO - Time: 2.651329 Epoch: 0 Batch: 40 Loss: 6.907755 Avg loss: 9.532827 Accuracy: 0.000000 Avg Accuracy: 0.000762 Top 5 Accuracy: 1.000000
    2018-08-15 06:54:11,879 - AlexNet - INFO - Time: 3.418311 Epoch: 0 Batch: 50 Loss: 6.907755 Avg loss: 9.018107 Accuracy: 0.000000 Avg Accuracy: 0.000919 Top 5 Accuracy: 1.000000
    2018-08-15 06:54:15,263 - AlexNet - INFO - Time: 3.383327 Epoch: 0 Batch: 60 Loss: 6.907755 Avg loss: 8.672148 Accuracy: 0.000000 Avg Accuracy: 0.001025 Top 5 Accuracy: 1.000000
    2018-08-15 06:54:19,154 - AlexNet - INFO - Time: 3.890871 Epoch: 0 Batch: 70 Loss: 6.907755 Avg loss: 8.423642 Accuracy: 0.000000 Avg Accuracy: 0.000880 Top 5 Accuracy: 1.000000
    2018-08-15 06:54:22,607 - AlexNet - INFO - Time: 3.452860 Epoch: 0 Batch: 80 Loss: 6.907755 Avg loss: 8.236495 Accuracy: 0.000000 Avg Accuracy: 0.000868 Top 5 Accuracy: 1.000000
    2018-08-15 06:54:25,781 - AlexNet - INFO - Time: 3.172624 Epoch: 0 Batch: 90 Loss: 6.907755 Avg loss: 8.090480 Accuracy: 0.000000 Avg Accuracy: 0.000773 Top 5 Accuracy: 1.000000
    2018-08-15 06:54:29,263 - AlexNet - INFO - Time: 3.480930 Epoch: 0 Batch: 100 Loss: 6.907755 Avg loss: 7.973378 Accuracy: 0.000000 Avg Accuracy: 0.000696 Top 5 Accuracy: 1.000000
    2018-08-15 06:54:32,737 - AlexNet - INFO - Time: 3.473827 Epoch: 0 Batch: 110 Loss: 6.907755 Avg loss: 7.877376 Accuracy: 0.000000 Avg Accuracy: 0.000704 Top 5 Accuracy: 1.000000
    2018-08-15 06:54:35,997 - AlexNet - INFO - Time: 3.259806 Epoch: 0 Batch: 120 Loss: 6.907755 Avg loss: 7.797242 Accuracy: 0.000000 Avg Accuracy: 0.000646 Top 5 Accuracy: 1.000000
    2018-08-15 06:54:39,531 - AlexNet - INFO - Time: 3.532853 Epoch: 0 Batch: 130 Loss: 6.907755 Avg loss: 7.729343 Accuracy: 0.000000 Avg Accuracy: 0.000656 Top 5 Accuracy: 1.000000
    2018-08-15 06:54:42,949 - AlexNet - INFO - Time: 3.416134 Epoch: 0 Batch: 140 Loss: 6.907755 Avg loss: 7.671074 Accuracy: 0.000000 Avg Accuracy: 0.000609 Top 5 Accuracy: 1.000000
    2018-08-15 06:54:46,287 - AlexNet - INFO - Time: 3.337396 Epoch: 0 Batch: 150 Loss: 6.907755 Avg loss: 7.620523 Accuracy: 0.000000 Avg Accuracy: 0.000621 Top 5 Accuracy: 1.000000
    2018-08-15 06:54:49,688 - AlexNet - INFO - Time: 3.399951 Epoch: 0 Batch: 160 Loss: 6.907755 Avg loss: 7.576253 Accuracy: 0.000000 Avg Accuracy: 0.000631 Top 5 Accuracy: 1.000000
    2018-08-15 06:54:53,093 - AlexNet - INFO - Time: 3.404040 Epoch: 0 Batch: 170 Loss: 6.907755 Avg loss: 7.537159 Accuracy: 0.000000 Avg Accuracy: 0.000777 Top 5 Accuracy: 1.000000
    2018-08-15 06:54:56,438 - AlexNet - INFO - Time: 3.344333 Epoch: 0 Batch: 180 Loss: 6.907755 Avg loss: 7.502385 Accuracy: 0.007812 Avg Accuracy: 0.000777 Top 5 Accuracy: 1.000000
    2018-08-15 06:55:00,018 - AlexNet - INFO - Time: 3.579973 Epoch: 0 Batch: 190 Loss: 6.907755 Avg loss: 7.471253 Accuracy: 0.000000 Avg Accuracy: 0.000777 Top 5 Accuracy: 1.000000
    2018-08-15 06:55:03,654 - AlexNet - INFO - Time: 3.619328 Epoch: 0 Batch: 200 Loss: 6.907755 Avg loss: 7.443218 Accuracy: 0.007812 Avg Accuracy: 0.000816 Top 5 Accuracy: 1.000000
    2018-08-15 06:55:06,996 - AlexNet - INFO - Time: 3.339940 Epoch: 0 Batch: 210 Loss: 6.907755 Avg loss: 7.417841 Accuracy: 0.000000 Avg Accuracy: 0.000852 Top 5 Accuracy: 1.000000
    2018-08-15 06:55:10,454 - AlexNet - INFO - Time: 3.456435 Epoch: 0 Batch: 220 Loss: 6.907755 Avg loss: 7.394760 Accuracy: 0.000000 Avg Accuracy: 0.000884 Top 5 Accuracy: 1.000000
    2018-08-15 06:55:14,217 - AlexNet - INFO - Time: 3.762058 Epoch: 0 Batch: 230 Loss: 6.907755 Avg loss: 7.373678 Accuracy: 0.000000 Avg Accuracy: 0.000913 Top 5 Accuracy: 1.000000
    2018-08-15 06:55:17,405 - AlexNet - INFO - Time: 3.185962 Epoch: 0 Batch: 240 Loss: 6.907755 Avg loss: 7.354345 Accuracy: 0.000000 Avg Accuracy: 0.000875 Top 5 Accuracy: 1.000000
    2018-08-15 06:55:20,889 - AlexNet - INFO - Time: 3.484184 Epoch: 0 Batch: 250 Loss: 6.907755 Avg loss: 7.336552 Accuracy: 0.000000 Avg Accuracy: 0.000872 Top 5 Accuracy: 1.000000
    2018-08-15 06:55:24,263 - AlexNet - INFO - Time: 3.373742 Epoch: 0 Batch: 260 Loss: 6.907755 Avg loss: 7.320123 Accuracy: 0.000000 Avg Accuracy: 0.000928 Top 5 Accuracy: 1.000000
    2018-08-15 06:55:27,601 - AlexNet - INFO - Time: 3.336430 Epoch: 0 Batch: 270 Loss: 6.907755 Avg loss: 7.304907 Accuracy: 0.000000 Avg Accuracy: 0.000923 Top 5 Accuracy: 1.000000
    2018-08-15 06:55:30,697 - AlexNet - INFO - Time: 3.095703 Epoch: 0 Batch: 280 Loss: 6.907755 Avg loss: 7.290773 Accuracy: 0.007812 Avg Accuracy: 0.000973 Top 5 Accuracy: 1.000000
    2018-08-15 06:55:34,075 - AlexNet - INFO - Time: 3.377865 Epoch: 0 Batch: 290 Loss: 6.907755 Avg loss: 7.277657 Accuracy: 0.000000 Avg Accuracy: 0.000940 Top 5 Accuracy: 1.000000
    2018-08-15 06:55:37,258 - AlexNet - INFO - Time: 3.181608 Epoch: 0 Batch: 300 Loss: 6.907755 Avg loss: 7.265368 Accuracy: 0.007812 Avg Accuracy: 0.000960 Top 5 Accuracy: 1.000000
    2018-08-15 06:55:40,592 - AlexNet - INFO - Time: 3.333960 Epoch: 0 Batch: 310 Loss: 6.907755 Avg loss: 7.253869 Accuracy: 0.000000 Avg Accuracy: 0.000980 Top 5 Accuracy: 1.000000
    2018-08-15 06:55:43,929 - AlexNet - INFO - Time: 3.336581 Epoch: 0 Batch: 320 Loss: 6.907755 Avg loss: 7.243087 Accuracy: 0.007812 Avg Accuracy: 0.000998 Top 5 Accuracy: 1.000000
    2018-08-15 06:55:47,365 - AlexNet - INFO - Time: 3.435523 Epoch: 0 Batch: 330 Loss: 6.907755 Avg loss: 7.232956 Accuracy: 0.000000 Avg Accuracy: 0.001015 Top 5 Accuracy: 1.000000

**2. Apply local response normalization(not applying currently) and make standard deviation 0.01**:

.. code::

    2018-08-15 06:59:22,239 - AlexNet.LSVRC2010 - INFO - There are 1000 categories in total
    2018-08-15 06:59:25,065 - AlexNet.LSVRC2010 - INFO - There are 1261405 total training images in the dataset
    2018-08-15 06:59:25,269 - AlexNet - INFO - Creating placeholders for graph...
    2018-08-15 06:59:25,281 - AlexNet - INFO - Creating variables for graph...
    2018-08-15 06:59:25,290 - AlexNet - INFO - Initialize hyper parameters...
    2018-08-15 06:59:25,291 - AlexNet - INFO - Building the graph...
    2018-08-15 06:59:38,311 - AlexNet - INFO - Time: 8.050453 Epoch: 0 Batch: 0 Loss: 12.315450 Avg loss: 12.315450 Accuracy: 0.000000 Avg Accuracy: 0.000000 Top 5 Accuracy: 0.015625
    2018-08-15 06:59:41,088 - AlexNet - INFO - Validation - Accuracy: 0.000000 Top 5 Accuracy: 0.000000
    2018-08-15 06:59:47,986 - AlexNet - INFO - Time: 9.674194 Epoch: 0 Batch: 10 Loss: 6.908002 Avg loss: 13.361458 Accuracy: 0.000000 Avg Accuracy: 0.000710 Top 5 Accuracy: 1.000000
    2018-08-15 06:59:53,612 - AlexNet - INFO - Time: 5.622507 Epoch: 0 Batch: 20 Loss: 6.907755 Avg loss: 10.288270 Accuracy: 0.000000 Avg Accuracy: 0.000744 Top 5 Accuracy: 1.000000
    2018-08-15 06:59:59,191 - AlexNet - INFO - Time: 5.577700 Epoch: 0 Batch: 30 Loss: 6.907755 Avg loss: 9.197782 Accuracy: 0.000000 Avg Accuracy: 0.000504 Top 5 Accuracy: 1.000000
    2018-08-15 07:00:04,791 - AlexNet - INFO - Time: 5.599974 Epoch: 0 Batch: 40 Loss: 6.907755 Avg loss: 8.639239 Accuracy: 0.007812 Avg Accuracy: 0.000953 Top 5 Accuracy: 1.000000
    2018-08-15 07:00:10,433 - AlexNet - INFO - Time: 5.641251 Epoch: 0 Batch: 50 Loss: 6.907755 Avg loss: 8.299732 Accuracy: 0.000000 Avg Accuracy: 0.000766 Top 5 Accuracy: 1.000000
    2018-08-15 07:00:16,001 - AlexNet - INFO - Time: 5.567920 Epoch: 0 Batch: 60 Loss: 6.907755 Avg loss: 8.071539 Accuracy: 0.000000 Avg Accuracy: 0.000640 Top 5 Accuracy: 1.000000
    2018-08-15 07:00:21,584 - AlexNet - INFO - Time: 5.582505 Epoch: 0 Batch: 70 Loss: 6.907755 Avg loss: 7.907626 Accuracy: 0.000000 Avg Accuracy: 0.000880 Top 5 Accuracy: 1.000000
    2018-08-15 07:00:27,187 - AlexNet - INFO - Time: 5.601774 Epoch: 0 Batch: 80 Loss: 6.907755 Avg loss: 7.784185 Accuracy: 0.000000 Avg Accuracy: 0.001061 Top 5 Accuracy: 1.000000
    2018-08-15 07:00:32,775 - AlexNet - INFO - Time: 5.587914 Epoch: 0 Batch: 90 Loss: 6.907755 Avg loss: 7.687874 Accuracy: 0.000000 Avg Accuracy: 0.001030 Top 5 Accuracy: 1.000000
    2018-08-15 07:00:38,379 - AlexNet - INFO - Time: 5.603066 Epoch: 0 Batch: 100 Loss: 6.907755 Avg loss: 7.610635 Accuracy: 0.000000 Avg Accuracy: 0.001006 Top 5 Accuracy: 1.000000
    2018-08-15 07:00:43,990 - AlexNet - INFO - Time: 5.610184 Epoch: 0 Batch: 110 Loss: 6.907755 Avg loss: 7.547312 Accuracy: 0.000000 Avg Accuracy: 0.000915 Top 5 Accuracy: 1.000000
    2018-08-15 07:00:49,617 - AlexNet - INFO - Time: 5.626644 Epoch: 0 Batch: 120 Loss: 6.907755 Avg loss: 7.494456 Accuracy: 0.000000 Avg Accuracy: 0.000968 Top 5 Accuracy: 1.000000
    2018-08-15 07:00:55,244 - AlexNet - INFO - Time: 5.626805 Epoch: 0 Batch: 130 Loss: 6.907755 Avg loss: 7.449670 Accuracy: 0.000000 Avg Accuracy: 0.000954 Top 5 Accuracy: 1.000000
    2018-08-15 07:01:00,892 - AlexNet - INFO - Time: 5.647589 Epoch: 0 Batch: 140 Loss: 6.907755 Avg loss: 7.411236 Accuracy: 0.000000 Avg Accuracy: 0.000887 Top 5 Accuracy: 1.000000
    2018-08-15 07:01:07,250 - AlexNet - INFO - Time: 6.356997 Epoch: 0 Batch: 150 Loss: 6.907755 Avg loss: 7.377893 Accuracy: 0.000000 Avg Accuracy: 0.000931 Top 5 Accuracy: 1.000000
    2018-08-15 07:01:13,032 - AlexNet - INFO - Time: 5.781278 Epoch: 0 Batch: 160 Loss: 6.907755 Avg loss: 7.348692 Accuracy: 0.000000 Avg Accuracy: 0.000922 Top 5 Accuracy: 1.000000
    2018-08-15 07:01:18,728 - AlexNet - INFO - Time: 5.695673 Epoch: 0 Batch: 170 Loss: 6.907755 Avg loss: 7.322906 Accuracy: 0.000000 Avg Accuracy: 0.000868 Top 5 Accuracy: 1.000000
    2018-08-15 07:01:24,419 - AlexNet - INFO - Time: 5.689912 Epoch: 0 Batch: 180 Loss: 6.907755 Avg loss: 7.299970 Accuracy: 0.007812 Avg Accuracy: 0.000950 Top 5 Accuracy: 1.000000
    2018-08-15 07:01:30,132 - AlexNet - INFO - Time: 5.712874 Epoch: 0 Batch: 190 Loss: 6.907755 Avg loss: 7.279435 Accuracy: 0.000000 Avg Accuracy: 0.000900 Top 5 Accuracy: 1.000000
    2018-08-15 07:01:35,792 - AlexNet - INFO - Time: 5.659493 Epoch: 0 Batch: 200 Loss: 6.907755 Avg loss: 7.260944 Accuracy: 0.000000 Avg Accuracy: 0.000972 Top 5 Accuracy: 1.000000
    2018-08-15 07:01:41,390 - AlexNet - INFO - Time: 5.597213 Epoch: 0 Batch: 210 Loss: 6.907755 Avg loss: 7.244205 Accuracy: 0.000000 Avg Accuracy: 0.000963 Top 5 Accuracy: 1.000000
    2018-08-15 07:01:47,112 - AlexNet - INFO - Time: 5.721606 Epoch: 0 Batch: 220 Loss: 6.907755 Avg loss: 7.228981 Accuracy: 0.000000 Avg Accuracy: 0.000954 Top 5 Accuracy: 1.000000
    2018-08-15 07:01:52,803 - AlexNet - INFO - Time: 5.689929 Epoch: 0 Batch: 230 Loss: 6.907755 Avg loss: 7.215075 Accuracy: 0.000000 Avg Accuracy: 0.000913 Top 5 Accuracy: 1.000000
    2018-08-15 07:01:58,468 - AlexNet - INFO - Time: 5.664722 Epoch: 0 Batch: 240 Loss: 6.907755 Avg loss: 7.202323 Accuracy: 0.007812 Avg Accuracy: 0.000908 Top 5 Accuracy: 1.000000
    2018-08-15 07:02:04,115 - AlexNet - INFO - Time: 5.646429 Epoch: 0 Batch: 250 Loss: 6.907755 Avg loss: 7.190587 Accuracy: 0.007812 Avg Accuracy: 0.000934 Top 5 Accuracy: 1.000000
    2018-08-15 07:02:09,733 - AlexNet - INFO - Time: 5.616850 Epoch: 0 Batch: 260 Loss: 6.907755 Avg loss: 7.179751 Accuracy: 0.007812 Avg Accuracy: 0.000988 Top 5 Accuracy: 1.000000
    2018-08-15 07:02:15,368 - AlexNet - INFO - Time: 5.634574 Epoch: 0 Batch: 270 Loss: 6.907755 Avg loss: 7.169714 Accuracy: 0.000000 Avg Accuracy: 0.000980 Top 5 Accuracy: 1.000000
    2018-08-15 07:02:21,036 - AlexNet - INFO - Time: 5.667386 Epoch: 0 Batch: 280 Loss: 6.907755 Avg loss: 7.160392 Accuracy: 0.007812 Avg Accuracy: 0.001001 Top 5 Accuracy: 1.000000
    2018-08-15 07:02:26,688 - AlexNet - INFO - Time: 5.651589 Epoch: 0 Batch: 290 Loss: 6.907755 Avg loss: 7.151710 Accuracy: 0.000000 Avg Accuracy: 0.000966 Top 5 Accuracy: 1.000000
    2018-08-15 07:02:32,290 - AlexNet - INFO - Time: 5.601314 Epoch: 0 Batch: 300 Loss: 6.907755 Avg loss: 7.143605 Accuracy: 0.000000 Avg Accuracy: 0.000934 Top 5 Accuracy: 1.000000
    2018-08-15 07:02:37,958 - AlexNet - INFO - Time: 5.667574 Epoch: 0 Batch: 310 Loss: 6.907755 Avg loss: 7.136022 Accuracy: 0.000000 Avg Accuracy: 0.000980 Top 5 Accuracy: 1.000000
    2018-08-15 07:02:43,572 - AlexNet - INFO - Time: 5.613247 Epoch: 0 Batch: 320 Loss: 6.907755 Avg loss: 7.128911 Accuracy: 0.000000 Avg Accuracy: 0.000998 Top 5 Accuracy: 1.000000
    2018-08-15 07:02:49,181 - AlexNet - INFO - Time: 5.608499 Epoch: 0 Batch: 330 Loss: 6.907755 Avg loss: 7.122229 Accuracy: 0.000000 Avg Accuracy: 0.001015 Top 5 Accuracy: 1.000000
    2018-08-15 07:02:54,788 - AlexNet - INFO - Time: 5.606282 Epoch: 0 Batch: 340 Loss: 6.907755 Avg loss: 7.115940 Accuracy: 0.000000 Avg Accuracy: 0.001031 Top 5 Accuracy: 1.000000
    2018-08-15 07:03:00,391 - AlexNet - INFO - Time: 5.602358 Epoch: 0 Batch: 350 Loss: 6.907755 Avg loss: 7.110008 Accuracy: 0.000000 Avg Accuracy: 0.001046 Top 5 Accuracy: 1.000000
    2018-08-15 07:03:05,982 - AlexNet - INFO - Time: 5.591020 Epoch: 0 Batch: 360 Loss: 6.907755 Avg loss: 7.104406 Accuracy: 0.000000 Avg Accuracy: 0.001039 Top 5 Accuracy: 1.000000
    2018-08-15 07:03:11,627 - AlexNet - INFO - Time: 5.644013 Epoch: 0 Batch: 370 Loss: 6.907755 Avg loss: 7.099105 Accuracy: 0.000000 Avg Accuracy: 0.001053 Top 5 Accuracy: 1.000000
    2018-08-15 07:03:17,242 - AlexNet - INFO - Time: 5.615359 Epoch: 0 Batch: 380 Loss: 6.907755 Avg loss: 7.094083 Accuracy: 0.000000 Avg Accuracy: 0.001025 Top 5 Accuracy: 1.000000
    2018-08-15 07:03:22,849 - AlexNet - INFO - Time: 5.605547 Epoch: 0 Batch: 390 Loss: 6.907755 Avg loss: 7.089318 Accuracy: 0.000000 Avg Accuracy: 0.000999 Top 5 Accuracy: 1.000000
    2018-08-15 07:03:28,513 - AlexNet - INFO - Time: 5.662822 Epoch: 0 Batch: 400 Loss: 6.907755 Avg loss: 7.084790 Accuracy: 0.000000 Avg Accuracy: 0.000974 Top 5 Accuracy: 1.000000
    2018-08-15 07:03:34,161 - AlexNet - INFO - Time: 5.648134 Epoch: 0 Batch: 410 Loss: 6.907755 Avg loss: 7.080482 Accuracy: 0.000000 Avg Accuracy: 0.000969 Top 5 Accuracy: 1.000000
    2018-08-15 07:03:39,775 - AlexNet - INFO - Time: 5.613095 Epoch: 0 Batch: 420 Loss: 6.907755 Avg loss: 7.076380 Accuracy: 0.000000 Avg Accuracy: 0.000984 Top 5 Accuracy: 1.000000
    2018-08-15 07:03:45,409 - AlexNet - INFO - Time: 5.633697 Epoch: 0 Batch: 430 Loss: 6.907755 Avg loss: 7.072467 Accuracy: 0.000000 Avg Accuracy: 0.001015 Top 5 Accuracy: 1.000000
    2018-08-15 07:03:51,062 - AlexNet - INFO - Time: 5.652628 Epoch: 0 Batch: 440 Loss: 6.907755 Avg loss: 7.068732 Accuracy: 0.000000 Avg Accuracy: 0.001027 Top 5 Accuracy: 1.000000
    2018-08-15 07:03:56,674 - AlexNet - INFO - Time: 5.610656 Epoch: 0 Batch: 450 Loss: 6.907755 Avg loss: 7.065163 Accuracy: 0.007812 Avg Accuracy: 0.001022 Top 5 Accuracy: 1.000000
    2018-08-15 07:04:02,290 - AlexNet - INFO - Time: 5.615841 Epoch: 0 Batch: 460 Loss: 6.907755 Avg loss: 7.061748 Accuracy: 0.000000 Avg Accuracy: 0.001000 Top 5 Accuracy: 1.000000
    2018-08-15 07:04:07,927 - AlexNet - INFO - Time: 5.636093 Epoch: 0 Batch: 470 Loss: 6.907755 Avg loss: 7.058479 Accuracy: 0.000000 Avg Accuracy: 0.000995 Top 5 Accuracy: 1.000000
    2018-08-15 07:04:13,552 - AlexNet - INFO - Time: 5.624659 Epoch: 0 Batch: 480 Loss: 6.907755 Avg loss: 7.055345 Accuracy: 0.000000 Avg Accuracy: 0.001007 Top 5 Accuracy: 1.000000
    2018-08-15 07:04:19,171 - AlexNet - INFO - Time: 5.618880 Epoch: 0 Batch: 490 Loss: 6.907755 Avg loss: 7.052340 Accuracy: 0.000000 Avg Accuracy: 0.000987 Top 5 Accuracy: 1.000000
    2018-08-15 07:04:24,746 - AlexNet - INFO - Time: 5.574440 Epoch: 0 Batch: 500 Loss: 6.907755 Avg loss: 7.049454 Accuracy: 0.000000 Avg Accuracy: 0.000998 Top 5 Accuracy: 1.000000
    2018-08-15 07:04:27,896 - AlexNet - INFO - Validation - Accuracy: 0.000000 Top 5 Accuracy: 1.000000

**3. Use L2 regularization methods to penalize the weights for the way they are, in the hope they will be positive, and make standard deviation 0.01.**:

.. code::

    2018-08-15 07:08:41,205 - AlexNet.LSVRC2010 - INFO - There are 1261405 total training images in the dataset
    2018-08-15 07:08:41,411 - AlexNet - INFO - Creating placeholders for graph...
    2018-08-15 07:08:41,423 - AlexNet - INFO - Creating variables for graph...
    2018-08-15 07:08:41,431 - AlexNet - INFO - Initialize hyper parameters...
    2018-08-15 07:08:41,432 - AlexNet - INFO - Building the graph...
    2018-08-15 07:08:55,812 - AlexNet - INFO - Time: 8.745808 Epoch: 0 Batch: 0 Loss: 15.418465 Avg loss: 15.418465 Accuracy: 0.000000 Avg Accuracy: 0.000000 Top 5 Accuracy: 0.000000
    2018-08-15 07:08:58,703 - AlexNet - INFO - Validation - Accuracy: 0.000000 Top 5 Accuracy: 0.007812
    2018-08-15 07:09:02,639 - AlexNet - INFO - Time: 6.825977 Epoch: 0 Batch: 10 Loss: 9.621270 Avg loss: 16.066724 Accuracy: 0.000000 Avg Accuracy: 0.000710 Top 5 Accuracy: 1.000000
    2018-08-15 07:09:05,266 - AlexNet - INFO - Time: 2.626585 Epoch: 0 Batch: 20 Loss: 9.542419 Avg loss: 12.976469 Accuracy: 0.000000 Avg Accuracy: 0.000744 Top 5 Accuracy: 1.000000
    2018-08-15 07:09:07,911 - AlexNet - INFO - Time: 2.644624 Epoch: 0 Batch: 30 Loss: 9.453053 Avg loss: 11.853431 Accuracy: 0.000000 Avg Accuracy: 0.000504 Top 5 Accuracy: 1.000000
    2018-08-15 07:09:10,698 - AlexNet - INFO - Time: 2.785900 Epoch: 0 Batch: 40 Loss: 9.347528 Avg loss: 11.253976 Accuracy: 0.000000 Avg Accuracy: 0.000572 Top 5 Accuracy: 1.000000
    2018-08-15 07:09:14,256 - AlexNet - INFO - Time: 3.557269 Epoch: 0 Batch: 50 Loss: 9.240594 Avg loss: 10.868564 Accuracy: 0.000000 Avg Accuracy: 0.000460 Top 5 Accuracy: 1.000000
    2018-08-15 07:09:17,845 - AlexNet - INFO - Time: 3.588229 Epoch: 0 Batch: 60 Loss: 9.138860 Avg loss: 10.592435 Accuracy: 0.000000 Avg Accuracy: 0.000512 Top 5 Accuracy: 1.000000
    2018-08-15 07:09:21,130 - AlexNet - INFO - Time: 3.284888 Epoch: 0 Batch: 70 Loss: 9.043264 Avg loss: 10.380237 Accuracy: 0.000000 Avg Accuracy: 0.000440 Top 5 Accuracy: 1.000000
    2018-08-15 07:09:24,701 - AlexNet - INFO - Time: 3.570084 Epoch: 0 Batch: 80 Loss: 8.953794 Avg loss: 10.209044 Accuracy: 0.000000 Avg Accuracy: 0.000482 Top 5 Accuracy: 1.000000
    2018-08-15 07:09:28,054 - AlexNet - INFO - Time: 3.351405 Epoch: 0 Batch: 90 Loss: 8.870095 Avg loss: 10.065997 Accuracy: 0.007812 Avg Accuracy: 0.000687 Top 5 Accuracy: 1.000000
    2018-08-15 07:09:31,360 - AlexNet - INFO - Time: 3.295533 Epoch: 0 Batch: 100 Loss: 8.791605 Avg loss: 9.943277 Accuracy: 0.000000 Avg Accuracy: 0.000619 Top 5 Accuracy: 1.000000
    2018-08-15 07:09:35,053 - AlexNet - INFO - Time: 3.692198 Epoch: 0 Batch: 110 Loss: 8.717808 Avg loss: 9.835833 Accuracy: 0.000000 Avg Accuracy: 0.000563 Top 5 Accuracy: 1.000000
    2018-08-15 07:09:38,521 - AlexNet - INFO - Time: 3.467491 Epoch: 0 Batch: 120 Loss: 8.648295 Avg loss: 9.740247 Accuracy: 0.000000 Avg Accuracy: 0.000646 Top 5 Accuracy: 1.000000
    2018-08-15 07:09:42,166 - AlexNet - INFO - Time: 3.644752 Epoch: 0 Batch: 130 Loss: 8.582747 Avg loss: 9.654116 Accuracy: 0.000000 Avg Accuracy: 0.000775 Top 5 Accuracy: 1.000000
    2018-08-15 07:09:45,541 - AlexNet - INFO - Time: 3.374652 Epoch: 0 Batch: 140 Loss: 8.520816 Avg loss: 9.575697 Accuracy: 0.000000 Avg Accuracy: 0.000831 Top 5 Accuracy: 1.000000
    2018-08-15 07:09:49,738 - AlexNet - INFO - Time: 4.194360 Epoch: 0 Batch: 150 Loss: 8.462228 Avg loss: 9.503685 Accuracy: 0.000000 Avg Accuracy: 0.000880 Top 5 Accuracy: 1.000000
    2018-08-15 07:09:53,146 - AlexNet - INFO - Time: 3.407838 Epoch: 0 Batch: 160 Loss: 8.406742 Avg loss: 9.437088 Accuracy: 0.007812 Avg Accuracy: 0.000922 Top 5 Accuracy: 1.000000
    2018-08-15 07:09:56,809 - AlexNet - INFO - Time: 3.662165 Epoch: 0 Batch: 170 Loss: 8.354177 Avg loss: 9.375130 Accuracy: 0.000000 Avg Accuracy: 0.000914 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:00,120 - AlexNet - INFO - Time: 3.309942 Epoch: 0 Batch: 180 Loss: 8.304290 Avg loss: 9.317196 Accuracy: 0.000000 Avg Accuracy: 0.000906 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:04,083 - AlexNet - INFO - Time: 3.962564 Epoch: 0 Batch: 190 Loss: 8.256856 Avg loss: 9.262788 Accuracy: 0.000000 Avg Accuracy: 0.000900 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:07,470 - AlexNet - INFO - Time: 3.386028 Epoch: 0 Batch: 200 Loss: 8.209414 Avg loss: 9.211486 Accuracy: 0.007812 Avg Accuracy: 0.000972 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:10,863 - AlexNet - INFO - Time: 3.392915 Epoch: 0 Batch: 210 Loss: 8.170787 Avg loss: 9.163830 Accuracy: 0.000000 Avg Accuracy: 0.000926 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:14,389 - AlexNet - INFO - Time: 3.524267 Epoch: 0 Batch: 220 Loss: 8.132809 Avg loss: 9.117964 Accuracy: 0.000000 Avg Accuracy: 0.000884 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:18,030 - AlexNet - INFO - Time: 3.640893 Epoch: 0 Batch: 230 Loss: 8.096512 Avg loss: 9.074466 Accuracy: 0.007812 Avg Accuracy: 0.000947 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:21,718 - AlexNet - INFO - Time: 3.687732 Epoch: 0 Batch: 240 Loss: 8.060225 Avg loss: 9.033060 Accuracy: 0.000000 Avg Accuracy: 0.000973 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:24,962 - AlexNet - INFO - Time: 3.242480 Epoch: 0 Batch: 250 Loss: 8.024289 Avg loss: 8.993511 Accuracy: 0.007812 Avg Accuracy: 0.000965 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:28,585 - AlexNet - INFO - Time: 3.622442 Epoch: 0 Batch: 260 Loss: 7.989549 Avg loss: 8.955639 Accuracy: 0.000000 Avg Accuracy: 0.000988 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:31,904 - AlexNet - INFO - Time: 3.318283 Epoch: 0 Batch: 270 Loss: 7.956272 Avg loss: 8.919310 Accuracy: 0.007812 Avg Accuracy: 0.001009 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:35,353 - AlexNet - INFO - Time: 3.448572 Epoch: 0 Batch: 280 Loss: 7.924600 Avg loss: 8.884413 Accuracy: 0.000000 Avg Accuracy: 0.000973 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:39,092 - AlexNet - INFO - Time: 3.737980 Epoch: 0 Batch: 290 Loss: 7.894399 Avg loss: 8.850855 Accuracy: 0.000000 Avg Accuracy: 0.000993 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:42,481 - AlexNet - INFO - Time: 3.388049 Epoch: 0 Batch: 300 Loss: 7.865584 Avg loss: 8.818549 Accuracy: 0.000000 Avg Accuracy: 0.000960 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:46,232 - AlexNet - INFO - Time: 3.751156 Epoch: 0 Batch: 310 Loss: 7.838048 Avg loss: 8.787417 Accuracy: 0.000000 Avg Accuracy: 0.000929 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:49,254 - AlexNet - INFO - Time: 3.021622 Epoch: 0 Batch: 320 Loss: 7.811705 Avg loss: 8.757387 Accuracy: 0.007812 Avg Accuracy: 0.001022 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:52,573 - AlexNet - INFO - Time: 3.317379 Epoch: 0 Batch: 330 Loss: 7.786509 Avg loss: 8.728396 Accuracy: 0.000000 Avg Accuracy: 0.000991 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:56,212 - AlexNet - INFO - Time: 3.637872 Epoch: 0 Batch: 340 Loss: 7.762360 Avg loss: 8.700382 Accuracy: 0.000000 Avg Accuracy: 0.000962 Top 5 Accuracy: 1.000000
    2018-08-15 07:10:59,916 - AlexNet - INFO - Time: 3.703312 Epoch: 0 Batch: 350 Loss: 7.740198 Avg loss: 8.673263 Accuracy: 0.007812 Avg Accuracy: 0.001046 Top 5 Accuracy: 1.000000
    2018-08-15 07:11:03,650 - AlexNet - INFO - Time: 3.733603 Epoch: 0 Batch: 360 Loss: 7.719005 Avg loss: 8.647095 Accuracy: 0.000000 Avg Accuracy: 0.001039 Top 5 Accuracy: 1.000000
    2018-08-15 07:11:07,572 - AlexNet - INFO - Time: 3.921867 Epoch: 0 Batch: 370 Loss: 7.697667 Avg loss: 8.621762 Accuracy: 0.000000 Avg Accuracy: 0.001011 Top 5 Accuracy: 1.000000
    2018-08-15 07:11:11,137 - AlexNet - INFO - Time: 3.564458 Epoch: 0 Batch: 380 Loss: 7.676858 Avg loss: 8.597205 Accuracy: 0.000000 Avg Accuracy: 0.001025 Top 5 Accuracy: 1.000000
    2018-08-15 07:11:14,507 - AlexNet - INFO - Time: 3.369130 Epoch: 0 Batch: 390 Loss: 7.656834 Avg loss: 8.573384 Accuracy: 0.000000 Avg Accuracy: 0.000999 Top 5 Accuracy: 1.000000
    2018-08-15 07:11:18,076 - AlexNet - INFO - Time: 3.562689 Epoch: 0 Batch: 400 Loss: 7.637684 Avg loss: 8.550263 Accuracy: 0.000000 Avg Accuracy: 0.001013 Top 5 Accuracy: 1.000000
    2018-08-15 07:11:21,528 - AlexNet - INFO - Time: 3.452190 Epoch: 0 Batch: 410 Loss: 7.619319 Avg loss: 8.527812 Accuracy: 0.000000 Avg Accuracy: 0.000988 Top 5 Accuracy: 1.000000
    2018-08-15 07:11:24,910 - AlexNet - INFO - Time: 3.380275 Epoch: 0 Batch: 420 Loss: 7.601664 Avg loss: 8.506001 Accuracy: 0.000000 Avg Accuracy: 0.000984 Top 5 Accuracy: 1.000000
    2018-08-15 07:11:28,416 - AlexNet - INFO - Time: 3.505445 Epoch: 0 Batch: 430 Loss: 7.584738 Avg loss: 8.484801 Accuracy: 0.007812 Avg Accuracy: 0.000997 Top 5 Accuracy: 1.000000
    2018-08-15 07:11:32,074 - AlexNet - INFO - Time: 3.657514 Epoch: 0 Batch: 440 Loss: 7.568498 Avg loss: 8.464188 Accuracy: 0.000000 Avg Accuracy: 0.000992 Top 5 Accuracy: 1.000000
    2018-08-15 07:11:35,532 - AlexNet - INFO - Time: 3.457564 Epoch: 0 Batch: 450 Loss: 7.552867 Avg loss: 8.444136 Accuracy: 0.000000 Avg Accuracy: 0.000970 Top 5 Accuracy: 1.000000
    2018-08-15 07:11:39,303 - AlexNet - INFO - Time: 3.770215 Epoch: 0 Batch: 460 Loss: 7.537779 Avg loss: 8.424622 Accuracy: 0.000000 Avg Accuracy: 0.000966 Top 5 Accuracy: 1.000000
    2018-08-15 07:11:42,516 - AlexNet - INFO - Time: 3.211631 Epoch: 0 Batch: 470 Loss: 7.523286 Avg loss: 8.405623 Accuracy: 0.000000 Avg Accuracy: 0.000979 Top 5 Accuracy: 1.000000
    2018-08-15 07:11:45,786 - AlexNet - INFO - Time: 3.268940 Epoch: 0 Batch: 480 Loss: 7.509285 Avg loss: 8.387118 Accuracy: 0.000000 Avg Accuracy: 0.000975 Top 5 Accuracy: 1.000000
    2018-08-15 07:11:49,222 - AlexNet - INFO - Time: 3.434496 Epoch: 0 Batch: 490 Loss: 7.495811 Avg loss: 8.369088 Accuracy: 0.000000 Avg Accuracy: 0.001002 Top 5 Accuracy: 1.000000
    2018-08-15 07:11:52,657 - AlexNet - INFO - Time: 3.434686 Epoch: 0 Batch: 500 Loss: 7.483023 Avg loss: 8.351515 Accuracy: 0.000000 Avg Accuracy: 0.000982 Top 5 Accuracy: 1.000000
    2018-08-15 07:11:56,047 - AlexNet - INFO - Validation - Accuracy: 0.000000 Top 5 Accuracy: 1.000000

The next thing I could think of is to change the **Optimzer**. I was using ``tf.train.AdamOptimizer`` (as it is more recent and it's faster) but the paper is using **Gradient Descent with Momentum**. After changing the optimizer to ``tf.train.MomentumOptimizer`` *only* didn't improve anything.

But when I changed the optimizer to ``tf.train.MomentumOptimizer`` *along with* standard deviation to ``0.01``, things started to change for the better. The top 5 accuracy was no longer ``1.000`` in the initial phase of training when top 1 accuracy was ``0.000``. A lot of positive values can also be seen in the output layer.

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
