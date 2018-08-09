# Regularization

Regularization is used to reducing model capacity, and therefore mitigate against overfitting. The idea behind regularization is to impose constraints on the model that make it more difficult for it to learn easy solutions that apply to the training set only and that don't generalize well to unseen data. There are several common techniques used to regularize a model. Try them if you find that your model is performing well on your training data, but not on your test or real-world data.

## Early Stopping

[Early stopping](https://en.wikipedia.org/wiki/Early_stopping) the most common form of regularization, although it may not even seem like regularization at all.You will use this technique in nearly all model training experiments that you create. Early stopping designates when you quit training your model, choosing to freeze the trained weights for model inference. In practice, you usually choose to stop training once the validation loss has stopped decreasing. The training loss will likely continue to decrease through training data memorization, but once you notice the validation loss is no longer decreasing you should stop training. Failing to do so can actually cause your validation loss to increase over time as the model begins to memorize the training data. Early stopping is considered regularization because you are reducing model capacity (in this case by limiting compute time).

## Dropout

[Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) is one of the most popular regularization techniques. It involves randomly disabling, or turning off, a subset of neurons at a layer during training. This causes each neuron to have to learn more to make up for its disabled neighbors. Once the model is trained, dropout is not used and all neurons remain on during inference and prediction.

## Batch Normalization

[Batch Normalization](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c) subtracts the mean and divides by the std dev for each mini-batch of input data.

## Gradient Clipping
[Gradient clipping](https://hackernoon.com/gradient-clipping-57f04f0adae) is a method to combat the *exploding gradients problem* where large gradient values multiply throughout the network in a destructive fashion. If you notice that the parameter gradients that are being back-propagated through the network during optimization have large magnitudes, this method may help to fix that. 

## Multiple Objectives

One particularly interesting method that I've heard of to regularize a model is to make your loss value the sum of two independent objective functions. If you are trying to build a classification system that identifies cat images, you could try and regularize it by adding a separate objective of trying to classify indoor/outdoor images, using a loss function that is the combination of the two objectives: `L = 0.8 * CAT_LOSS + 0.2 * INDOOR_LOSS`. This is just an example, and I have no idea if this particular use case would actually be helpful, but the idea is that by optimizing for multiple objectives simultaneously the model capacity becomes restricted and encouraged to learn patterns that are useful to both objectives.

Previous: [Normalization and Preprocessing](normalization-and-preprocessing.html)