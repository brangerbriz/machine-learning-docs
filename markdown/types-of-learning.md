# Types of learning

There are two general categories that ML algorithms usually fall into: supervised and unsupervised.

## Supervised

Supervised learning relies on labeled data to train a model. For each piece of input data `X` in a training set, there exists a corresponding labeled output data `y`. For example, a picture of a truck will have the corresponding class label, "truck". This type of data is very easy to learn from and it is for this reason most traditional machine learning algorithms are supervised. You can think of supervised machine learning as analogous to training an animal. During training, the model is rewarded for correct predictions of `y` and punished for incorrect predictions of `y`. If you have labeled data, supervised learning is certainty the way to go. Common downfalls with supervised learning is that labeled data is often scarce. There is far more unlabeled data in the world than labeled data and labeling can often be cost- or time-prohibitive.

One common misconception newcomers often have about supervised learning is that it requires human supervision in some way. While this is often true, as in the case of using a dataset that has been hand-labeled, many supervised learning algorithms leverage data that is "self-labeled". Take for example a [character-level language modeling](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) task that attempts to predict the next letter in a sentence given the previous five letters. To predict the string "hi there!", an input of "hi th" should output "e", and "i the", should output "r", etc... We could train such a model on the entire 27 billion English characters in Wikipedia without the need for any human supervision. This remains a supervised learning task, because each piece of input data has a corresponding output data that is labeled.

In situations where data is not self-labeling, labeling it can be expensive. It is for this reason that recent research initiatives have put more resources towards developing better unsupervised learning algorithms.

## Unsupervised

Unsupervised learning algorithms learn from unlabeled data. This is particularly advantageous because unlabeled data is cheap and abundant. New developments in unsupervised learning have the potential to make some of the most radical advances for the ML field in the coming decades. Clustering algorithms like [K-Nearest Neighbor](knn.html) (k-NN) and [t-SNE](t-sne.html) both attempt to group unlabeled data into classes or geometric regions. Both of these methods are incredibly helpful processes used to [learn about your data](learning-about-your-data.html).

[Generative Adversarial Networks (GANs)](network-types.html) use a novel model dichotomy that pits networks against each-other in order to strengthen both of them. Most of the example algorithms in this document will be supervised in nature, however, we will also make an attempt to cover the basics of unsupervised learning.

Next: [Types of Tasks](types-of-tasks.html)