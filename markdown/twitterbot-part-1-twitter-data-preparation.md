# Twitterbot Part 1: Twitter Data Preparation

## Introduction

In this technical four-part tutorial series we will create a character-level<span class="marginal-note" data-info="We'll create a model inspired by [Andrej Karpathy's Infamous Char-RNN](https://karpathy.github.io/2015/05/21/rnn-effectiveness/), with a few tweaks and additions introduced by [YuXuan Tay](https://github.com/yxtay/char-rnn-text-generation)."></span> text-generation model that can be trained to output synthetic tweets in the style of a particular twitter user. We'll walk through the entire machine learning pipeline (see [The ML Pipeline](the-ml-pipeline.html)): from gathering data and training models in python, to automating the hyperparameter search process and deploying client-side models in the browser using JavaScript and Tensorflow.js. By the time we're done you should have an understanding of what it takes to develop an ML solution to a real-world problem, as well as some working code that you can use to train your own twitter bots. 

To make this project a bit more interesting, let's add some constraints. **Given only a user's public twitter handle, let's create an ML model that can generate new tweets in the style of that user, using relatively-short model training times and consumer-grade hardware like laptops or even smart phones.**

These constraints will allow us to explore some creative and practical solutions to our problem that may be helpful in future ML tasks as well. Relying on "consumer-grade" hardware also frees us 💸 from the leash of GPU/TPU accelerated cloud computing, and encourages us to develop a resource-limited solution to an otherwise unbounded problem. In other words, we aren't looking to find the ultimate state-of-the-art RNN text-generation solution for the tasks of automated tweet generation, but rather, we are looking for a practical "good enough" solution with limited twitter user data and compute resources. Let's get started!

## Data Challenges

Deep learning models like RNNs usually require lots of data before they begin to perform well. Anywhere from tens or hundreds of megabytes to many gigabytes of data depending on the network architecture, data representation, and task. The problem is that most twitter users probably haven't even generated a single megabyte worth of tweets (one million characters), a dataset size that Andrej Karpathy himself claims is "very small", and even if they have, Twitter's API restricts tweet downloads to a mere [3,200 per user](https://developer.twitter.com/en/docs/tweets/timelines/api-reference/get-statuses-user_timeline.html).

This poses a difficult problem. We have a data-hungry RNN algorithm but only very limited access to training data for a specific twitter user. With too little data, our model will likely underfit and not be able to produce text that looks like a tweet, or even intelligible english. On the other hand, if we are able to train a model using only a few thousand tweets without underfitting, we'd likely have to train it for tens or hundreds of epochs which could lead to dramatic overfitting, or memorization of the training data; It may start to output specific lines from the training data but it wouldn't be able to generalize twitter-like patterns, memes, or idioms like @mentions and RTs.

Wouldn't it be nice if we could somehow leverage the combined data from millions of different twitter users instead of only the few tweets that we are afforded from a specific user that we are attempting to model? Fortunately for us, we can use a technique called transfer learning to do just that! We'll train a base model that learns to produce generic tweets in the ethos of thousands of combined tweeters. We'll then fine-tune this pre-trained base model using the sparse twitter data we have access to for the specific twitter user we intend to model.

## Data Download & Preparation

