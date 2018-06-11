# Guides

I've created several guides to include in this resource. These guides, and their reference code, take on the form of:

1) IPython/Jupyter Notebooks
2) [Tensorflow.js](http://js.tensorflow.org/) examples using [Electron](https://electronjs.org/)
3) High-level tutorials that make use of multiple languages and frameworks

The machine learning library landscape has changed since we began writing these docs and Tensorflow.js has now emerged as a practical library for building and deploying machine learning models using web technologies. I began writing these tutorials using IPython + Keras but have now added several Tensorflow.js + Electron examples as well. Luckily, Python + JavaScript are very similar, and the Tensorflow.js Layers API was modeled after Keras, so it should be pretty easy to bounce back and forth between examples in either language.

You will also notice that although the frameworks and languages may differ, the general process looks similar between tutorials. That is because the [machine learning pipeline](the-ml-pipeline.html) is rather standard and looks similar between different tasks. Most machine learning algorithms require:

1) Data acquisition and Preprocessing
2) Model training
3) Model evaluation (and usually more model training)
4) Model inference and deployment

Whether those basic steps present themselves as a single python script, separate `train.sh` and `predict.sh` BASH scripts, or some other manifestation, rest assured that the basic principles between different machine learning pipelines, examples, and research projects are likely very similar. 

## Preprocessing Data

 - [Pokemon classification](https://github.com/brangerbriz/ml-notebooks/blob/master/notebooks/preprocessing.ipynb) (IPython): Here you will learn about normalizing input data and feature standardization. For this classification task we are attempting to classify Pokemon (grass, water, etc.) given 6 statistics like HP, Attack, Speed, etc. as features. This task was a weekly coding challenge from Siraj Raval's [dataset preparation video](https://www.youtube.com/watch?v=0xVqLJe9_CY). The 5-7 participants that submitted results seemed to have a classification accuracy from ~14% to 75% with a mode of ~30%. We achieved similar results.
 - [Human activity classification using mobile accelerometer data](https://github.com/brangerbriz/ml-notebooks/blob/master/notebooks/preprocessing_human_activity_classification.ipynb) (IPython): This example revisits the topic of normalization and standardization with more visual examples showing what these transformations "look" like. Here we train a recurrent neural network (RNN) to predict human activities like walking, standing, and sitting using windows of temporal accelerometer sensor data. We reach ~75% classification accuracy!

## [Clustering and Dimensionality Reduction](https://github.com/brangerbriz/ml-notebooks/blob/master/notebooks/clustering_and_dimensionality_reduction.ipynb) (IPython)

This example demonstrates unsupervised learning using k-means clustering to group similar english words together using [word embeddings](https://nlp.stanford.edu/projects/glove/).

## [Hyperparameter search](https://github.com/brangerbriz/ml-notebooks/blob/master/notebooks/hyperparameter_search.ipynb) (IPython)

Training neural networks can be very difficult and it is often the role of the programmer to select the hyperparameters that will yield the best results. It is not uncommon for a programmer to run dozens of experiments, each with different hyperparameters, before arriving at a a "good enough" solution.

Hyperparameter search (often called Hyperparameter Optimization) is a method used to automate the discovery of effective hyperparameters for a network. Rather than using intuition and experience to fine-tune your hyperparameters by hand, hyperparameter search can be used to automatically discover optimal hyperparameters given enough compute time and resources.

## [Handwriting classification in browser](https://github.com/brangerbriz/tf-electron/blob/master/examples/mnist/index.js) (Tensorflow.js)

I've [added extensive code comments](https://github.com/tensorflow/tfjs-examples/pull/92) to the official MNIST handwriting classification in Tensorflow.js. I've also wrapped this example in our [tf-electron](https://github.com/brangerbriz/tf-electron) repository so you can run it as a desktop app using Electron.

Just clone the repo, install the dependencies, and view the `examples/mnist` page in the desktop app:

```
git clone https://github.com/brangerbriz/tf-electron
cd tf-electron

# install npm dependencies
npm install

# start the electron app
npm start
```

## [Text generation in browser](https://github.com/brangerbriz/tf-electron/blob/master/examples/char-rnn/js/main.js) (Tensorflow.js)

[Char-rnn](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) is an effective method of character-level text generation popularized by Andrej Karpathy. While it struggles to maintain long-term dependence and structure, given enough input text, it can learn to create new text that appears to be written in the style of an author, even if it's output is somewhat nonsensical. 

Ported the [Keras char-rnn example](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py) to Tensorflow.js so that it can be used in the browser, or with Electron. To try it out yourself, download and run the tf-electron repo. Once you've got it open, navigate to `examples/char-rnn` and open your developer console to see what's happening. This example trains and runs an RNN on 1MB of text from shakespeare plays.

```
git clone https://github.com/brangerbriz/tf-electron
cd tf-electron

# install npm dependencies
npm install

# start the electron app
npm start
```

## [Real-time pose estimation](https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5) (Tensorflow.js)

This high-level tutorial was written by Dan Oved. Dan ported the PoseNet model to the browser and created a simple API to interface with it. This is a nice example of how an ML model can be abstracted into a user-friendly library.

## [Simple linear network](https://github.com/brangerbriz/tf-electron/blob/master/examples/simple-linear-network/main.js) (Tensorflow.js)

This example borrows from [a Tensorflow.js intro tutorial](https://medium.com/tensorflow/getting-started-with-tensorflow-js-50f6783489b2) to train a linear one neuron network to learn the function `y = (x) => 2 * x -1`. The real function is used to produce 100 training samples that are learned by the model for 100 epochs.  
 