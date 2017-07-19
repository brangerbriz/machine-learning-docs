# Intro

## Machine Learning?
- What is machine learning?
	- For hard problems, don't write rules (heuristics), instead learn from data
	- Easier to do so as long as you have the data, and it generally works better
- What kind of problems is it good at solving? When and when not to use.
	- Traditional programming is better for simple, well-described tasks that can be easily expressed in software.
	- Candidate tasks for machine learning are generally the tasks whose solutions are hard to explain to others.
		- Natural Language Processing
		- Computer Vision (classification)
		- Speech-to-text, text-to-speech
		- etc...

- What skills does it require?
	- Basic scripting skills, usually in python
	- General comfortability with:
		- Linear Algebra
		- Statistics
		- Probability
 		- I want to stress that these are more avenues of further research in order to improve your ml skillz. You don't have to be know much about them to get started.
 	- Note that writing software for machine learning is indeed very strange. Programs are often very short (< 100 lines of code), but increadibly dense and assume a significant amount of prior domain-knowledge to read. 

- A brief history (so brief, like 1-2 paragraphs. Mainly to contextualize the current hype.)
	- Data explosion (big data)
	- GPGPU
	- Previous AI winters (beware the hype)

## General Purpose Algorithms
- Machine learning techniques are general purpose. The same ML architecture/algorithm can be used to:
	- General
		- Pattern recognition of all types (DLB p. 97)
		- Classification
		- Classification w/ missing inputs
		- Regression
		- Transcription
		- Machine Translation
		- ~~Structured Output~~
		- Anomaly detection
		- Synthesis and sampling
		- ~~Imputation of missing values~~
		- Denoising
		- ~~Density / probability mass function estimation~~
	- Specific
	 	- Filter your email spam.
	 	- Predict your liability for an insurance company.
		- Write new content in the style of a dead author.
		- Sell you sh*t on the internet.
		- Generate millions of new songs.
		- Predict stock prices.
		- Produce spam email.
		- Predict the effects of global warming.
		- Provide an “understanding” of an environment than can be used to drive an autonomous vehicle.

- Universal function approximators (what an insanely powerful idea)
- Has the potential to be as big as general purpose computers

## Data is 🔑

- Models can only be as good as the data that is used to train them. 
- The bottleneck in an ML pipeline is in access to high-quantities of high-quality data.
- The way you represent your data is as import, if not more important, than the raw data itself. 
	- Example
- Data quantities
	- When you are working w/ a machine learning system, it is often very helpful to think in orders-of-magnitude and exponentials instead of linearly. Twice as much data will likely not significantly increase your accuracy/performance, but 10x as much data probably will. (not sure if this is the right location for this. But this is a super helpful point that it took me a while to realize and internalize).
	- There are rules of thumb about how much data is enough, but the answer is always more data. A common solution to many problems in machine learning is "get more data".
- Training and test data (maybe mention validation split, or maybe hold that for a practical example later below)

## Models

- A model is a synthetic representation of the way something in the real world works.
	- There is a function that describes everything non-random. Knowing this true real-world function is impossible without knowing every example of it (past and future). But extimating/approximating these functions can be done if you have enough examples of the function. The extimation function is called a model.
- Model vs architecure
- Models are:
    - Trained/learned.
    - The product of machine learning.
    - The algorithm.
    - Function approximators.
- Difference between training a model and using it in production (testing)

## Machine Learning Pipeline

- Get data -> process/clean data -> train model <-> evaluate model -> deploy model
- Explain importance of seperating data from model when authoring ml software. It is a great practice/paradigm once you understand it, and many deep learning frameworks require it (tensorflow).

## Types of learning: Supervised vs Unsupervised

- Supervised:
	- Labeled data.
    - Easiest to learn from.
    - Scarce.
    - Expensive (often, but not always).
    - Classic machine learning. We've been doing this well for decades. But now we are doing it REALLY well.

- Unsupervised:
    - Unlabeled data.
    - Abundant.
	- Cheap.
	- Surprising.
	- Advancements in unsupervised learning are hot right now.
	- learning have potential to make the most impact.
	- GANs as example

# Types of Tasks: Classification vs Regression

- Classification:
    - Discrete.
	- Produces a label (or class).
	- Example: Type of car, 

- Regression:
	- Continuous.
	- Produces a number, or series of numbers.
	- Example: Audio waveforms, global GDP, etc… (any kind of time series data)

## Performance Measures

- Method of evaluation is increadibly important. Not just for you, with supervised learning this is how the machine knows what type of good behavior to reward and what type to punish.
- Cost/Loss function
- Categorical accuracy (cross-entropy), F-score, vs MSE and Maximum Log Likelyhood

## Linear Regression

- The "Hello, World!" of machine learning
- Multi-deminsional space. If we express our data (any data) as real-valued multi-dimensional tensor vectors, we can perform geometry on it.
- Gradient descent and derivatives (links to 3Blue1Brown)
- ~~mention svms and decision trees~~

## Neural Networks and Deep Learning

- Multi-layer perceptron (1960s style)
- Super brief overview of a single-layer feed forward MLP
	- Input, multiply weights, add biases, activation function, output (repeat per layer)
	- should this also include the anatomy of a neural network?
		- activation functions
- Layers represent heirarchy of information
- Activation functions bend otherwise liner models

## Features, Design Matrices, and Tensors

- What are features?
 	- Iris dataset
 	- Pixels in an image (mnist)
 	- Feature = Input
- Tensors (having trouble figuring out where to put this, but here seems... ok?)
	- scalars
	- vectors
	- matrices
	- tensors

- Features are often hand chosen, especially in traditional machine learning pipelines. However there is much contemporary research that encourages against hand-selected features. Biases are introduced based on assumptions of what is (and is not) useful to the learning system. But results are often better if the human doesn't make the choice for the model, but rather the model decides for itself what is and is-not helpful to know about.

- Features should have low co-variance.
	- Use some co-variance matrices as an example

## __Should a full wekinator demo be here, before code below? How do I fit in wekinator?__ 

## "Hello, World!" Machine Learning Pipeline

- MNIST
- Includes brief code examples (to introduce numpy and keras)
- This section is predominently used to connect ML ideas to their code implementations.

### Data Acquisition

- Download the MNIST dataset (don't use the one bundled with keras, because we wan't to illustrate this process)

### Data Split

- Training data
- Validation/dev data
- Testing data

### Data Pre-processing and Feature Selection

- Normalization
	- Zero mean and unit variance
- Little co-variance between features

### Design Architecture

- Model capacity and overfitting
- Wide
	- easier to train
- Deep
	- much harder to train, but if you do it right it is known to generalize better
- Try a bunch of architectures and move towards the ones that work. Best results in deep learning are found emperically. This holds true for all kinds of things: activation functions, optimizers, etc...

### Training

- Training error vs validation error
- Checkpoints
- Early stopping (return to this in regularization)

### Evaluation and Tweaks

- Are you underfitting or overfitting.
- Adjust in orders of magnitude.
- Only change one thing between experiments. Changing more introduces ambiguity in what caused the results.

### Using your trained model (Deploy (train->test))

- Just like training, but without the weight update (backprop)
	- But sometimes with extra steps. Like with auto-regression.
- How you sample matters
	- Greedy argmax :/
	- Sample from output distrobution :)

## Troubleshooting/debugging ML pipelines

- Debugging an ML pipeline is harder than traditional software debugging because there are more places the problem could lie. First try to identify if the problem is:
	- With the data, or data representation
	- In your code implementation (error in your python logic for instance)
	- A flaw in your model architecture or graph

## Going Deeper

### Network Types

- Vanilla NN
- Convolutional Neural Networks
- Recurrent Neural Networks
- Maybe? (likely explain and then link to good resources)
	- Autoencoders
	- GANs (this isn't a network so much as a method of training multiple networks of arbitrary architecture)

### Latent Space

- What is it? Why is it powerful?
	- Word2Vec as example
- Each layer in a NN is a linear transformation in a latent space. 

### Measuring Distance

- Why is it important?
	- if you know the distance two things are away from eachother, you know something about their relationship
- Euclidean distance
- Cosine distance (often times you should go with cosine)
- Link to 🔥 fast 🔥 search algos like facebook's new sheit

### Learning about your Data

- KNearestNeighbor
- t-SNE

### Regularization

- meant to reducing model capacity
- Batch normalization
- Gradient clipping (vanishing/exploading gradients)
- Dropout
	- touch on ensemble learning
- Early stopping
	- yes this is a form of regularization, because you are reducing model capacity (in this case by limiting compute time)
- Anorthodox stuff
	- multiple objective learning (can't remember if that is the right name)

### Optimizers

	- SGD
	- Adam
	- RMSProp
	- etc...
	- Talk about momentum (maybe too specific)

## Activation Functions

- linear (lol)
- sigmoid
- tanh
- relu
- etc...

### Feature Learning

- Use an autoencoder. Grab the latent embedding weights and use it as the first layer in a classification task.

### Hyperparameter Search

- Let the machine do the work

### Experiment structure

- Super important! Most of the code written for an ML pipeline is more about managing your experiments in a way where you stay sane. A good ML library will do the heavy lifting (don't implement stuff yourself, it will take too long and you will do it wrong. Trust me.)
- Show an example folder structure

## BB Projects

- char-rnn
- knn/t-sne
- GloVe experiments
- midi-rnn
- ML4MusicWorkshop
- Python Notebooks

## Apendix 

- GPU training w/ CUDA and nvidia-docker

## Glossary of terms

- Hyperparameters
- Hyperparameter search
- Autoencoder
- Convolutional Neural Network
- Recurrent Neural Network
- LSTM
- GRU
- Backpropagation
- Gradient Descent
- Training Loss
- Validation Loss
- Training data
- Validation data
- Test data
- Auto-regressive
- Dropout
- Regularization
- Model Capacity
- Overfitting
- Underfitting
- Supervised Learning
- Unsupervised Learning
- GANs
- Adversarial Learning
- Generalization

## Frameworks/tools

- wekinator
- ipython
- keras
- tensorflow
- nvidia-docker

## Resources

- links
