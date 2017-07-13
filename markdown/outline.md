# Intro

## Machine Learning?
- What is machine learning?
	- For hard problems, don't write rules (heuristics), instead learn from data
	- Easier to do so as long as you have the data, and it generally works better
- What kind of problems is it good at solving? When and when not to use.
- What skills does it require?
	- Basic scripting skills, usually in python
	- General comfortability with:
		- Linear Algebra
		- Statistics
		- Probability 
		- I want to stress that these are more avenues of further research in order to improve your ml skillz. You don't have to be know much about them to get started.
- A brief history (so brief, like 1-2 paragraphs. Mainly to contextualize the current hype.)
	- Data explosion (big data)
	- GPGPU
	- Previous AI winters (beware the hype)

## General Purpose Algorithms

- Machine learning techniques are general purpose. The same ML architecture/algorithm can be used to:
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

- Models can only be as good as the 
- The bottleneck in an ML pipeline is in access to high-quantities of high-quality data

## Models

- A model is a synthetic representation of the way something in the real world works.
- Models are:
    - Trained/learned.
    - The product of machine learning.
    - The algorithm.
    - Function approximators.

## Machine Learning Pipeline

Get data -> process/clean data -> train model <-> evaluate model -> deploy model

## Types of learning: Supervised vs Unsupervised

- Supervised:
	- Labeled data.
    - Easiest to learn from.
    - Scarce.
    - Expensive (often, but not always).

- Unsupervised:
    - Unlabeled data.
    - Abundant.
	- Cheap.
	- Surprising.
	- Advancements in unsupervised
	- learning have potential to
	- make the most impact.

# Types of Tasks: Classification vs Regression

- Classification:
    - Discrete.
	- Produces a label (or class).
	- Example: Type of car, 

- Regression:
	- Continuous.
	- Produces a number, or series of numbers.
	- Example: Audio waveforms, global GDP, etc… (any kind of time series data)

## Linear Regression

-  The "Hello, World!" of machine learning
- mention svms and decision trees

## Neural Networks and Deep Learning

- Multi-layer perceptron (1960s style)
- Super brief overview of a single-layer feed forward MLP
	- Input, multiply weights, add biases, activation function, output (repeat per layer)
- Activation functions bend otherwise liner models
- What is an architecture? How does it relate to a model?

## Let's build an Machine Learning Pipeline

## Data Aquisition 

## Design Architecture (your model architecture has a relationship to the way we process the data)

## Data Pre-processing

## Training

## Evaluation

## Tweak

##



## Regularization 

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

## Frameworks/tools

- wekinator
- ipython
- keras
- tensorflow
- nvidia-docker

## Projects

- char-rnn
- knn/t-sne
- GloVe experiments
- midi-rnn
- ML4MusicWorkshop
- Python Notebooks

## Resources
