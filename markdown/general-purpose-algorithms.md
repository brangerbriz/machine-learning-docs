# General Purpose Algorithms

Machine learning can be thought of as a collection of general purpose algorithms. This is arguably its most powerful trait; one machine learning algorithm can be trained to perform ten different tasks, compared to ten different algorithms each handwritten to solve one of those same ten tasks with traditional programming. This has groundbreaking implications for both speed and productivity of development. It means that small meta-improvements to a machine learning algorithm itself can have a rippling effect, improving the performance of thousands of tasks that it can be trained to solve. This idea of a general purpose algorithm that machine learning provides has the potential to be one of the most radical technology since the advent of the general purpose computer.

Artificial Neural Networks (ANNs, or simply NNs), one of the most popular family of machine learning algorithms, are perhaps the best example of this. Neural networks function as universal function approximators; Given an infinite number of neurons (units of computation), they can represent any mathematical function or [model](machine-learning-models.html) any real-world occurrence. Let that sink in... Though practically unfeasible, this one fairly straightforward algorithm, can solve any computable problem in the universe.

Before we [dig into](neural-networks-and-deep-learning.html) exactly how this works, I want to introduce some of the common problem domains that machine learning algorithms function particularly well for. At their core, todays machine learning algorithms excel most at various forms of pattern recognition. Given enough data, machine learning algorithms are able to learn the inherent patterns in the data. This is a particularly useful trait, especially for the following types of problems.

## Classification

Classification tasks are perhaps the most traditional application of machine learning. A model that predicts which [discrete](https://stats.stackexchange.com/questions/206/what-is-the-difference-between-discrete-data-and-continuous-data) category a piece of input data belongs to solves a classification task. The number of classes varies per problem, but is always greater than one. Examples of classification problems include:

- Predicting if a stock will be higher or lower by the end of the month (2 classes)
- Choosing whether or not an email should be flagged as spam (2 classes)
- Recognizing a handwritten digit (10 classes)
- Guessing the age of a human using computer vision (100+ classes)
- Predicting the next sample in a raw audio buffer (2048 classes)
- Predicting the next move in a game of chess (10,000+ classes)

## Regression

Rather than outputting a specific class, regression problems output [continuous](discrete-vs-continuous-data.html) floating-point numbers. Examples of regression problems include:

- Predicting the exact price of a stock at the end of the month
- Predicting the change in earth's temperature each year
- Predicting the amount of money an insured person will claim each year

## Translation and Transcription

Machine translation applications have seen a huge boost in performance correlated with development of ML algorithms. This task necessitates the translation of arbitrary data from one form to another:

- Translation from English to Spanish
- Translation from Tweets to Facebook posts
- Translation from LaTeX math formulas to plotted 2D functions
- Translation from sentences to Emoji
- Transcription from audio recording to text (speech-to-text)

## Anomaly Detection

ML models are exceptionally good at detecting irregularities in patterns. In these situations the number of positive examples (irregularity) is far fewer than negative examples (common data). They are often used to:

- Detect credit card fraud
- Detect malicious traffic on a computer network
- Detect irregular behavior on the ground using aerial photography
- Diagnose rare diseases

## Synthesis and Sampling

ML models that synthesize new data based on example data are called [generative models](https://towardsdatascience.com/deep-generative-models-25ab2821afd3). They can be used to:

- Create 3D assets for a procedurally generated video game
- Create real-time generative music that plays forever
- [Hallucinate a video from still images](https://www.theverge.com/2016/9/12/12886698/machine-learning-video-image-prediction-mit)
- Apply the style from one image to the content of another ([style transfer](https://towardsdatascience.com/artistic-style-transfer-b7566a216431))

## De-noising

De-noising problems involve removing artifacts and noise from data in order to produce a clean sample. De-noising models are often trained by artificially introducing noise to otherwise clean samples in order to learn de-noising patterns. A trained model can then apply what its learned from the synthesized data to noisy data found in the real-world. Examples include:

- Removing noise from a weak audio or video signal
- Re-coloring faded photographs
- Recovering missing information from redacted documents

Note that the above list of common machine learning problem domains is far from extensive. Rather, it is a small collection of a few of the tasks that ML-based approaches are dependably good at solving. Many tasks also fall into a few of these categories at once.

Next: [Data is Key](data-is-key.html)
