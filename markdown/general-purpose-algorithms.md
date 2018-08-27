# General Purpose Algorithms

Machine learning can be thought of as a collection of general purpose algorithms. This is arguably its most powerful trait; one machine learning algorithm can be trained to perform ten different tasks, compared to ten different algorithms each handwritten to solve one of those same ten tasks with traditional programming. This has groundbreaking implications for both speed and productivity of development. It means that small meta-improvements to a machine learning algorithm itself can have a rippling effect, improving the performance of thousands of tasks that it can be trained to solve. This idea of a general purpose algorithm that machine learning provides has the potential to be one of the most radical technologies since the advent of the general purpose computer.

Artificial Neural Networks (ANNs, or simply NNs), one of the most popular family of machine learning algorithms, are perhaps the best example of this. Neural networks act as universal function approximators; Given an infinite number of neurons (units of computation), they can represent any mathematical function or model any real-world occurrence. Let that sink in... Though practically unfeasible, this one fairly straightforward algorithm, can solve any computable problem in the universe.

Before we dig into exactly how this works in the [Neural Networks & Deep Learning](neural-networks-and-deep-learning.html) section, I want to introduce some of the common problem domains that machine learning algorithms function particularly well for. At their core, todays machine learning algorithms excel at various forms of pattern recognition. Given enough data, machine learning algorithms are able to learn the inherent patterns in the data. This is a particularly useful trait, especially for the following types of problems.

## Classification

Classification tasks are perhaps the most traditional application of machine learning. A model that predicts which [discrete](https://stats.stackexchange.com/questions/206/what-is-the-difference-between-discrete-data-and-continuous-data) category, or class, a piece of input data belongs to solves a classification task.<span class="marginal-note" data-info="Classification problems output a probability distribution called a [probability mass function](https://en.wikipedia.org/wiki/Probability_mass_function) over a set of categories (e.g. 22% probability an email is spam, 88% probability an email is legitimate). The probability distribution produced by a classification problem sums to 1.0 (e.g. 0.22 + 0.88 = 1.0) (see [Probability Distributions](probability-distributions.html))."></span> The number of classes varies per problem, but is always greater than one. Examples of classification problems include:

- [Predicting how a company's stock will perform](https://www.microsoft.com/developerblog/2017/12/04/predicting-stock-performance-deep-learning/) (3 classes: high, medium, and low performance)
- [Choosing whether or not an email should be flagged as spam](https://www.codeproject.com/Articles/1232040/Spam-classification-using-Python-and-Keras) (2 classes)
- [Recognizing a handwritten digit](https://js.tensorflow.org/tutorials/mnist.html) (10 classes)
- [Guessing the age of a human using computer vision](https://www.analyticsvidhya.com/blog/2017/06/hands-on-with-deep-learning-solution-for-age-detection-practice-problem/) (100+ classes)
- Predicting the next sample in a raw audio buffer (2048 classes)
- Predicting the next move in a game of chess (10,000+ classes)

## Regression

Rather than outputting a discrete class, regression problems output continuous floating-point numbers.<span class="marginal-note" data-info="Regression problems produce real-valued floats, or rather, a distribution called a [probability density function](https://en.wikipedia.org/wiki/Probability_density_function) over real-valued continuous floats that integrate to 1.0."></span> Examples of regression problems include:

- Predicting the exact price of a stock at the end of the month
- Predicting the change in earth's temperature each year
- Predicting the amount of money an insured person will claim each year

## Translation and Transcription

Machine translation applications have seen a huge boost in performance correlated with development of ML algorithms. This task necessitates the translation of arbitrary data from one form to another:

- [Translation from English to German](https://www.tensorflow.org/tutorials/seq2seq)
- Translation from Tweets to Facebook posts
- Translation from LaTeX math formulas to plotted 2D functions
- [Translations from a video of one person to another](https://twitter.com/brannondorsey/status/808461108881268736)<span class="marginal-note" data-info="This was actually our first machine learning experiment. Researchers at UC Berkeley recently published a much more convincing version of a similar task [here](https://www.youtube.com/watch?v=PCBTZh41Ris)."></span>
- Translation from sentences to Emoji
- [Transcription from audio recording to text](https://medium.com/@ageitgey/machine-learning-is-fun-part-6-how-to-do-speech-recognition-with-deep-learning-28293c162f7a) (speech-to-text)

## Anomaly Detection

ML models are exceptionally good at detecting irregularities in patterns. In these situations the number of positive examples (irregularity) is far fewer than negative examples (common data). They are often used to:

- [Detect credit card fraud](https://github.com/ellisvalentiner/credit-card-fraud)
- [Detect malicious traffic on a computer network](https://securityintelligence.com/applying-machine-learning-to-improve-your-intrusion-detection-system/)
- Detect irregular behavior on the ground using aerial photography
- [Diagnose rare diseases](http://file.scirp.org/pdf/JILSA_2017012413273284.pdf)

## Synthesis and Sampling

ML models that synthesize new data based on example data are called [generative models](https://towardsdatascience.com/deep-generative-models-25ab2821afd3). They can be used to:

- Create 3D assets for a procedurally generated video game
- Create real-time generative music that plays forever
- [Hallucinate a video from still images](https://www.theverge.com/2016/9/12/12886698/machine-learning-video-image-prediction-mit)
- Apply the style from one image to the content of another ([style transfer](https://towardsdatascience.com/artistic-style-transfer-b7566a216431))

## De-noising

De-noising problems involve removing artifacts and noise from data in order to produce a clean sample. De-noising models are often trained by artificially introducing noise to otherwise clean samples in order to learn de-noising patterns. A trained model can then apply what its learned from the synthesized data to noisy data found in the real-world. Examples include:

- Removing noise from a weak audio or video signal
- [Coloring black-and-white images](https://hackernoon.com/remastering-classic-films-in-tensorflow-with-pix2pix-f4d551fa0503)
- Recovering missing information from redacted documents

Note that the above list of common machine learning problem domains is far from extensive. Rather, it is a small collection of a few of the tasks that ML-based approaches are dependably good at solving. Many tasks also fall into a few of these categories at once.

Next: [Data is Key](data-is-key.html)<br>
Previous: [What is Machine Learning](what-is-machine-learning.html)