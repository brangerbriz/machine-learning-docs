# Machine Learning Pipeline

<section class="media" data-fullwidth="false">
    <img src="images/ml-pipeline.png" alt="The machine learning pipeline">
</section>

## Get Data

The first step in any machine learning pipeline is data acquisition. There are generally four methods of obtaining data:

- Use a [publicly available dataset](https://github.com/caesar0301/awesome-public-datasets)
- Scrape data from a publicly accessible API or service
- Use a private dataset you or your company has collected
- Design and deploy a method for collecting new data

The first three are fairly similar. The last provides the power to design custom features with the trade-off of money and time.

## Clean and Pre-process Data

The data preparation, preprocessing and normalization
<span class="marginal-note" data-info="[Machinelearningmastery.com](https://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/) has a great blog post on how to prepare data for machine learning. In general, I've found this website to be a very helpful resource, especially when it comes to code examples."></span>
 steps are often the most time consuming and code-heavy part of the pipeline (see [Preprocessing & Normalization](normalization-and-preprocessing.html)). It's also the most important step to get right. Most machine learning algorithms only operate well on data that is within a certain range (e.g. small values between `-1.0` and `1.0` or `0.0` and `1.0`) with a standard variance. Real-world data might have arbitrary values with units in the millions, or tons of outliers that need to be removed. Once you've cleaned and pre-processed your data, you will need split it into groups to be used for training and testing. Correctly preparing and partitioning your data before using it for training is often the key to developing a successful model.

## Train Your Model

Now comes the fun part, its time to train your model (using your training data only).
<span class="marginal-note" data-info="[Here](https://elitedatascience.com/model-training) is a nice short tutorial that gives an alternative overview to model training, and the ML pipeline in general."></span>
 Training can take anywhere from a few minutes to hours or days depending on your model architecture or compute hardware. As a rule of thumb, training your model on a GPU will yield training times orders of magnitude faster than CPU training and should almost always be preferenced.

During the training process, your data is iteratively fed into the model in small batches, subsets of the entire training dataset, called mini-batches. Once your model has seen all of the training data, it has completed one epoch. Training usually requires multiple epochs and ends when [validation loss](https://bit.ly/1CHXsNH) (error) stops decreasing. More on that soon...

## Evaluate Your model

Once you've trained your model, its time to evaluate its performance (see [Performance Measures](performance-measures.html)). You usually do this by freezing the model parameters, running the model on the test data, but this time without updating the parameters like you do during training. During evaluation the goal is to measure the trained model's error/accuracy on unseen data. Once you've got a measure of how effective your model is, its time to train another model to try and beat it. The ML pipeline often requires many iterations of training and evaluating; the goal being to reduce the error on your test data.
<span class="marginal-note" data-info="Here are a few things to consider when considering what to change to improve your model's performance: 1) Are you underfitting or overfitting? 2) Adjust values in orders of magnitude (value, value * 10, value * 100, etc) when you are trying to find the right value for something like a hyperparameter, you will cover ground more quickly. 3) Only change one thing at a time between experiments. Changing more introduces ambiguity in what caused the results. Additionally, [This blog post](https://machinelearningmastery.com/improve-deep-learning-performance/) has some good pointers for improving deep learning performance."></span>
It is not uncommon to do this `10` to `100+` times, choosing to use the model that performs the best on the testing data in production. For this reason, it is important to keep track of different experiments in an organized way.
<span class="marginal-note" data-info="Keeping track of changes and results from experiments is **super important**! Keep each experiment (trained model) in a separate place, and save all of the information you need to reproduce the experiment alongside it (notes, hyperparameter values, etc.). Here is an [example](https://gist.github.com/brannondorsey/5bd30f894c3dd3a9f290068c92156dba) of a directory structure that I use frequently, as well as a [similar approach](https://andrewhalterman.com/2018/03/05/managing-machine-learning-experiments/) by Andy Halterman."></span>
An advanced technique to quickly and effectively automate the training and evaluation cycle is to use [hyperparameter search](https://cloud.google.com/ml-engine/docs/tensorflow/hyperparameter-tuning-overview).

## Deploy Your Model

Once you've got a trained model that performs adequately it's time to deploy it live. This process of using your trained model, called model *inference* instead of *learning*
<span class="marginal-note" data-info="Model inference is usually just like model training, but you don't update the weights in the process. Instead the best trained model is &quot;frozen&quot;, test/production data is fed in, and you use the output as the model prediction. How choose to sample from the output prediction can change from model training though. In some cases you sample from the output probability distribution and in others you may choose the output unit with the highest value (e.g. greedy argmax sampling)."></span>
, looks different for every application, but can include things like integrating it into a server-side process that handles web API requests, bundling it in a mobile application, or using [Tensorflow.js](https://js.tensorflow.org) to deploy it to the web. Most models are trained to learn model weights that are then frozen during production. However, some models incorporate live unseen data into the training process. These models are said to operate, "online", and their is little distinction between training and deployment; the model is always learning.

Next: [Types of Learning](types-of-learning.html)