# Machine Learning Pipeline

![The Machine Learning Pipeline](images/ml-pipeline.png)

## Get Data

The first step in any machine learning pipeline is data acquisition. There are generally four methods of obtaining data:

- Use a [publicly available dataset](https://github.com/caesar0301/awesome-public-datasets)
- Scrape data from a publicly accessible API or service
- Use a private dataset you or your company has collected
- Design and deploy a method for collecting new data

The first three are fairly similar. The last provides the power to design custom features with the trade-off of money and time.

## Clean and Pre-process Data

The data preparation, [pre-processing](https://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/) and [normalization](normalization-and-whitening.html) (COME BACK) steps are often the most time consuming and code-heavy part of the pipeline. It's also the most important step to get right. Most machine learning algorithms only operate well on data that is within a certain range (e.g. small values between `-1.0` and `1.0` or `0.0` and `1.0`) with a standard variance. Real-world data might have arbitrary values with units in the millions, or tons of outliers that need to be removed. Once you've cleaned and pre-processed your data, you will need split itd into groups to be used for training and testing. Correctly preparing and partitioning your data before using it for training is often the key to developing a successful model.

## Train Your Model

Now comes the fun part, its time to [train your model](https://elitedatascience.com/model-training) (using your training data only). Training can take anywhere from a few minutes to hours or days depending on your [model architecture](model-architecture.html) or compute hardware. As a rule of thumb, training your model on a GPU will yield training times orders of magnitude faster than CPU training and should almost always be preferenced.

During the training process, your data is iteratively fed into the model in small batches, subsets of the entire training dataset, called mini-batches. Once your model has seen all of the training data, it has completed one epoch. Training usually requires multiple epochs and ends when [validation loss](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) (error) stops decreasing. More on that soon...

## Evaluate Your model

Once you've trained your model, its time to [evaluate its performance](performance-measures.html). You usually do this by freezing the model parameters, running the model on the test data, but this time without updating the parameters like you do during training. During evaluation the goal is to measure the trained model's error/accuracy on unseen data. Once you've got a measure of how effective your model is, its time to train another model to try and beat it. The ML pipeline often requires [many iterations of training and evaluating](model-evaluation-and-tweaks.html); the goal being to reduce the error on your test data. It is not uncommon to do this `10` to `100+` times, choosing to use the model that performs the best on the testing data in production. For this reason, it is important to keep track of [different experiments](experiment-structure.html) in an organized way. An advanced technique to quickly and effectively automate the training and evaluation cycle is to use [hyperparameter search](https://cloud.google.com/ml-engine/docs/tensorflow/hyperparameter-tuning-overview).

## Deploy Your Model

Once you've got a trained model that performs adequately it's time to deploy it live. This process of [using your trained model](using-your-trained-model.html), called model *inference* instead of *learning*, looks different for every application, but can include things like integrating it into a server-side process that handles web API requests or bundling it in a mobile application. Most models are trained to learn model weights that are then frozen during production. However, some models incorporate live unseen data into the training process. [These models](online-models.html) are said to operate, "online", and their is little distinction between training and deployment; the model is always learning.

Next: [Types of Learning](types-of-learning.html)