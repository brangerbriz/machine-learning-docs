# Data is 🔑

Large amounts of quality data are the key to the success of any machine learning algorithm. After all, models can only be as good as the data used to train them. The bottleneck in an [ML pipeline](the-ml-pipeline.html) is often in access to high-quantities of high-quality data. It is not uncommon for the majority of the development time it takes to implement an ML solution will be spent preparing, [pre-processing](data-preprocessing.html), and cleaning your data.

## Data Quantities

There are so many factors at play in any given ML algorithm that it is often difficult to know the minimum amount of data needed for your model to start performing well in practice. Simple data statistics like standard deviation, number and quality of features, number of model parameter, and problem task all play a significant role in determining how much data is needed. That said, the answer to the question of "how much data do I need?" is usually always "more data."

I've seen rules of thumb that suggest that the minimum number of samples needed to train a model range anywhere from `10xN` to `N^2` where `N` is the number of features (columns) in your data. The truth is that the amount of data required scales with [model capacity](model-capacity.md) and complexity. Like many things in machine learning, quantity of data must be [empirically tested](empirical-testing.html).

## Data Representation

The way that you represent your data is arguably more important than the amount of raw data itself. Data is represented using [features](features-and-design-matrices.html), and the ones you choose (or better yet, [learn](feature-learning.html)) influence how effectively your model can learn about the data. Just because you have 20 columns in your database to represent a user doesn't mean that all 20 of those columns are necessary to solve a simple classification task. You may find that your model performs best using only 4 features from each data sample.

Its also very common not to use the features from your dataset directly, but rather process them first to compute a new feature that is more helpful for your model. For instance, if you were training a classifier to predict durations of metro-rail commutes given a database of card swipe timestamps, a duration feature (`swipe_out - swipe_in`) may be more beneficial than two separate swipe timestamp features. 

## Training Data vs Test Data

There is a critically important distinction to be made between data used to train a model and data used to test the [performance of that model](performance-measures.html). The same data **cannot** be used to both train your model and evaluate its performance, as this will lead to dramatic [overfitting](overfitting-and-underfitting.html). The training of every [supervised](types-of-learning.html) machine learning model requires that you split your data, using the majority of it for training and holding out the minority for testing and model evaluation. A split of 80% training data and 20% test data is common. If you have a lot of data, you can experiment with 85%+ training data. Model performance will likely always be better on your test data, but hopefully only a few percentage points away from evaluation on your test data. Test performance is a measure of how well your model will generalize to unseen real-world data it will encounter "in the wild" once it is deployed. Data holdout is an important topic, and I recommend checking out the [Training/Test/Validation Data](training-test-validation-data.html) page for more info.

Finally, [Kaggle](https://kaggle.com) is an amazing data science community resource that publishes publicly available datasets and code examples. Perusing the site should give you a good overview of the types of data representations that are often used in machine learning.

Next: [Machine Learning Models](machine-learning-models.html)
