# Features and Design Matrices

Features are any attribute or measurable property that can be used to describes a data sample. With tabular data, features can be thought of as column names. For instance, in a database of forum users, the features that describe each user may include `user_id`, `username`, `email`, `date_joined`, `number_of_posts`, `number_of_comments`, `number_of_visits`, `time_on_site`, `last_seen`. Similarly, the Iris dataset describes observations of each wild Iris flower using five features: `iris_species`, `sepal_length`, `sepal_width`, `petal_length` and `petal_width`. In a traditional supervised learning task, a model can be trained to predict the `iris_species` from the four other features.

![Iris Dataset](images/iris.png) While it can help to think of features as the column names in a database of samples, the most important thing to realize about features are that they are any information that you use to describes your data. **Features are the input to a machine learning model**. The process by which you determine which features to include in the data that is fed to your model is called *feature selection*.

## Raw Features

It is often satisfactory to use the "as is" representation of your data, with preprocessing of course, with your machine learning model successfully (see [Normalization and Preprocessing](normalization-and-preprocessing.html)). For instance, when your data represents an image, the features are usually the raw pixel values of that image. When your data is audio, the features may be the raw audio samples, between `-1.0` and `1.0` sampled at 44.1kHz. 

For categorical features a [one-hot encoding](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) transformation must be used, to convert class label strings into a binary representation.

## Derived Features

Sometimes it is helpful to augment your data representation with derived features before using it as input to an ML model. Think of derived features as a metadata of sorts, features about your data rather than the raw features themselves. For image data, perhaps you use a quantized version of the pixel brightness histogram instead of the raw image pixels themselves. For audio data, perhaps you use the fast-fourier transform of the audio data as features instead of the raw samples themselves. Raw features and derived features are not mutually exclusive, and it is perfectly fine to use both raw features and derived features at the same time. 

The reason that derived features are helpful is that they provides your model with an initial understanding of the data that its fed in a way that may be helpful "prior knowledge" for the type of task it is solving. In theory, a neural network with infinite resources and the correct training algorithm could learn to perfectly learn a set of neurons that equivalently match any derived feature given enough data. But in reality, if a pixel value histogram is important to the task at hand, it may be easier to provide that information to the model directly as a feature instead of tasking the model with learning to extract the relevant information itself using only the raw pixels.

Derived features should be used with some caution because they introduce a significant amount of developer bias into the machine learning pipeline (see [The ML Pipeline](the-ml-pipeline.html)). Just because we *think* that a particular set of features would be useful to a model in learning to execute some task doesn't mean that those features *are actually* useful.

## Dimensionality Reduction

A significant amount of recent research has suggested that [feature learning](https://en.wikipedia.org/wiki/Feature_learning) is often preferable to using hand-crafted or derived features because the model has the opportunity to learn what kind of features to care about rather than being explicitly told by the programmer.

One common technique to achieve feature learning is dimensionality reduction. Dimensionality reduction uses an automated process to combine features, resulting in a transformation of input features that results in fewer output features while preserving as much information from the original input features as possible. Algorithms like [PCA](http://setosa.io/ev/principal-component-analysis/) (linear), [t-SNE](https://nicola17.github.io/tfjs-tsne-demo/) (non-linear) and model architectures like autoencoders, take `N` features as input and output `N^` features, where `length(N^) < length(N)`. Dimensionality reduction can be thought of as a form of lossy compression. This technique can be used as a form of data preprocessing, as the output from a dimensionality reduction algorithm can be used as the input features to another machine learning algorithm.

## What Makes Good Features?

<!-- Come back here once you learn a bit more about the relationship of covariance to good features.

> Marginal note: "linearly independent from other features": Footnote about i.i.d and covariance. As a rule of thumb, features with low covariance often perform better. Talk about covariance (when covariance is positive they change together, when negative, they are inversely related). Zero covariance == independent vars. Show covariance matrix.
-->

The best features provide the most information in the smallest amount of space and are often linearly independent from other features to avoid repetition. That said, it's difficult to describe the perfect attributes of good features because different ML algorithms respond to features, and their relationships, differently. 

The more features you have the more computational resources your model will take to process them, and the more training samples you will need for your model to learn a successful mapping from features to outputs.
<span class="marginal-note" data-info="A typical rule of thumb is that there should be at least 5 training examples for each dimension in the representation (see the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality#Machine_learning) Wikipedia page)."></span>
 The fewer features you have the simpler your model is and the fewer training samples you will require. But with too few features your model might not have access to the information that is required to learn a mapping from the input features to the correct output.

There is no one-size fits all solution to this problem, so trial-and-error and empirical testing are required to choose a feature representation that fits the model you are using and the task you are trying to solve. 

## Design Matrices

Close to the idea of features is the *Design Matrix* (sometimes called a data matrix). A design matrix is simply a collection of data samples. If features can be thought of as a row in a table that corresponds to a single data point, the design matrix is the table itself. A design matrix `X` contains multiple data samples described by their features `x1`, `x2`, `x3`, etc. 

<pre class="code">
    <code class="js" data-wrap="true">
// individual samples, containing 3 features each
let x1 = [0.3, 0.1, 0.1]
let x2 = [0.2, 0.0, 0.3]
let x3 = [0.0, 0.3, 0.2]

// this is the design matrix
let X = [x1, x2, x3]
    </code>
</pre>

Most machine learning algorithms operate on batches of data, instead of single data points, at a time. Design matrices are used to group data samples and feed them into models for training or inference.

Next: [Linear Regression](linear-regression.html)
