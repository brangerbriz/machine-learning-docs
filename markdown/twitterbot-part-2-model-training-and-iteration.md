# Twitterbot Part 2: Model Training and Iteration

In [Part 1](twitterbot-part-1-twitter-data-preparation.html) of this tutorial, we learned how to prepare twitter data for training. In this section, we'll learn how to use that data to train models. We'll begin with a basic introduction to model training as well as common training idioms, jargon, and practices. Then we'll learn how we can automate part of the training process using a technique called hyperparameter search to find a good model configuration, given our limited compute resources.

## Model Training

At a high level, training machine learning models is simple. You initialize a model with random weights<span class="marginal-note" data-info='You will see people use the words "weights" and "parameters" interchangeably. They both refer to the values a model learns during training and later uses for inference. Hyperparameters on the other hand are a totally different animal. More on that soon.'></span> and feed it batches of training data `X`, comparing its output `y^`<span class="marginal-note" data-info='Pronounced "Y Hat"'></span> to the expected output `y` provided by labeled data, tweaking its weights in the process in an attempt to minimize the error between `y^` and `y`. Each time you feed your entire training dataset to the model, it's called an epoch. 

After each epoch, you evaluate your model by feeding it unseen validation data and again comparing its outputs `y^` to the expected outputs `y`. During this evaluation you don't update the model weights as you did during training. The average error from the training epoch, called the `loss`, is then compared with the average error from the validation evaluation, called the `val_loss`. If the `val_loss` is lower than it was last epoch, you usually train for another epoch and then repeat the process. If it's higher than it used to be, it may be that your model is learning to memorize the training data instead of using it to extract useful patterns that lead to good generalization on unseen data. In this case, you may be overfitting your data, and you usually stop training there, saving the current model weights to disk. At this point you consider your model "trained."<span class="marginal-note" data-info="After which, you often enter a seemingly infinite loop of training another model to outperform the one you just trained. This is called model iteration and we'll discuss it later in this chapter."></span>

There are lots of variables at play during the model training process and the complexity of some of the steps can seem daunting. There are lots of ways to do it wrong and only a few ways to do it right.<span class="marginal-note" data-info="This short post does a nice job of explaining [why machine learning is so 'hard'](https://ai.stanford.edu/~zayd/why-is-machine-learning-hard.html)"></span> That said, we'll blaze ahead covering the most important details. If parts of the code we introduce next goes over your head, don't worry about it. There are a lot of moving parts and big concepts to digest so take it in stride and don't worry if entire pieces of what we do next seem foggy. The goal is to get you up and running with working code that don't look totally foreign.

Enter the `char-rnn-text-generation/` directory that we created in [Part 1](twitterbot-part-1-twitter-data-preparation.html). From here, we'll install several python dependencies. If you prefer to do that inside of a python virtual environment, that's fine too. Just make sure that you are using python3.

<pre class="code">
    <code class="bash" data-wrap="false">
# download requirements.txt, which contains the python dependencies needed for model training
wget -O requirements.txt https://raw.githubusercontent.com/brangerbriz/char-rnn-text-generation/master/requirements.txt

# install the dependencies
sudo -H pip3 install -r requirements.txt
    </code>
</pre>

If you have an NVIDIA GPU in your machine and have CUDA setup<span class="marginal-note" data-info="See the [GPU ML Development Environment](file:///home/braxxox/Documents/Branger_Briz/machine-learning-docs/www/ml-development-environment.html) section."></span>, you can install the GPU accelerated version of tensorflow with `pip3 install tensorflow-gpu==1.10.1`.

You should now have several python libraries installed.

- **[Keras](https://keras.io/)**: A high-level toolkit for composing networks out of simple layers objects
- **[Tensorflow](https://www.tensorflow.org/)**: Google's ML and Tensor computation library. This is the backend library that Keras uses under the hood.
- **[Numpy](http://www.numpy.org/)**: A utility library for matrix and vector operations in python.
- **[H5py](http://www.h5py.org/)**: A library for saving and loading HDF5 binary files. Keras model weights are saved in this format after training.
- **[Hyperopt](https://hyperopt.github.io/hyperopt/)**: A hyperparameter search library that we'll use to find good settings for our models.

Next, we'll copy over some of the data loading utilities we wrote in [Part 1](twitterbot-part-1-twitter-data-preparation.html) to a file called [`utils.py`](https://github.com/brangerbriz/char-rnn-text-generation/blob/master/utils.py).

<pre class="code">
    <code class="bash" data-wrap="false">
wget -O utils.py https://raw.githubusercontent.com/brangerbriz/char-rnn-text-generation/master/utils.py
    </code>
</pre>

You may notice a few extra functions in this file  but most of it should look familiar. Notice that we've renamed `data_generator()` to `io_batch_generator()`. We've also taken advantage of Numpy's powerful array operations to simplify some of our encoding/decoding logic. Keras operates on numpy N-dimensional arrays, so not only do our Numpy modifications produce more concise code, they also prepare the data in a way that Keras prefers to accept it.

<pre class="code">
    <code class="python" data-wrap="false">
def encode_text(text, char2id=CHAR2ID):
    """
    encode text to array of integers with CHAR2ID
    """
    return np.fromiter((char2id.get(ch, 0) for ch in text), int)


def decode_text(int_array, id2char=ID2CHAR):
    """
    decode array of integers to text with ID2CHAR
    """
    return "".join((id2char[ch] for ch in int_array))


def one_hot_encode(indices, num_classes):
    """
    one-hot encoding
    """
    return np.eye(num_classes)[indices]
    </code>
</pre>

As you can imagine, we'll use `util.py` to hold some of our general purpose utility functions like those used for loading and managing data. This is good practice because these functions will be used by several of our scripts. Next, let's create a new file called `train.py` and add the following code snippet.

<pre class="code">
    <code class="python" data-wrap="false">
# a few basic imports
import os
import utils
# some more complex imports from Keras. We'll use these in a bit.
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import EarlyStopping, LambdaCallback
from keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed
from keras.models import load_model, Sequential

# these are our hyperparameters. They define the architecture and 
# configuration of our model.
BATCH_SIZE=64
SEQ_LEN=64
EMBEDDINGS_SIZE=32
RNN_SIZE=128
NUM_LAYERS=1
DROP_RATE=0.2 # 20% dropout
NUM_EPOCHS=1

# we'll save our model to checkpoints/checkpoint.hdf5
CHECKPOINT_DIR='checkpoints/'
CHECKPOINT_PATH=os.path.join(CHECKPOINT_DIR, 'checkpoint.hdf5')
TRAIN_TEXT_PATH=os.path.join('data', 'tweets-split', 'train.txt')
VAL_TEXT_PATH=os.path.join('data', 'tweets-split', 'test.txt')

# here's where we'll put our code for training the model
def train():
    pass

# here's where we'll put our code for constructing the model.
# this will be called from inside train().
def build_model():
    pass

# run train() as the main function for this file.
train()
    </code>
</pre>

We'll use this template to build out our actual training algorithm. Before we do, let's dive into the meaning of some of these hyperparameter constants.

### Hyperparameters

Hyperparameters are the parameters that the machine learning practitioner chooses. This is in contrast to the model's parameters, or weights, which are updated automatically by the learning algorithm during training. Think of hyperparameters like meta-parameters: They are the high-level parameters that make up the model's configuration and training behavior, not what the model learns during training. Different algorithms will have different hyperparameters to tweak, but there are several that you will encounter quite regularly.

- `BATCH_SIZE`: This is the number of data samples `X` that will be fed into the model before updating the model's weights. Weight updates are calculated across an average of all samples in the batch, so a low batch size means more updates dependent on fewer samples while a high batch size means fewer updates, but each update will factor in a smoother average across more samples. This hyperparameter is often constrained by your computer's memory resources, as the batch size defines how many samples your model must be able to hold in memory (or GPU memory) at once.
- `RNN_SIZE`: The number of RNN units, or weights, per layer. The more units per layer the higher the model's capacity, but also the more compute resources and disk space required to run and save it. The `RNN_SIZE` can be considered the width of the model.<span class="marginal-note" data-info="Here, we're calling it RNN_SIZE, but the number of units per layer isn't unique to RNNs. Maybe a more appropriate name is NN_SIZE or UNIT_SIZE."></span>
- `NUM_LAYERS`: The depth of the model, or number of `RNN_SIZE` layers it has. The number of units is `RNN_SIZE * NUM_LAYERS`. However, the time performance cost of adding a new layer seems to be far greater than doubling the `RNN_SIZE`. As a rule of thumb, wider networks are often easier to change, though in some circumstances deeper networks perform better.
- `DROP_RATE`: The Dropout rate defines what percentage of neurons will be randomly turned off during each training batch. Dropout is an effective regularization method<span class="marginal-note" data-info="More on regularization [here](regularization.html)."></span> that is commonly used to combat overfitting. If you find that your training loss is consistently lower than your validation loss, try increasing the dropout percentage.
- `NUM_EPOCHS`: The number of epochs, or full passes through your training dataset, to train your model before stopping. While this might not sound like a hyperparameter, it is. Early stopping, the process of halting model training once `val_loss` has stopped improving is actually a form of regularization. Keras has a method for doing just that which we will explore later. Here, we define `NUM_EPOCHS` as the **maximum** number epochs before forcing training to stop.

Several of our hyperparameters are more unique to our RNN character-generation model. You will likely see them in other models in the future, but perhaps more rarely than the others.

- `SEQ_LEN`: The number of individually encoded characters to group as a sequence and feed as input to the model. `SEQ_LEN=10` might encode the input sequence `hello ther` while `SEQ_LEN=5` would only include `hello`.
- `EMBEDDING_SIZE`: The length, or size, of the learned word embeddings for each character. This size can be set arbitrarily, however it's value will effect model performance and compute resources.

For our model, we'll use word embeddings to encode our example data `X`. The input to our model will have a shape that is equal to `(BATCH_SIZE, SEQ_LEN, EMBEDDING_SIZE)` and each batch of samples will have `BATCH_SIZE * SEQ_LEN * EMBEDDING_SIZE` float values. We'll encode our labeled data `y`, using one-hot encoding, so its shape will be `(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)`, with the number of float values for `y` equal to the product of these there hyperparameters as well. During training, we'll use a sequence of characters and expect a sequence of the same number of characters as output. This type of model is called a sequence-to-sequence model.

Further in the chapter we'll explore a technique called hyperparameter search that can be used to choose a set of values for these hyperparemeters. For now, we'll leave these values as they are and fill in our template with code to actually train our models.

### Training Code

We'll start by using the Keras API, along with our hyperparameters, to build a model. Keras provides a high-level API for building basic neural network architectures using common building blocks like layers. It abstracts away the actual matrix multiplications and calculus needed to train a model. You'll be left with the task of designing your model architecture and figuring out how to wrangle data into and out from the model, Keras it handles the actual "learning". Go ahead and fill in the empty `build_model()` function with the contents below.<span class="marginal-note" data-info="Keras' use of the word 'Sequential()' doesn't mean the data has to be sequential, but rather the construction of the model's layers are sequential: one after another. It just so happens that we are using a Sequential() model to process sequential data using an RNN."></span>

<pre class="code">
    <code class="python" data-wrap="false">
def build_model():

    # Keras' Sequential() model API allows us to construct a hierarchical
    # model where the output of each layer acts as the input to the next.
    model = Sequential()
    model.add(Embedding(utils.VOCAB_SIZE, EMBEDDINGS_SIZE,
                        batch_input_shape=(BATCH_SIZE, SEQ_LEN)))
    model.add(Dropout(DROP_RATE))
    for _ in range(NUM_LAYERS):
        model.add(LSTM(RNN_SIZE, return_sequences=True, stateful=True))
        model.add(Dropout(DROP_RATE))
    model.add(TimeDistributed(Dense(utils.VOCAB_SIZE, activation="softmax")))

    # Here is a breakdown of the layers as well as the shape of our model
    # at each level.
    # add an embedding layer to map input indexes to word embeddings
    # Embedding layer, maps class int labels to word vectors
    #   input shape: (BATCH_SIZE, SEQ_LEN)
    #   output shape: (BATCH_LEN, SEQ_LEN, EMBEDDING_SIZE)
    # Dropout
    #   randomly "turns off" 20% of the input neurons each batch
    # LSTM Layer 1 ... LSTM Layer N
    #   input shape: (BATCH_SIZE, SEQ_LEN, EMBEDDING_SIZE)
    #   output shape: (BATCH_SIZE, SEQ_LEN, RNN_SIZE)
    #   followed by another layer of dropout
    # Time Distributed Dense
    #   input shape: (BATCH_SIZE, SEQ_LEN, EMBEDDING_SIZE)
    #   output shape: (BATCH_SIZE, SEQ_LEN, utils.VOCAB_SIZE)
    #   uses softmax activation function to map each input character to a 
    #      probability distribution over a likely output character
    #   the output distributions can then be sampled from to predict the next
    #      character in the sequence

    return model
    </code>
</pre>

Let's break these layers down. First, we've got an embedding layer, which we define as having `utils.VOCAB_SIZE`<span class="marginal-note" data-info="Remember from Part 1 that VOCAB_SIZE is the equal to the number of Python printable characters, minus several characters we've removed. That leaves us with 98 unique character classes."></span> unique classes and an output size of `EMBEDDING_SIZE`. This is our look up table that transforms our integer valued class labels to word embeddings. Next we add a dropout layer which will randomly null out 20% of our input features before feeding them to the input layer. This effectively destroys input data and as a result adds variation to our input samples in a way that makes it more difficult for the model to memorize the training data. By doing so we encourage the model to work harder to extract more meaningful patterns from the data.

Dropout is followed by series of LSTM layers, starting with the input layer. LSTM stands for long short-term memory and it's a popular unit type to use with recurrent neural networks. Since it was introduced in the late 1990s it has been shown to perform better than "vanilla" RNN units at learning long-term dependence between input sequences, a task that all neural network models actually perform pretty poorly at.<span class="marginal-note" data-info="True long-term dependence is an unsolved problem. It's the reason that models can create seemingly realistic short utterances that start to lose meaning and structure as the length of the generated text grows. That's why we can model shakespeare's speech in a way that *sounds* like shakespeare, but doing so doesn't leave us with an infinite number of quality new shakespeare plays. Check out [this delightful short film](https://www.youtube.com/watchv=LY7x2Ihqjmc) where a team of actors actually perform a screenplay written by an LSTM model."></span> The LSTM layers are where the *learning* happens, though we won't go into the details just of how that works here. Each layer after the input layer is called a "hidden" layer as the relationship between the input values and the neurons becomes opaque and non-linear, or "hidden."

Finally we pass the output from our final hidden layer to a set of dense output layers. This is a common practice in classification tasks. A dense layer is a flat set of vanilla neural network units. If we use a dense layer as the final layer we set the number of output units equal to the number of class labels. The value at each index then represents the likelihood that a given input sample belongs to the one-hot class label at that same index. By using a `"softmax"` activation function, we normalize the output from the final dense layer, transforming it into a probability distribution where all elements in the vector sum to `1.0`, and thus each index represents the percent chance that an input belongs to its corresponding class label. We have one dense layer for each character in the output sequence, which is handled by the call to `TimeDistributed()`.

If you followed that it means that our model is expecting to be fed an input of integers<span class="marginal-note" data-info="Which it will internally map to word embeddings whose' values are learned during training."></span> and will output a sequence of vector values, each with a length equal to the number of characters we have in our vocab size. Each vector represents a predicted character in the output sequence. The values in each of these vectors corresponds to the probability the model has assigned to each possible output class.

Now that we've added code that creates our model, let's add the code train it. Apologies ahead of time for slamming you in the face with a wall full of `monospaced source code` but there isn't a lot of sugar coating for this bit of training code. I've done my best to annotate it with helpful comments and we'll cover the basics in more detail once you've powered through reading it.

<pre class="code">
    <code class="python" data-wrap="false">
def train():

    # create our model using the function we just wrote ^
    model = build_model()

    # once we've loaded/built the model, we need to compile it using an 
    # optimizer and a loss. The loss must be categorical_crossentropy, because
    # our task is a multi-class classification problem.
    model.compile(loss="categorical_crossentropy", optimizer='rmsprop')

    # Callbacks are hooked functions that keras will call automagically
    # during model training. For more info, see https://keras.io/callbacks/
    #     - EarlyStopping: Will stop model training once val_loss plateuas.
    #     - TensorBoard: logs training metrics so that they may be viewed by 
    #       tensorboard.
    #     - ModelCheckpoint: save model weights as checkpoints after
    #       each epoch.
    #       Only save a checkpoint if val_loss has improved.
    #     - LabmdaCallback: Your own custom hooked function. Here we reset the
    #       model's RNN states between Epochs.
    callbacks = [
        # if val_loss doesn't improve more than 0.01 for three epochs in a row
        # stop training the model
        EarlyStopping(monitor='val_loss', patience=3, min_delta=0.01),
        TensorBoard(os.path.join(CHECKPOINT_DIR, 'logs')),
        # only save model checkpoints if the val_loss improved during this 
        # epoch. If it didn't, don't overwrite a better model we already saved
        ModelCheckpoint(CHECKPOINT_PATH, verbose=1, save_best_only=True),
        # you MUST reset the model's RNN states between epochs
        LambdaCallback(on_epoch_end=lambda epoch, logs: model.reset_states())
    ]

    # because we may not be able to fit all of our training data into RAM at 
    # once we will use a python generator to lazy-loads (and release) data 
    # from disk into RAM as we need it. We will use one generator for 
    # validation data and another for training data.
    val_generator = utils.io_batch_generator(VAL_TEXT_PATH,
                                             batch_size=BATCH_SIZE,
                                             seq_len=SEQ_LEN,
                                             one_hot_labels=True)
    train_generator = utils.io_batch_generator(TRAIN_TEXT_PATH,
                                               batch_size=BATCH_SIZE,
                                               seq_len=SEQ_LEN,
                                               one_hot_labels=True)

    # the way the generator is written, we won't know how many samples are in
    # the entire dataset without processing it all once. We need to know this
    # number so that we can know the number of batch steps per epoch. This 
    # isn't elegant, but it is a tradeoff that is worth making to have no 
    # limit to the amount of training data we can process using our generator.
    train_steps_per_epoch = get_num_steps_per_epoch(train_generator)
    val_steps_per_epoch = get_num_steps_per_epoch(val_generator)
    print('train_steps_per_epoch: {}'.format(train_steps_per_epoch))
    print('val_steps_per_epoch: {}'.format(val_steps_per_epoch))

    # now that we've computed train_steps_per_epoch and val_steps_per_epoch
    # we will re-create the generators so that they begin at 
    # epoch 0 instead of 1
    val_generator = utils.io_batch_generator(VAL_TEXT_PATH,
                                             batch_size=BATCH_SIZE,
                                             seq_len=SEQ_LEN,
                                             one_hot_labels=True)
    train_generator = utils.io_batch_generator(TRAIN_TEXT_PATH,
                                               batch_size=BATCH_SIZE,
                                               seq_len=SEQ_LEN,
                                               one_hot_labels=True)

    # our io_batch_generator returns (x, y, epoch) but model.fit_generator()
    # expects only (x, y) so we create generator_wrapper() which simply 
    # disposes of the extra "epoch" tuple element.
    val_generator = generator_wrapper(val_generator)
    train_generator = generator_wrapper(train_generator)

    model.reset_states()

    # train the model using train_generator for training data and val_generator
    # for validation data. The results from model.fit_generator() is a history
    # object holds information about model training and evaluation for each
    # epoch. We won't use it, but it's worth mentioning that it exists here in
    # case you do want to do something with that information in the future.
    history = model.fit_generator(train_generator,
                                  epochs=NUM_EPOCHS,
                                  steps_per_epoch=train_steps_per_epoch,
                                  validation_data=val_generator,
                                  validation_steps=val_steps_per_epoch,
                                  callbacks=callbacks)

# our io_batch_generator yields tuples of (x, y, epoch), but 
# model.fit_generator() requires ONLY (x, y) tuples, so this is a simple 
# wrapper that throws away the extra epoch variable.
def generator_wrapper(generator): 
    while True:
        x, y, _ = next(generator)
        yield x, y

# because we're using a generator to lazy load training data from disk, we 
# don't know the full size of an epoch without actually iterating through the 
# entire generator. Function iterates through it's generator argument for the 
# entire first epoch of data, counting the number of steps it takes to do so.
def get_num_steps_per_epoch(generator):
    num_steps = 0
    while True:
        _, _, epoch = next(generator)
        if epoch > 1:
            return num_steps
        else:
            num_steps += 1  # add batch_size samples
    </code>
</pre>

We begin by creating our model and "compiling" it, whereby we define our optimization algorithm and loss function. A loss function is a measurement of error<span class="marginal-note" data-info="See [Performance Measures](performance-measures.html)."></span> between what our model outputs and what our labeled data says it should output. There are several common loss functions used in machine learning and [Keras supports most of them](https://keras.io/losses/). Categorical cross entropy (i.e. `categorical_crossentropy`) should be used exclusively for multi-class classification problems. It measures the divergence between the predicted probability distribution and the true probability distribution of the labeled data.<span class="marginal-note" data-info="Or rather, a sample drawn from the underlying data distribution."></span> You don't have to remember what it does so long as you remember to use it whenever you are trying to solve a classification problem with more than two classes. 

Once we've defined our loss function as a performance measure for a model, we can automagically optimize our model weights with respect to the this loss function in an attempt to minimize it. This optimization method attempts to find a suitable set of model weights `W` such that the scalar loss value `L` produced by our loss `categorical_crossentropy` loss function decreases. 

There are several popular optimization algorithms used in deep learning today and most of them derive from stochastic gradient descent (SGD). RMSProp is a commonly used to optimize RNNs, so we'll try that first, but feel free to experiment with the other optimization functions that Keras supports. You'll find some of them may arrive at lower losses while others converge to a "good enough" loss more quickly.