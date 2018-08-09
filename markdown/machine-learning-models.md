# Machine Learning Models

Models are a synthetic representation of the way that something in the real life works. There exists a function that describes every non-random occurrence in the universe. Knowing any particular real-world function is impossible without knowing its output for every input past and future. But estimating/approximating a function can be done if you have enough example outputs of that function. The estimation of this function is called a model. 

In machine learning, models are the result of training. With [parametric](https://en.wikipedia.org/wiki/Parametric_model) ML algorithms like neural networks, the model is the trained network parameters (the weights and biases, more on that soon). Essentially, models are:

- Trained/learned
- The product of machine learning
- An algorithm
- Function approximators (see [General Purpose Algorithms](general-purpose-algorithms.html))
- Used to create predictions or generate new data once trained

Sometimes the word model and architecture are used interchangeably, however, model architecture usually describes the structure of the model (how many parameters, multiple combined models, depth of the model, etc.).
<span class="marginal-note" data-info="Choosing or engineering a model architecture that works well for your machine learning problem is no easy task. As a rule of thumb, wide networks (more neurons per layer, but fewer layers) are often easier to train, while deep networks are more difficult to train correctly, but if you do it right they often generalize better. In practice, you should try experiment with several architectures and move towards the ones that work. Best results in deep learning are found empirically. This holds true for all kinds of things: activation functions, optimizers, etc."></span>
A machine learning model usually refers to the product of training; the learned algorithm.

<!--
It is not uncommon to slightly alter the way that a model is used depending on whether it is being trained or being used in production (often confusingly called testing or sampling). For example, with Autoregressive Recurrent Neural Networks, models are fed ground-truth data during testing. During training, they are then fed their own past predictions in a sort of tail-eating feedback loop.
-->

Next: [The ML Pipeline](the-ml-pipeline.html)