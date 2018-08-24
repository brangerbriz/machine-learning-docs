# Tensorflow.js

Tensorflow.js (`tfjs`) is a WebGL accelerated library for performing ML/Deep Learning tasks in-browser. It evolved out of the deeplearn.js library, which is now called Tensorflow.js Core.

With `tfjs`, you can run the same code using multiple backends:

- CPU
- WebGL
- Native Tensorflow w/ C Bindings (Node.js only)

The WebGL backend can be expected to run at 1.5-2x slower than Python Tensorflow. With larger models it has been found to run 10-15x slower. That said, it is amazing that Tensorflow.js brings graphics accelerated machine learning to the web!<span class="marginal-note" data-info="For a great demo of Tensorflow.js in browser see the [Teachable Machine project](https://teachablemachine.withgoogle.com/)"></span> What you may lose in speed, you gain in potential reach and deployability. Users of a product written in Tensorflow.js need only a modern web browser, while native ML libraries that leverage CUDA require special NVIDIA graphics hardware and drivers. There is little-to-no barrier to entry with Tensorflow.js.

Tensorflow.js tries to provide an API that is as close to Python Tensorflow as possible, but does not include all of the functions present in Python. The Layers API (`tf.layers`) is a high-level API for creating neural networks inspired by [Keras](https://keras.io/).

- [Tensorflow.js Introductory Post](https://medium.com/tensorflow/introducing-tensorflow-js-machine-learning-in-javascript-bf3eab376db)
- [Tensorflow.js Core Concepts](https://js.tensorflow.org/tutorials/core-concepts.html)
	- Tensors are Immutable (except when using their `.buffer()` method)
	- Variables are not
	- Ops return Tensors
	- GPU memory must be explicitly freed using dispose or `tf.tidy(...)`
	- `tf.layers` provides a high-level Keras-like API for building neural networks and models
- [Tensorflow.js Examples](https://github.com/tensorflow/tfjs-examples)
- [Tensorflow.js API Reference](https://js.tensorflow.org/api/latest/index.html)
- [FAQ](https://js.tensorflow.org/faq)

## Electron

With Electron, you can also build desktop applications using Tensorflow.js that also benefit from zero hardware, library, or driver dependence. We've bundled several `tfjs` examples as an Electron app in our [brangerbriz/tf-electron](https://github.com/brangerbriz/tf-electron) repository. Here is an [introductory Electron tutorial](https://github.com/electron/electron/blob/master/docs/tutorial/first-app.md) as well.
