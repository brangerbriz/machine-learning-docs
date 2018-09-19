# Twitter Transfer Learning Guide

## 0. Architecture

- NNs, RNNs, and CNNs
- Recurrent Neural Networks
- Stateful RNNs
- Time Distributed Dense

## 1. Prep data for training

- Introduction and overview.

- Twitter CIKM 2010 (9,000,000+ million tweets)
- Training, Validation, Testing split
- Python printable characters only ("" everything else)
- Character Embeddings (vs one hot)
- Generators and batch loading

## 2. Use data to Iteratively train models

- Introduction to Model training
    - Batches and Epochs
    - Training Data & Validation Data
    - Loss and Optimizers
    - Saving model checkpoints
- `train.py` common training interface
- Hyperparameter search and Tensorboard
    - hyperparameter-search.py
    - trials.csv
- Loss vs resource and time tradeoffs
- Lowered learning rate

## 3. Deploy pre-trained model and using it (sampling)

- Sampling (Greedy argmax,two sampling from a distribution, top-n sampling)
- Generation
    - Adjusted batch and sequence sizes
    - Freeze model weights and set trainable = false
    - generate.py
- Model conversion
    - Using the command-line tool
- Deploying model in tfjs
- Saving/loading models to indexddb:// 

## 4. Fine tuning and user personalization

- Introduction to transfer learning and model fine-tuning
- Downloading twitter data per-user using tweet-server
- Model fine-tuning and hyperparameter search