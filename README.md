# Fashion-MNIST Autoencoder and t-SNE Visualization

## Description

This repository contains code for implementing an autoencoder using TensorFlow/Keras to encode and decode Fashion-MNIST images. It further utilizes t-SNE (t-distributed Stochastic Neighbor Embedding) to visualize the encoded images in a lower-dimensional space.

## Prerequisites

Ensure you have the following libraries installed:

- NumPy
- Matplotlib
- TensorFlow
- scikit-learn

## Getting Started

1. Clone this repository to your local machine.
2. Ensure all the required libraries are installed.
3. Run the provided Python script `t-SNE_Dimentionality_Reduction.ipynb`.

## Dataset

The Fashion-MNIST dataset consists of 60,000 grayscale images (28x28 pixels) of clothing items belonging to 10 different categories. It serves as a drop-in replacement for the original MNIST dataset.

## Autoencoder Architecture

The autoencoder architecture comprises an encoder and a decoder. The encoder reduces the dimensionality of the input images, while the decoder aims to reconstruct the original images from the encoded representations. The architecture used in this project is as follows:
- Input layer: 784 neurons (corresponding to flattened 28x28 images)
- Three hidden layers in the encoder: 128, 64, and 32 neurons respectively, with ReLU activation functions
- Three hidden layers in the decoder: 64, 128, and 784 neurons respectively, with ReLU activation functions in the intermediate layers and a sigmoid activation function in the output layer

## Training

The autoencoder is trained using the Fashion-MNIST training data. It minimizes the binary cross-entropy loss function using the Adam optimizer.

## t-SNE Visualization

After encoding the Fashion-MNIST images into a lower-dimensional space, t-SNE is applied to further reduce the dimensionality to 2 dimensions for visualization purposes. The t-SNE algorithm aims to preserve the local structure of the high-dimensional data in the lower-dimensional space.

## Results

The t-SNE visualization plots the encoded Fashion-MNIST images in a 2D scatter plot, where each point represents an image. Points belonging to the same category are typically clustered together, demonstrating the effectiveness of the autoencoder in capturing the underlying structure of the data.

## Acknowledgments

- The Fashion-MNIST dataset was originally created by [Zalando Research](https://github.com/zalandoresearch/fashion-mnist).
- Inspiration for the autoencoder architecture and t-SNE visualization was drawn from various sources in the machine learning community.
