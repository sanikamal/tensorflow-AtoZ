# Introduction
Machine Learning come across many different terms such as `artificial intelligence`, `machine learning`, `neural network`, and `deep learning`. But what do these terms actually mean and how do they relate to each other?

Below given a brief description of these terms:

**Artificial Intelligence:** A field of computer science that aims to make computers achieve human-style intelligence. There are many approaches to reaching this goal, including `machine learning` and `deep learning`.

**Machine Learning:** A set of related techniques in which computers are trained to perform a particular task rather than by explicitly programming them.

**Neural Network:** A construct in `Machine Learning` inspired by the network of neurons (nerve cells) in the biological brain. `Neural networks` are a fundamental part of `deep learning`.

**Deep Learning:** A subfield of `machine learning` that uses multi-layered neural networks. Often, “machine learning” and “deep learning” are used interchangeably.
![AI DIAGRAM](images/ai-diagram.png)

**Machine learning** and **deep learning** also have many subfields, branches, and special techniques. A notable example of this diversity is the separation of **Supervised Learning** and **Unsupervised Learning**.

To over simplify — in supervised learning you know what you want to teach the computer, while unsupervised learning is about letting the computer figure out what can be learned. Supervised learning is the most common type of machine learning.

## Applications of Machine Learning
Coming soon......
## Prerequisites
Coming soon.....
#### - Introduction to Python

# Introduction to Machine Learning
## What is Machine Learning?
coming soon....

There are many types of neural network architectures. However, no matter what architecture you choose, the math it contains (what calculations are being performed, and in what order) is not modified during training. Instead, it is the internal variables (“weights” and “biases”) which are updated during training.

For example, in the Fahrenheit to Celsius conversion problem, the model starts by multiplying the input by some number (the weight) and adding another number (the bias). Training the model involves finding the right values for these variables, not changing from multiplication and addition to some other operation.

#### Fahrenheit to Celsius conversion problem
#### Celsius to Fahrenheit conversion problem

Some ML Terminology:

- **Feature:** The input(s) to our model
- **Examples:** An input/output pair used for training
- **Labels:** The output of the model
- **Layer:** A collection of nodes connected together within a neural network.
- **Model:** The representation of your neural network
- **Dense and Fully Connected (FC):** Each node in one layer is connected to each node in the previous layer.
- **Weights and biases:** The internal variables of model
- **Loss:** The discrepancy between the desired output and the actual output
- **MSE:** Mean squared error, a type of loss function that counts a small number of large discrepancies as worse than a large number of small ones.
- **Gradient Descent:** An algorithm that changes the internal variables a bit at a time to gradually reduce the loss function.
- **Optimizer:** A specific implementation of the gradient descent algorithm. (There are many algorithm.e.g. the “Adam” Optimizer, which stands for ADAptive with Momentum. It is considered the best-practice optimizer.)
- **Learning rate:** The “step size” for loss improvement during gradient descent.
- **Batch:** The set of examples used during training of the neural network
- **Epoch:** A full pass over the entire training dataset
- **Forward pass:** The computation of output values from input
- **Backward pass (backpropagation):** The calculation of internal variable adjustments according to the optimizer algorithm, starting from the output layer and working back through each layer to the input.
