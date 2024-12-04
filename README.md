# neural-net-in-cxx
Implementation of neural network architecture in c++. This is designed to predict housing prices from the [California housing dataset](https://scikit-learn.org/dev/modules/generated/sklearn.datasets.fetch_california_housing.html). The model uses He weight initialisation, and also leaky-relu. Feature vectors are standardised before being passed into the neural network. Backpropogation uses batch gradient descent. MSE loss is used as a criterion. 

Included also in the discussion below, as well as in [pytorch_implementation.py](pytorch_implementation.py), is an implementation of a neural network in pytorch also trained on the [California housing dataset](https://scikit-learn.org/dev/modules/generated/sklearn.datasets.fetch_california_housing.html). This will serve as a touchstone to compare the effectiveness of my neural-net implentation against.

## Files/Folders
1) pytorch_implementation.py - contains a neural network implemented using pytorch
2) C++ implementation - contains all relevant files to my implementation of a neural network in c++.
3) input.txt is the input file containing data from the California housing dataset - used as input for the c++ implementation (input taken in main.cpp)
4) Output: contains outputs from both the python and cpp implementations.
    a) "cpp: predicted vs actual.txt": contains input value (x), output of the model (predicted) and target (y), ascertained in final epoch of training.
    b) cpp_errors_per_epoch.txt: essentially the output generapted by main.cpp (in the c++ implementation folder). Contains a list of train and test errors (model was not trained on the test data set) for each epoch.
   c) python output: contains the output of python_implementation.py - essentially the test and train loss per epoch
5) Graphs - a folder containing graphs generated from matplotlib.pyplot that contain error vs epoch curves for both the pytorch and c++ implementations of the neural network.
6) README_images: images I have made to assist my explanations throughout the rest of this article
7) Proof of competency: contains images of my rough working to show that I the gradient descent logic used in this project - contains mostly scribbles of linear algebra

## How to compile my c++ code
Download the "C++ implementation" folder and migrate into it via command prompt. Run the following command to compile: (note: using gcc compiler)
```
g++ main.cpp optimiser.cpp matrix.cpp neuralnet.cpp
```
Then the following command to run:
```
a.exe < input.txt > output.txt
```

## Results
Here is the epoch-loss graph for the pytorch implementation: ![pyTorchImplementation](Graphs/pytorchImplementation.png) 

and from my cpp implementation: 

![cppImplementation](Graphs/cppImplementation.png)

We can see that the y axis has a different scale - at the moment I believe this is due using a different divisor in my implementation of mse loss. Overall I am happy to say that my c++ neural network implementation is a success!

# An explanation of my neural network implementation
a.k.a an explanation of the theory behind gradient descent

## What is a neural network
A neural network is a sequence of layers - this can be thought of as a sequetially applied mathematical functions that maps a vector input $a$ of $i$ dimension ($a \in \mathbb{R}^i$, $a$ being a row vector) and applies matrix transformation producing an output vector $b$ of $o$ dimension ($b \in \mathbb{R}^o$, $b$ being a row vector). Each layer will apply the following transformation: 
```math
b = w \cdot a + b
```
where $w$ (our weights) is a matrix of dimensions $a$ x $b$ , and b (our biases) is a vector of dimension $b$, and $w \cdot a$ is the dot product of $w$ and $a$. This is a linear transformation.

To allow for non-linear mappings between inputs and outputs of each layer, we apply an activation function at each layer (some non-linear transformation of the data after each linear transformation). I have chosen the leaky-ReLU (leaky rectified linear unit function). This builds upon the ReLU activation. This is defined as 
```math
ReLU(x) = max(x, 0) = \frac{x + |x|}{2}
```
which prevents any negative values from being passed forwards. ReLU activaions can potentially lead to dead neurons: neurons that will never fire due to learning large negative biases. Instead of returning 0 for a negative input $X$, leaky-Rely instead returns $\alpha X$ where $\alpha$ is some coefficient that we define. 

![leakyRelu gradients](README_images/LeakyReluDesc.png)

By stacking these layers we create a mathematical function mapping between the dependant variable $X$ of some dimensions, to the vector space of our desired output $Y$. Note, I simply use a linear activation function for my final layer as this is what is typically deemed appropriate for regression problems. 

## How do we train neural networks so they can better model the relationship between X (inputs) and Y (outputs) 
We train our neural networks in order to optimise a cost function. I use the mean squared error loss defined as:
```math
\frac{1}{n} \Sigma_{i=1}^n (\hat{y}_i - y_i)^2
```
By optimising this with respects to the weights and biases in our neural network, we minimise our cost function and thus hope to improve the accuracy of our model. Unlike linear regression, this optimisation problem does not have a closed form solution - so we use a process called gradient descent.

Gradient descent involves calculating the gradient of our loss with respect to each weight and bias in our model. This gradient represents the steepest incline at the state the model is currently in. By subtracting the respective gradient from each weight, we are moving down the gradient (i.e. down the curve - see below diagram) and moving to the point of lowest cost.

![Grad descent](README_images/GradDescDesc.png)


## Weight initialisation

## Data standardisation
