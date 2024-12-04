# neural-net-in-cxx
Implementation of neural network architecture in c++. This is designed to predict housing prices from the [California housing dataset](https://scikit-learn.org/dev/modules/generated/sklearn.datasets.fetch_california_housing.html). The model uses He weight initialisation, and also leaky-relu. 
Currently works with stochastic gradient descent. Still working out bugs with batch-gradient descent.

Included also in the discussion below, as well as in the "Pytorch implementation" folder, is an implementation of a neural network in pytorch also trained on the [California housing dataset](https://scikit-learn.org/dev/modules/generated/sklearn.datasets.fetch_california_housing.html). This will serve as a touchstone to compare the effectiveness of my neural-net implentation against.

## Files/Folders
1) Pytorch implementation - contains a neural network implemented using pytorch
2) C++ implementation - contains all relevant files to my implementation of a neural network in c++.
3) input.txt is the input file containing data from the California housing dataset - used as input for the c++ implementation (input taken in main.cpp)
4) Output: contains outputs from both the python and cpp implementations.
    a) "cpp: predicted vs actual.txt": contains input value (x), output of the model (predicted) and target (y), ascertained in final epoch of training.
    b) cpp_errors_per_epoch.txt: essentially the output generapted by main.cpp (in the c++ implementation folder). Contains a list of train and test errors (model was not trained on the test data set) for each epoch.
   c) python output: contains the output of python_implementation.py - essentially the test and train loss per epoch
5) Graphs - a folder containing graphs generated from matplotlib.pyplot that contain error vs epoch curves for both the pytorch and c++ implementations of the neural network.

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

We can see that the y axis has a different scale - at the moment I believe this is due to my implementation of mse loss and potentially due to using a different divisor. So overall I am happy to say that my c++ neural network implementation is a success!

## An explanation of my neural network implementation
a.k.a an explanation of the theory behind gradient descent
