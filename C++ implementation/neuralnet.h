#pragma once
#include <vector>
#include "matrix.h"

using namespace std;


struct reluLayer{
    //technically leaky relu
    matrix weights, weightsGrad;
    matrix biases, biasesGrad;
    matrix lastInput;
    matrix lastOutput;
    bool isRelu;

    reluLayer(const int inputs, const int outputs, bool reluTrue = 1); 

    matrix relu(matrix &input);
    matrix forward(matrix forward);
    void updateWeights(double learningRate);

};

struct model{

    vector<reluLayer> structure;

    model(vector<int> &nodes);
        //takes in a vector containing the number of nodes in each layer
        //with nodes[0] containing the number of inputs
        //and nodes[-1] having to number of outputs

    matrix forwardPass(matrix input);

};

