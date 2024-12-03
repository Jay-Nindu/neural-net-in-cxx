#include "matrix.h"
#include "neuralnet.h"

// need to define rule of 4 for both relu and model classes

reluLayer::reluLayer(const int inputs, const int outputs, bool reluTrue){
    weights = matrix(inputs, outputs);
    weightsGrad = matrix(inputs, outputs, 0);
    biases = matrix(1, outputs, 0);
    biasesGrad = matrix(1, outputs, 0);
    isRelu = reluTrue;
}

matrix reluLayer::relu(matrix &input){
    for(int i = 0; i < input.rows; i ++){
        for(int j = 0; j < input.columns; j++){
            input[i][j] = max(input[i][j]*0.01, input[i][j]); //coefficient is set here!


        }
    }
    return input;
}

matrix reluLayer::forward(matrix input){
    // need to store each input at different levels so does not take in a reference!
    lastInput = input;

    input = input*weights + biases;

    if(isRelu){
        lastOutput = relu(input);
    }
    else{
        lastOutput = input;
    }

    return lastOutput;
}

void reluLayer::updateWeights(double learningRate){
    // cout << "WEIGHTS " << weightsGrad << endl;
    weights -= weightsGrad*learningRate;
    weightsGrad.clear(); //clear gradients
    biases -= biasesGrad*learningRate;
    // cout << "BIASES " << biasesGrad << endl;
    biasesGrad.clear();

}

model::model(vector<int> &nodes){
    for(int i = 0; i < nodes.size()-1; i++){
        if(i == nodes.size()-2){
            structure.push_back(reluLayer(nodes[i], nodes[i+1], 0));
        }
        else{
            structure.push_back(reluLayer(nodes[i], nodes[i+1], 1));

        }
    }
}

matrix model::forwardPass(matrix input){
    for(int i = 0; i < structure.size(); i++){
        input = structure[i].forward(input);
        // cout << "Next output " << input;
    }
    return input;
}
