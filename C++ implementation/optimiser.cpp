#include "optimiser.h"
#include <utility>

pair<matrix, double> mse_loss(matrix &estimated, matrix &actual){
    //defined for a 1d horizontal vectors
    // defined as 1/(2N) * sum((pred - actual)^2).  1/N is calculated in forwards function defined below;
    // Note: replacing 1/M with 1/2 for ease w/ backpropogation algorithm
    if(estimated.rows != 1 || estimated.columns != actual.columns|| actual.rows != 1){
        throw std::invalid_argument( "y_estimated and y_actual are not of same dimensions!" );
    }

    matrix error = estimated-actual;

    // want to return gradient: only defined for 1 output -> can probably change by returning matrix and having error as a matrix
    return make_pair(error, (error*error.transpose())[0][0]); // returns (pred-y), error
    
}


void backpropogate(model &myModel, double learningRate){
    for(int i = 0; i < myModel.structure.size(); i++){
        reluLayer &currLayer = myModel.structure[i];
        currLayer.updateWeights(learningRate);
    }
}

double forward(model &myModel, matrix &x, matrix &y, double batchNum){    
    matrix prediction = myModel.forwardPass(x);
    matrix errorGrad; //will maintain invariant that errorGrad is always of shape 1x(num_inputs_for_layer);
    double error;

    tie(errorGrad, error) = mse_loss(prediction, y);
    
    //backpropagation mathematics
    error = error / batchNum; // 1/N for actual error
    errorGrad = errorGrad/batchNum; // d(estimate-output)^2 / d(estimate-output)

    for(int i = myModel.structure.size()-1; i >= 0; i--){
        reluLayer& currLayer = myModel.structure[i];

        if(errorGrad.columns != currLayer.lastOutput.columns){
            throw std::invalid_argument( "there is an error in my backpropogation code" );
        }
        if(myModel.structure[i].isRelu){
            for(int j = 0; j < errorGrad.columns; j++){
                errorGrad[0][j] = (currLayer.lastOutput[0][j] > 0) ? errorGrad[0][j] : 0.01*errorGrad[0][j];//// d relu(wx+b) / d(wx+b); 
            }
        }


        currLayer.weightsGrad = currLayer.weightsGrad + currLayer.lastInput.transpose() * errorGrad; // d(wx+b) / d(w)
        currLayer.biasesGrad = currLayer.biasesGrad + errorGrad;

        errorGrad = errorGrad * currLayer.weights.transpose(); // d(wx+b) / d(x);
    }

    return error;
}

double forward_notrain(model &myModel, matrix &x, matrix &y, double batchNum){
    matrix prediction = myModel.forwardPass(x);
    matrix errorGrad; //will maintain invariant that errorGrad is always 1x(num_inputs_for_layer);
    double error;

    tie(errorGrad, error) = mse_loss(prediction, y);
    error = error/batchNum;

    return error;    
}
