#include "optimiser.h"
#include<iostream>

using namespace std;

void readIn(vector<matrix> &inputs, vector<matrix> &outputs, int numSize){
    // reading in X
    double count = 0.0;
    matrix mean(1, 8, 0.0);
    matrix m2(1, 8, 0.0);

    for(int i = 0; i < numSize; i++){
        matrix x(1, 8);
        cin >> x;


        count = count+1.0;
        matrix delta = x-mean;
        mean += delta/count;
        matrix delta2 = x-mean;
        m2 += elementWiseMult(delta2, delta);
        inputs.push_back(x);
    }

    m2 = sqrt(m2/numSize); 

    //data normalisation
    for(int i = 0; i < numSize; i++){
        inputs[i] = elementWiseDiv(inputs[i]-mean, m2);
    }

    // reading in Y
    for(int i = 0; i < numSize; i++){
        matrix y(1,1);
        cin >> y;
        outputs.push_back(y);
    }

}

int main(){
    // int numSize; cin >> numSize;

    vector<matrix> trainInputs, trainOutpus, testInputs, testOutputs;
    readIn(trainInputs, trainOutpus, 14447); //14447
    readIn(testInputs, testOutputs, 6193);




    vector<int> structure{8, 24, 12, 6, 1}; 
    model myModel(structure);


    int numEpochs = 50;
    for(int e = 0; e < numEpochs; e++){
        double currError = 0.0;
        for(int i = 0; i < trainInputs.size(); i++){
            currError += forward(myModel, trainInputs[i], trainOutpus[i], 14447.0); //14447.0
        }
        backpropogate(myModel, 0.1);

        cout << "train ERROR FOR EPOCH " << currError << endl;


        currError = 0.0;    
        for(int i = 0; i < 6193; i++){
            currError += forward_notrain(myModel, testInputs[i], testOutputs[i], 6193.0);
        }
        cout << "test ERROR FOR EPOCH " << currError << endl << endl;
    }


}
