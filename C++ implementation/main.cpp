#include "optimiser.h"
#include<iostream>

void readIn(vector<matrix> &inputs, vector<matrix> &outputs, int numSize){
    // reading in X
    for(int i = 0; i < numSize; i++){
        matrix x(1, 8);
        cin >> x;

        inputs.push_back(x);
    }

    // reading in Y
    for(int i = 0; i < numSize; i++){
        matrix y(1,1);
        cin >> y;
        outputs.push_back(y);
    }

}

using namespace std;
int main(){
    // int numSize; cin >> numSize;

    vector<matrix> trainInputs, trainOutpus, testInputs, testOutputs;
    readIn(trainInputs, trainOutpus, 14447); //14447
    readIn(testInputs, testOutputs, 6193);




    vector<int> structure{8, 24, 12, 6, 1}; 
    model myModel(structure);


    int numEpochs = 10;
    for(int e = 0; e < numEpochs; e++){
        double currError = 0.0;
        for(int i = 0; i < trainInputs.size(); i++){
            currError += forward(myModel, trainInputs[i], trainOutpus[i], 14447.0); //14447.0
        }
        backpropogate(myModel, 0.001);

        cout << "train ERROR FOR EPOCH " << currError << endl;


        currError = 0.0;    
        for(int i = 0; i < 6193; i++){
            currError += forward_notrain(myModel, testInputs[i], testOutputs[i], 6193.0);
        }
        cout << "test ERROR FOR EPOCH " << currError << endl;
    }


}
