#include "optimiser.h"
#include<iostream>

using namespace std;
int main(){
    int numSize; cin >> numSize;

    vector<matrix> inputs, outputs;
    
    matrix mx(1, 1, -1.0);
    matrix mn(1, 1, 1000000.0);

    for(int i = 0; i < numSize; i++){
        matrix x(1, 1);
        matrix y(1, 1);
        cin >> x >> y;

        inputs.push_back(x);
        outputs.push_back(y);

        for(int col = 0; col < mx.columns; col++){
            if(x[0][col] > mx[0][col]) mx[0][col] = x[0][col];
            if(x[0][col] < mn[0][col]) mn[0][col] = x[0][col];
        }
    }

    //regularising the data!
    matrix denom = mx-mn;
    for(int i = 0; i < numSize; i++){
        inputs[i] = elementWiseDiv((inputs[i] - mn), denom);
    }

    vector<int> structure{1, 5, 5, 1}; // testing ability to approximate Ek = 1/2 mv^2
    model myModel(structure);


    int numEpochs = 10;
    for(int e = 0; e < numEpochs; e++){
        double currError = 0.0;
        for(int i = 0; i < numSize; i++){
            // cerr << "input " << inputs[i] << endl;
            currError += forward(myModel, inputs[i], outputs[i], numSize);
        }
        backpropogate(myModel, 0.01);

        // for(int i = 0; i < myModel.structure.size(); i++){
        //     cerr << myModel.structure[i].weights << endl;
        // }
        cerr << "FINAL ERROR FOR EPOCH " << currError << endl;
    }


}