#include "optimiser.h"
#include<iostream>

using namespace std;
int main(){
    int numSize; cin >> numSize;

    vector<matrix> inputs, outputs;
    
    matrix mx(1, 2, -1.0);
    matrix mn(1, 2, 1000000.0);

    for(int i = 0; i < numSize; i++){
        matrix x(1, 2);
        matrix y(1, 1);
        cin >> x >> y;

        inputs.push_back(x);
        outputs.push_back(y);

        for(int col = 0; col < 2; col++){
            if(x[0][col] > mx[0][col]) mx[0][col] = x[0][col];
            if(x[0][col] < mn[0][col]) mn[0][col] = x[0][col];
        }
    }

    //regularising the data!
    matrix denom = mx-mn;
    for(int i = 0; i < numSize; i++){
        inputs[i] = elementWiseDiv((inputs[i] - mn), denom);
        cout << inputs[i] << endl;
    }


}