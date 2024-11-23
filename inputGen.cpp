#include<math.h>
#include<random>
#include<vector>
#include <iostream>

using namespace std;

int main(){
    int numSize = 50;
    cout << numSize << endl;
    for(int i = 0; i < numSize; i++){
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(1.0, 100.0);

        double mass = dis(gen);
        double velocity = dis(gen);
        double Ek = mass*velocity*velocity;

        cout << mass << " " << velocity << " " << Ek << endl;
    }
}