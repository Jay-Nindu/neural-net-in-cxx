#include "neuralnet.h"


pair<matrix, double> mse_loss(matrix &estimated, matrix &actual);
void backpropogate(model &myModel, double learningRate);

double forward(model &myModel, matrix &x, matrix &y, double batchNum); 

