#include "pch.h"
#include "nnmethods.h"
#include <cmath>


void PropogateForward(float input[], float hiddenLayer[], int hiddenCount, float** hiddenInputWeights, float hiddenBiases[], int outputCount, float** hiddenOutputWeights, float outputBiases[], float output[]) {
    // Calculate hidden layer
    for (int h = 0; h < hiddenCount; ++h) {
        float sum = dotArrays(input, hiddenInputWeights[h], hiddenCount); // Assuming input and hidden weights have the same size
        hiddenLayer[h] = ReLUActivation(sum + hiddenBiases[h]);
    }

    // Calculate output layer
    for (int o = 0; o < outputCount; ++o) {
        float sum = 0.0f;
        for (int h = 0; h < hiddenCount; ++h) {
            sum += hiddenLayer[h] * hiddenOutputWeights[h][o];
        }
        output[o] = TanhActivation(sum + outputBiases[o]);
    }
}

float dotArrays(float a[], float b[], int size) {
    float sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float ReLUActivation(float x) {
	return (x > 0) ? x : 0.0f;
}

float TanhActivation(float x) {
	return std::tanh(x);
}

int test(int val)
{
	return int(val);
}
