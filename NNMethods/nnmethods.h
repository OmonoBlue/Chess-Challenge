#pragma once

#ifdef NNMETHODS_EXPORTS
#define NNMETHODS_API __declspec(dllexport)
#else
#define NNMETHODS_API __declspec(dllimport)
#endif

extern "C" NNMETHODS_API void PropogateForward(float input[], float hiddenLayer[], int hiddenCount, float** hiddenInputWeights, float hiddenBiases[], int outputCount, float** hiddenOutputWeights, float outputBiases[], float output[]);

extern "C" NNMETHODS_API float dotArrays(float a[], float b[], int size);

extern "C" NNMETHODS_API float ReLUActivation(float x);

extern "C" NNMETHODS_API float TanhActivation(float x);

extern "C" NNMETHODS_API int test(int val);