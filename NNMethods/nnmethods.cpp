#include "pch.h"
#include "nnmethods.h"


float dotArrays(float* arr1, size_t arr1Size, float* arr2, size_t arr2Size)
{
	float sum = 0;
	for (int i = 0; i < arr1Size; ++i) {
		sum += arr1[i] * arr2[i];
	}
	return sum;
}

int test(int val)
{
	return int(val);
}
