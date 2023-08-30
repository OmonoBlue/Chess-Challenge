#pragma once

#ifdef NNMETHODS_EXPORTS
#define NNMETHODS_API __declspec(dllexport)
#else
#define NNMETHODS_API __declspec(dllimport)
#endif

extern "C" NNMETHODS_API float dotArrays(float* arr1, size_t arr1Size, float* arr2, size_t arr2Size);

extern "C" NNMETHODS_API int test(int val);