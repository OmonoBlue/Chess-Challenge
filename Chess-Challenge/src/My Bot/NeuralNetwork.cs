using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;


public class NeuralNetwork
{
    private float[,] hiddenWeights;
    private float[] hiddenBiases;
    private float[,] outWeights;
    private float[] outBiases;

    private Random random;
    private int inputCount;
    private int hiddenCount;
    private int outputCount;

    public NeuralNetwork(int inputCount, int hiddenCount, int outputCount)
    {
        this.inputCount = inputCount;
        this.hiddenCount = hiddenCount;
        this.outputCount = outputCount;

        random = new Random();

        hiddenWeights = new float[hiddenCount, inputCount];
        hiddenBiases = new float[hiddenCount];
        outWeights = new float[outputCount, hiddenCount];
        outBiases = new float[outputCount];

        // initialize weights and biases
        for (int i = 0; i < hiddenCount; i++)
        {
            for (int j = 0; j < inputCount; j++)
            {
                hiddenWeights[i, j] = (float)random.NextDouble() - 0.5f;
            }

            for (int k = 0; k < outputCount; k++)
            {
                outWeights[k, i] = (float)random.NextDouble() - 0.5f;
            }
        }
    }

    private float[][] PropogateForward(float[][] inputBatch)
    {
        int batchSize = inputBatch.GetLength(0);
        float[][] outputs = new float[batchSize][];
        for (int b = 0; b < batchSize; b++)
        {
            float[] input = inputBatch[b];
            float[] hiddenLayer = new float[hiddenCount];
            outputs[b] = new float[outputCount];

            // Calculate hidden layer
            for (int h = 0; h < hiddenCount; h++)
            {
                float sum = 0f;
                for (int i = 0; i < inputCount; i++)
                {
                    sum += input[i] * hiddenWeights[h, i];
                }
                hiddenLayer[h] = ActivationFunction(sum + hiddenBiases[h]);
            }

            // Calculate output layer
            for (int o = 0; o < outputCount; o++)
            {
                float sum = 0f;
                for (int h = 0; h < hiddenCount; h++)
                {
                    sum += hiddenLayer[h] * outWeights[o, h];
                }
                outputs[b][o] = ActivationFunction(sum + outBiases[o]);
            }
        }
        return outputs;
    }

    private float[] PropogateForward(float[] input)
    {
        return PropogateForward(new float[][] { input })[0];
    }

    private static float ActivationFunction(float value)
    {
        return value > 0f ? value : 0f;
    }

    private static float ActivationFunctionPartialDerivative(float value)
    {
        return value > 0f ? value : 0f;
    }
}

