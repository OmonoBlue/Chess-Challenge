using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;


public class NeuralNetwork
{
    private int inputCount;
    private int hiddenCount;
    private int outputCount;

    private float[] inputLayer;
    private float[] hiddenLayer;
    private float[,] inputHiddenWeights;
    private float[] hiddenBiases;

    private float[,] hiddenOutputWeights;
    private float[] outputBiases;
    private float[] outputLayer;

    private Random random;

    public NeuralNetwork(int inputCount, int outputCount, float[,] hiddenWeights, float[] hiddenBiases, float[,] outWeights, float[] outBiases, int seed = default)
        :this(inputCount, hiddenBiases.Length, outputCount, seed)
    {
        this.inputHiddenWeights = hiddenWeights;
        this.hiddenBiases = hiddenBiases;
        this.hiddenOutputWeights = outWeights;
        this.outputBiases = outBiases;
    }

    public NeuralNetwork(int inputCount, int hiddenCount, int outputCount, int seed = default)
    {
        if (seed == default) random = new Random();
        else random = new Random(seed);

        this.inputCount = inputCount;
        this.hiddenCount = hiddenCount;
        this.outputCount = outputCount;

        inputLayer = new float[inputCount];
        hiddenLayer = new float[hiddenCount];
        inputHiddenWeights = new float[hiddenCount, inputCount];
        hiddenBiases = new float[hiddenCount];
        hiddenOutputWeights = new float[outputCount, hiddenCount];
        outputBiases = new float[outputCount];
        outputLayer = new float[outputCount];

        // initialize weights and biases
        for (int i = 0; i < hiddenCount; i++)
        {
            for (int j = 0; j < inputCount; j++)
            {
                inputHiddenWeights[i, j] = (float)random.NextDouble() - 0.5f;
            }

            for (int k = 0; k < outputCount; k++)
            {
                hiddenOutputWeights[k, i] = (float)random.NextDouble() - 0.5f;
            }
        }
    }

    private float[][] PropogateForward(float[][] inputBatch)
    {
        int batchSize = inputBatch.GetLength(0);
        float[][] outputs = new float[batchSize][];
        for (int b = 0; b < batchSize; b++)
        {
            inputLayer = inputBatch[b];
            hiddenLayer = new float[hiddenCount];
            outputs[b] = new float[outputCount];

            // Calculate hidden layer
            for (int h = 0; h < hiddenCount; h++)
            {
                float sum = 0f;
                for (int i = 0; i < inputCount; i++)
                {
                    sum += inputLayer[i] * inputHiddenWeights[h, i];
                }
                // ReLU activation for hidden layer
                hiddenLayer[h] = ReLUActivation(sum + hiddenBiases[h]);
            }

            // Calculate output layer
            for (int o = 0; o < outputCount; o++)
            {
                float sum = 0f;
                for (int h = 0; h < hiddenCount; h++)
                {
                    sum += hiddenLayer[h] * hiddenOutputWeights[o, h];
                }
                // Tanh activation for outputs
                outputs[b][o] = TanhActivation(sum + outputBiases[o]);
            }
        }
        return outputs;
    }

    private float[] PropogateForward(float[] input)
    {
        inputLayer = input;
        hiddenLayer = new float[hiddenCount];
        outputLayer = new float[outputCount];

        // Calculate hidden layer
        for (int h = 0; h < hiddenCount; h++)
        {
            float sum = 0f;
            for (int i = 0; i < inputCount; i++)
            {
                sum += inputLayer[i] * inputHiddenWeights[h, i];
            }
            // ReLU activation for hidden layer
            hiddenLayer[h] = ReLUActivation(sum + hiddenBiases[h]);
        }

        // Calculate output layer
        for (int o = 0; o < outputCount; o++)
        {
            float sum = 0f;
            for (int h = 0; h < hiddenCount; h++)
            {
                sum += hiddenLayer[h] * hiddenOutputWeights[o, h];
            }
            // Tanh activation for outputs
            outputLayer[o] = TanhActivation(sum + outputBiases[o]);
        }
        return outputLayer;
    }

    private static float ReLUActivation(float value)
    {
        return value > 0f ? value : 0f;
    }

    private static float ReLUActivationDerivative(float value)
    {
        return value > 0f ? value : 0f;
    }

    private static float TanhActivation(float value)
    {
        return (float)Math.Tanh(value);
    }

    private static float TanhActivationDerivative(float value)
    {
        double tanh = Math.Tanh(value);
        return (float)(1-(tanh * tanh));
    }

    private static float MeanSquaredError(float[] expected, float[] actual)
    {
        float sumError = actual.Zip(expected, (a, e) => (a - e) * (a - e)).Sum();
        return sumError / actual.Length;
    }

    private static float[][] MakeMatrix(int rows, int cols)
    {
        float[][] matrix = new float[rows][];
        for (int r = 0; r < matrix.Length; ++r)
            matrix[r] = new float[cols];
        return matrix;
    }
}

