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
    private float[][] inputHiddenWeights;
    private float[] hiddenBiases;

    private float[][] hiddenOutputWeights;
    private float[] outputBiases;
    private float[] outputLayer;

    private Random random;

    public NeuralNetwork(int inputCount, int outputCount, float[][] hiddenWeights, float[] hiddenBiases, float[][] outWeights, float[] outBiases, int seed = default)
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
        inputHiddenWeights = MakeMatrix(inputCount, hiddenCount);
        hiddenBiases = new float[hiddenCount];
        hiddenOutputWeights = MakeMatrix(hiddenCount, outputCount);
        outputBiases = new float[outputCount];
        outputLayer = new float[outputCount];

        // initialize weights and biases
        for (int i = 0; i < inputCount; i++)
            for (int h = 0; h < hiddenCount; h++)
                inputHiddenWeights[i][h] = (float)random.NextDouble() - 0.5f;

        for (int o = 0; o < outputCount; o++)
            for (int h = 0; h < hiddenCount; h++)
                hiddenOutputWeights[h][o] = (float)random.NextDouble() - 0.5f;

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
                    sum += inputLayer[i] * inputHiddenWeights[i][h];
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
                    sum += hiddenLayer[h] * hiddenOutputWeights[h][o];
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
/*        hiddenLayer = new float[hiddenCount];
        outputLayer = new float[outputCount];*/

        // Calculate hidden layer
        for (int h = 0; h < hiddenCount; h++)
        {
            float sum = 0f;
            for (int i = 0; i < inputCount; i++)
            {
                sum += inputLayer[i] * inputHiddenWeights[i][h];
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
                sum += hiddenLayer[h] * hiddenOutputWeights[h][o];
            }
            // Tanh activation for outputs
            outputLayer[o] = TanhActivation(sum + outputBiases[o]);
        }
        return outputLayer;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="trainingData">Array of training input-output pairs</param>
    /// <param name="maxEpochs">Number of epochs to train over training data</param>
    /// <param name="learningRate">Learning rate</param>
    /// <param name="momentum">Amount the learning rate changes</param>
    /// <returns></returns>
    public void Train((float[], float[])[] trainingData, int maxEpochs, float learningRate, float momentum)
    {
        // train using back-prop
        // back-prop specific arrays
        float[][] hoGrads = MakeMatrix(hiddenCount, outputCount); // hidden-to-output weight gradients
        float[] obGrads = new float[outputCount];                   // output bias gradients

        float[][] ihGrads = MakeMatrix(inputCount, hiddenCount);  // input-to-hidden weight gradients
        float[] hbGrads = new float[hiddenCount];                   // hidden bias gradients

        float[] oSignals = new float[outputCount];                  // local gradient output signals - gradients w/o associated input terms
        float[] hSignals = new float[hiddenCount];                  // local gradient hidden node signals

        // back-prop momentum specific arrays 
        float[][] ihPrevWeightsDelta = MakeMatrix(inputCount, hiddenCount);
        float[] hPrevBiasesDelta = new float[hiddenCount];
        float[][] hoPrevWeightsDelta = MakeMatrix(hiddenCount, outputCount);
        float[] oPrevBiasesDelta = new float[outputCount];

        int epoch = 0;
        float[] inputValues = new float[inputCount]; // inputs
        float[] targetOutput = new float[outputCount]; // target values
        float[] actualOutput = new float[outputCount]; // actual output values
        float[][] allOutputs = MakeMatrix(trainingData.Length, outputCount);
        float derivative = 0.0f;
        float errorSignal = 0.0f;

        int[] sequence = Enumerable.Range(0, trainingData.Length).ToArray();

        int errInterval = 1; // interval to check error

        while (epoch < maxEpochs)
        {
            ++epoch;
            if (epoch % errInterval == 0 && epoch < maxEpochs)
            {
                float trainErr = MeanSquaredError(trainingData.Select(e => e.Item1).ToArray(), allOutputs);
                Console.WriteLine("epoch = " + epoch + "  error = " +
                  trainErr.ToString("F4"));
                //Console.ReadLine();
            }

            Shuffle<int>(random, sequence);
            foreach (int s in sequence) { 
                inputValues = trainingData[s].Item1;
                targetOutput = trainingData[s].Item2;

                actualOutput = PropogateForward(inputValues);
                allOutputs[s] = actualOutput;

                // 1. compute output node signals
                for (int o =  0; o < outputCount; ++o)
                {
                    errorSignal = targetOutput[o] - actualOutput[o];
                    derivative = TanhActivationDerivative(actualOutput[o]);
                    oSignals[o] = errorSignal * derivative;
                }

                // 2. compute hidden-output weight gradients using output signals
                for (int h = 0; h < hiddenCount; ++h)
                    for (int o = 0; o < outputCount; ++o)
                    {
                        hoGrads[h][o] = oSignals[o] * hiddenLayer[h];
                        // 2b. compute output bias gradients using output signals
                        obGrads[o] = oSignals[o] * 1;
                    }
                // 3. hidden node signals
                for (int h = 0; h < hiddenCount; ++h)
                {
                    derivative = ReLUActivationDerivative(hiddenLayer[h]);
                    float sum = 0f;
                    for (int o = 0; o < outputCount; ++o)
                        sum += oSignals[o] * hiddenOutputWeights[h][o];
                    hSignals[h] = derivative * sum;
                }

                // 4. input-hidden weight gradients
                for (int h = 0; h < hiddenCount; ++h)
                {
                    for (int i = 0; i < inputCount; ++i)
                        ihGrads[i][h] = hSignals[h] * inputValues[i];

                    hbGrads[h] = hSignals[h] * 1; // dummy 1.0 input
                }

                // update input-to-hidden weights
                for (int i = 0; i < inputCount; ++i)
                {
                    for (int h = 0; h < hiddenCount; ++h)
                    {
                        float delta = ihGrads[i][h] * learningRate;
                        inputHiddenWeights[i][h] += delta; // would be -= if (o-t)
                        inputHiddenWeights[i][h] += ihPrevWeightsDelta[i][h] * momentum;
                        ihPrevWeightsDelta[i][h] = delta; // save for next time
                    }
                }

                // update hidden biases
                for (int h = 0; h < hiddenCount; ++h)
                {
                    float delta = hbGrads[h] * learningRate;
                    hiddenBiases[h] += delta;
                    hiddenBiases[h] += hPrevBiasesDelta[h] * momentum;
                    hPrevBiasesDelta[h] = delta;
                }

                // update hidden-to-output weights
                for (int h = 0; h < hiddenCount; ++h)
                {
                    for (int o = 0; o < outputCount; ++o)
                    {
                        float delta = hoGrads[h][o] * learningRate;
                        hiddenOutputWeights[h][o] += delta;
                        hiddenOutputWeights[h][o] += hoPrevWeightsDelta[h][o] * momentum;
                        hoPrevWeightsDelta[h][o] = delta;
                    }
                }

                // update output node biases
                for (int o = 0; o < outputCount; ++o)
                {
                    float delta = obGrads[o] * learningRate;
                    outputBiases[o] += delta;
                    outputBiases[o] += oPrevBiasesDelta[o] * momentum;
                    oPrevBiasesDelta[o] = delta;
                }
            }
        }
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

    private static float MeanSquaredError(float[][] expected, float[][] actual)
    {
        float sumErrors = Enumerable.Range(0, actual.Length).Select(x => MeanSquaredError(expected[x], actual[x])).Sum();
        return sumErrors / actual.Length;
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

    private static void Shuffle<T>(Random rng, T[] array)
    {
        int n = array.Length;
        while (n > 1)
        {
            int k = rng.Next(n--);
            T temp = array[n];
            array[n] = array[k];
            array[k] = temp;
        }
    }
}

