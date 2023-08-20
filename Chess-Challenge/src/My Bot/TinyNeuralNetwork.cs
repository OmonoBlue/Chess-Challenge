﻿using System.Globalization;
using System;
using System.Linq;
using System.IO;

/// <summary>
/// A tiny neural network with one hidden layer and configurable parameters.
/// </summary>
public class TinyNeuralNetwork
{
    internal float[] Weights;
    internal float[] Biases;
    internal float[] HiddenLayer;
    internal float[] OutputLayer;
    internal int InputCount;
    internal Random Random;

    /// <summary>
    /// Creates an instance of an untrained neural network.
    /// </summary>
    /// <param name="inputCount">Number of inputs or features.</param>
    /// <param name="hiddenCount">Number of hidden neurons in a hidden layer.</param>
    /// <param name="outputCount">Number of outputs or classes.</param>
    /// <param name="seed">A seed for random generator to produce predictable results.</param>
    public TinyNeuralNetwork(
        int inputCount,
        int hiddenCount,
        int outputCount,
        int seed)
    {
        Random = new Random(seed);
        InputCount = inputCount;
        Weights = Enumerable
            .Range(0, hiddenCount * (inputCount + outputCount))
            .Select(_ => (float)Random.NextDouble() - 0.5f)
            .ToArray();
        Biases = Enumerable
            .Range(0, 2)
            .Select(_ => (float)Random.NextDouble() - 0.5f)
            .ToArray(); // Tinn only supports one hidden layer so there are two biases.
        HiddenLayer = new float[hiddenCount];
        OutputLayer = new float[outputCount];
    }

    /// <summary>
    /// Creates an instance of an untrained neural network.
    /// </summary>
    /// <param name="inputCount">Number of inputs or features.</param>
    /// <param name="hiddenCount">Number of hidden neurons in a hidden layer.</param>
    /// <param name="outputCount">Number of outputs or classes.</param>
    public TinyNeuralNetwork(
        int inputCount,
        int hiddenCount,
        int outputCount) : this(inputCount, hiddenCount, outputCount, seed: default)
    {
    }

    private TinyNeuralNetwork(
        float[] weights,
        float[] biases,
        float[] hiddenLayer,
        float[] outputLayer,
        int inputCount,
        int seed)
    {
        Weights = weights;
        Biases = biases;
        HiddenLayer = hiddenLayer;
        OutputLayer = outputLayer;
        InputCount = inputCount;
        Random = new Random(seed);
    }


    /// <summary>
    /// Loads a pre-trained neural network from a `*.tinn` file.
    /// </summary>
    /// <param name="path">An absolute or a relative path to the `*.tinn` file.</param>
    /// <returns>An instance of a pre-trained <see cref="TinyNeuralNetwork"/>.</returns>
    public static TinyNeuralNetwork Load(string path)
    {
        return Load(path, seed: default);
    }

    /// <summary>
    /// Loads a pre-trained neural network from a `*.tinn` file.
    /// </summary>
    /// <param name="path">An absolute or a relative path to the `*.tinn` file.</param>
    /// <param name="seed">A seed for random generator to produce predictable results.</param>
    /// <returns>An instance of a pre-trained <see cref="TinyNeuralNetwork"/>.</returns>
    public static TinyNeuralNetwork Load(string path, int seed)
    {
        using var reader = new StreamReader(path);
        var metaData = ReadLine();
        var counts = metaData.Split(' ').Select(int.Parse).ToList();
        var inputCount = counts[0];
        var hiddenCount = counts[1];
        var outputCount = counts[2];

        var weights = new float[hiddenCount * (inputCount + outputCount)];
        var biases = new float[2];
        var hiddenLayer = new float[hiddenCount];
        var outputLayer = new float[outputCount];
        var biasCount = 2;

        for (var i = 0; i < biasCount; i++)
        {
            biases[i] = float.Parse(ReadLine(), CultureInfo.InvariantCulture);
        }

        for (var i = 0; i < weights.Length; i++)
        {
            weights[i] = float.Parse(ReadLine(), CultureInfo.InvariantCulture);
        }

        var network = new TinyNeuralNetwork(weights, biases, hiddenLayer, outputLayer, inputCount, seed);
        return network;

        string ReadLine()
        {
            return reader.ReadLine() ?? throw new ArgumentException($"Corrupted file '{path ?? ""}', missing data.");
        }
    }

    /// <summary>
    /// Predicts outputs from a given input.
    /// </summary>
    /// <param name="input">A float array matching the length of input count.</param>
    /// <returns>An array of predicted probabilities for each class. </returns>
    public float[] Predict(float[] input)
    {
        PropagateForward(input);
        return OutputLayer;
    }

    /// <summary>
    /// Trains neural network on a single data record.
    /// </summary>
    /// <param name="input">Records input or feature values.</param>
    /// <param name="expectedOutput">Actual record's class in a categorical format.</param>
    /// <param name="learningRate">Learning rate of a training.</param>
    public float Train(float[] input, float[] expectedOutput, float learningRate)
    {
        PropagateForward(input);
        PropagateBackward(input, expectedOutput, learningRate);
        return GetTotalError(expectedOutput);
    }

    /// <summary>
    /// Saves a trained neural network to a `*.tinn` file.
    /// </summary>
    /// <param name="path">An absolute or a relative path to the `*.tinn` file.</param>
    public void Save(string path, bool backup = false)
    {
        if (backup && File.Exists(path))
        {
            string directory = Path.GetDirectoryName(path);
            string extension = Path.GetExtension(path);
            string filenameWithoutExtension = Path.GetFileNameWithoutExtension(path);
            string newFilename = $"{filenameWithoutExtension}_{DateTime.Now:yyyy-MM-dd_HH-mm-ss-fff}{extension}";
            string newFilepath = Path.Combine(directory, newFilename);

            File.Move(path, newFilepath);
        }
        using var writer = new FormattingStreamWriter(path, CultureInfo.InvariantCulture);
        writer.WriteLine($"{InputCount} {HiddenLayer.Length} {OutputLayer.Length}");

        foreach (var bias in Biases)
        {
            writer.WriteLine(bias);
        }

        foreach (var weight in Weights)
        {
            writer.WriteLine(weight);
        }
    }

    /// <summary>
    /// Get total error
    /// </summary>
    /// <param name="expectedOutput">Actual record's class in a categorical format.</param>
    /// <returns>Aggregated error value indicating how far off the neural network is on the training data set.</returns>
    public float GetTotalError(float[] expectedOutput)
    {
        return GetTotalError(expectedOutput, OutputLayer);
    }

    private void PropagateForward(float[] input)
    {
        // Calculate hidden layer neuron values.
        for (var i = 0; i < HiddenLayer.Length; i++)
        {
            var sum = 0.0f;
            for (var j = 0; j < InputCount; j++)
            {
                sum += input[j] * Weights[i * InputCount + j];
            }

            HiddenLayer[i] = ActivationFunction(sum + Biases[0]);
        }

        // Calculate output layer neuron values.
        for (var i = 0; i < OutputLayer.Length; i++)
        {
            var sum = 0.0f;

            for (var j = 0; j < HiddenLayer.Length; j++)
            {
                sum += HiddenLayer[j] * Weights[(HiddenLayer.Length * InputCount) + i * HiddenLayer.Length + j];
            }

            OutputLayer[i] = ActivationFunction(sum + Biases[1]);
        }
    }

    private void PropagateBackward(float[] input, float[] expectedOutput, float learningRate)
    {
        for (var i = 0; i < HiddenLayer.Length; i++)
        {
            var sum = 0.0f;

            // Calculate total error change with respect to output.
            for (var j = 0; j < OutputLayer.Length; j++)
            {
                var a = LossFunctionPartialDerivative(expectedOutput[j], OutputLayer[j]);
                var b = ActivationFunctionPartialDerivative(OutputLayer[j]);
                var weightIndex = (HiddenLayer.Length * InputCount) + j * HiddenLayer.Length + i;
                sum += a * b * Weights[weightIndex];

                // Correct weights in hidden to output layer.
                Weights[weightIndex] -= learningRate * a * b * HiddenLayer[i];
            }

            // Correct weights in input to hidden layer.
            for (var j = 0; j < InputCount; j++)
            {
                Weights[i * InputCount + j] -= learningRate * sum * ActivationFunctionPartialDerivative(HiddenLayer[i]) * input[j];
            }
        }
    }

    private static float ActivationFunction(float value)
    {
        return 1.0f / (1.0f + (float)Math.Exp(-value));
    }

    private static float ActivationFunctionPartialDerivative(float value)
    {
        return value * (1f - value);
    }

    private static float LossFunction(float expected, float actual)
    {
        return 0.5f * (expected - actual) * (expected - actual);
    }

    // Partial derivative of loss function with respect to the actual output.
    private static float LossFunctionPartialDerivative(float expected, float actual)
    {
        return actual - expected;
    }

    private static float GetTotalError(float[] expected, float[] actual)
    {
        var totalError = expected.Zip(actual, LossFunction).Sum();
        return totalError;
    }

    private class FormattingStreamWriter : StreamWriter
    {
        private readonly IFormatProvider _formatProvider;

        public FormattingStreamWriter(string path, IFormatProvider formatProvider)
            : base(path)
        {
            _formatProvider = formatProvider;
        }

        public override IFormatProvider FormatProvider => _formatProvider;
    }
}
