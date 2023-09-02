using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Threading;

public class NeuralNetwork
{
    [DllImport("NNMethods.dll", EntryPoint = "dotArrays", CallingConvention = CallingConvention.Cdecl)]
    private static extern float CdotArrays(float[] arr1, float[] arr2, int size);

    [DllImport("NNMethods.dll", EntryPoint = "PropogateForward", CallingConvention = CallingConvention.Cdecl)]
    private static extern void CPropogateForward(float[] input, float[] hiddenLayer, int hiddenCount, float[][] hiddenInputWeights, float[] hiddenBiases, int outputCount, float[][] hiddenOutputWeights, float[] outputBiases, float[] output);

    [DllImport("NNMethods.dll", EntryPoint = "ReLUActivation", CallingConvention = CallingConvention.Cdecl)]
    private static extern float CReLUActivation(float x);
    [DllImport("NNMethods.dll", EntryPoint = "TanhActivation", CallingConvention = CallingConvention.Cdecl)]
    private static extern float CTanhActivation(float x);

    private int inputCount;
    private int hiddenCount;
    private int outputCount;

    [ThreadStatic] private static float[] hiddenLayer;

    private float[][] hiddenInputWeights;
    private float[] hiddenBiases;
    private float[][] hiddenOutputWeights;
    private float[] outputBiases;

    private Random random;

    public NeuralNetwork(int inputCount, int outputCount, float[][] hiddenWeights, float[] hiddenBiases, float[][] outWeights, float[] outBiases, int seed = default)
        : this(inputCount, hiddenBiases.Length, outputCount, seed)
    {
        this.hiddenInputWeights = hiddenWeights;
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

        hiddenLayer = new float[hiddenCount];
        hiddenInputWeights = MakeMatrix(hiddenCount, inputCount);
        hiddenBiases = new float[hiddenCount];
        hiddenOutputWeights = MakeMatrix(hiddenCount, outputCount);
        outputBiases = new float[outputCount];

        // initialize weights and biases
        for (int i = 0; i < inputCount; i++)
            for (int h = 0; h < hiddenCount; h++)
                hiddenInputWeights[h][i] = (float)random.NextDouble() * 0.2f - 0.1f;

        for (int o = 0; o < outputCount; o++)
            for (int h = 0; h < hiddenCount; h++)
                hiddenOutputWeights[h][o] = (float)random.NextDouble() * 0.2f - 0.1f;

    }

    public float[] Predict(float[] input)
    {
        return PropogateForward(input);
    }

    public float[][] PropogateForward(float[][] inputBatch)
    {
        int batchSize = inputBatch.GetLength(0);
        float[][] outputs = new float[batchSize][];
        for (int b = 0; b < batchSize; b++)
        {
            float[] input = inputBatch[b];
            hiddenLayer = new float[hiddenCount];
            outputs[b] = new float[outputCount];

            // Calculate hidden layer
            for (int h = 0; h < hiddenCount; h++)
            {
                float sum = 0f;
                for (int i = 0; i < inputCount; i++)
                {
                    sum += input[i] * hiddenInputWeights[h][i];
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

    public float[] PropogateForward(float[] input)
    {
        hiddenLayer = new float[hiddenCount];
        float[] output = new float[outputCount];

        // Calculate hidden layer
        for (int h = 0; h < hiddenCount; ++h)
        {
            //float sum = CdotArrays(input, hiddenInputWeights[h], inputCount);
            float sum = 0;
            for (int i = 0; i < inputCount; ++i)
            {
                sum += input[i] * hiddenInputWeights[h][i];
            }
            // ReLU activation for hidden layer

            hiddenLayer[h] = ReLUActivation(sum + hiddenBiases[h]);
        }

        // Calculate output layer
        for (int o = 0; o < outputCount; ++o)
        {
            float sum = 0f;
            for (int h = 0; h < hiddenCount; ++h)
            {
                sum += hiddenLayer[h] * hiddenOutputWeights[h][o];
            }
            // Tanh activation for outputs
            output[o] = TanhActivation(sum + outputBiases[o]);
        }
        return output;
    }

    private float[] BatchError((float[], float[])[] trainingData)
    {
        List<float[]> exceptedOutputs = new List<float[]>();
        List<float[]> actualOutputs = new List<float[]>();
        foreach ((float[] input, float[] output) in trainingData)
        {
            exceptedOutputs.Add(output);
            actualOutputs.Add(PropogateForward(input));
        }
        return AggregateMeanSquaredError(exceptedOutputs.ToArray(), actualOutputs.ToArray());
    }


    /// <summary>
    /// 
    /// </summary>
    /// <param name="trainingData">Array of training input-output pairs</param>
    /// <param name="maxEpochs">Number of epochs to train over training data</param>
    /// <param name="learningRate">Learning rate</param>
    /// <param name="momentum">Amount the learning rate changes</param>
    /// <returns></returns>
    public void Train((float[], float[])[] trainingData, int batchSize, int maxEpochs, float learningRate, float momentum, int maxThreads = 8)
    {
        int errInterval = 8; // interval to check error
        int numBatches = trainingData.Length / batchSize;

        // train using back-prop

        // back-prop momentum specific arrays 
        float[][] ihPrevWeightsDelta = MakeMatrix(inputCount, hiddenCount);
        float[] hPrevBiasesDelta = new float[hiddenCount];
        float[][] hoPrevWeightsDelta = MakeMatrix(hiddenCount, outputCount);
        float[] oPrevBiasesDelta = new float[outputCount];

        // back-prop specific arrays
        float[][] hoGrads = MakeMatrix(hiddenCount, outputCount); // hidden-to-output weight gradients
        float[] obGrads = new float[outputCount];                   // output bias gradients

        float[][] ihGrads = MakeMatrix(inputCount, hiddenCount);  // input-to-hidden weight gradients
        float[] hbGrads = new float[hiddenCount];                   // hidden bias gradients

        // local gradient arrays (one index per thread)
        float[][][] threadHoGrads = Enumerable.Range(0, maxThreads).Select(x => MakeMatrix(hiddenCount, outputCount)).ToArray(); // hidden-to-output weight gradients
        float[][] threadObGrads = Enumerable.Range(0, maxThreads).Select(x => new float[outputCount]).ToArray();                   // output bias gradients

        float[][][] threadIhGrads = Enumerable.Range(0, maxThreads).Select(x => MakeMatrix(inputCount, hiddenCount)).ToArray();  // input-to-hidden weight gradients
        float[][] threadHbGrads = Enumerable.Range(0, maxThreads).Select(x => new float[hiddenCount]).ToArray();

        long totalTime = 0;
        int epoch = 0;
        while (epoch < maxEpochs)
        {
            ++epoch;

            if (epoch % errInterval == 0 && epoch < maxEpochs)
            {
                /*trainingData.Select(e => e.Item2).ToList().ForEach(p => Console.WriteLine(p[0]));*/
                float trainErr = BatchError(trainingData.Take(trainingData.GetLength(0)).ToArray())[0];
                //float trainErr = AggregateMeanSquaredError(trainingData.Select(e => e.Item2).ToArray(), allOutputs)[0];
                Console.WriteLine($"epoch = {epoch} error = {trainErr.ToString("F4")}");
                //Console.ReadLine();
            }

            Shuffle<(float[], float[])>(random, trainingData);

            ClearMatrix(ihPrevWeightsDelta);
            Array.Clear(hPrevBiasesDelta, 0, hiddenCount);
            ClearMatrix(hoPrevWeightsDelta);
            Array.Clear(oPrevBiasesDelta, 0, outputCount);

            Stopwatch sw = Stopwatch.StartNew();
            for (int b = 0; b < numBatches; ++b)
            {
                ClearMatrix(hoGrads);
                Array.Clear(obGrads, 0, outputCount);
                ClearMatrix(ihGrads);
                Array.Clear(hbGrads, 0, hiddenCount);

                Parallel.For(0, maxThreads, threadNum =>
                {
                    float[][] localHoGrads = threadHoGrads[threadNum]; // hidden-to-output weight gradients
                    float[] localObGrads = threadObGrads[threadNum];                   // output bias gradients
                    float[][] localIhGrads = threadIhGrads[threadNum];  // input-to-hidden weight gradients
                    float[] localHbGrads = threadHbGrads[threadNum];

                    ClearMatrix(localHoGrads);
                    Array.Clear(localObGrads, 0, outputCount);
                    ClearMatrix(localIhGrads);
                    Array.Clear(localHbGrads, 0, hiddenCount);

                    float derivative = 0.0f;
                    float errorSignal = 0.0f;

                    float[] oSignals = new float[outputCount];                  // local gradient output signals - gradients w/o associated input terms
                    float[] hSignals = new float[hiddenCount];                  // local gradient hidden node signals

                    int localBatchSize = batchSize / maxThreads;
                    int startIndex = threadNum * localBatchSize;
                    if (threadNum == maxThreads - 1) localBatchSize += batchSize % maxThreads; // if last thread, do the remaining batches
                    int endIndex = startIndex + localBatchSize;
                    for (int s = startIndex; s < endIndex; ++s)
                    {
                        int index = b * batchSize + s;
                        float[] inputValues = trainingData[index].Item1; // inputs
                        float[] targetOutput = trainingData[index].Item2; // target values
                        float[] actualOutput = PropogateForward(inputValues);
                        // CPropogateForward(inputValues, hiddenLayer, hiddenCount, hiddenInputWeights, hiddenBiases, outputCount, hiddenOutputWeights, outputBiases, actualOutput); // actual output values
                        /*                    lock (allOutputs)
                                                allOutputs[index] = actualOutput;*/

                        // 1. compute output node signals
                        for (int o = 0; o < outputCount; ++o)
                        {
                            errorSignal = targetOutput[o] - actualOutput[o];
                            derivative = TanhActivationDerivative(actualOutput[o]);
                            oSignals[o] = errorSignal * derivative;
                            /* Console.WriteLine($"Target: {targetOutput[o]} Actual: {actualOutput[o]} Error: {errorSignal}");*/
                        }

                        // 2. compute hidden-output weight gradients using output signals
                        for (int o = 0; o < outputCount; ++o)
                        {
                            for (int h = 0; h < hiddenCount; ++h)
                                localHoGrads[h][o] += oSignals[o] * hiddenLayer[h];

                            // 2b. compute output bias gradients using output signals
                            localObGrads[o] += oSignals[o] * 1;
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
                                localIhGrads[i][h] += hSignals[h] * inputValues[i];

                            localHbGrads[h] += hSignals[h] * 1; // dummy 1.0 input
                        }
                    }
                });

                // Console.WriteLine($"Num Threads: {maxThreads}, Num localGrads: {localHoGrads.Values.Count}");
                // Add each local gradient to global gradients
                foreach (float[][] localHoGrad in threadHoGrads)
                {
                    for (int i = 0; i < hiddenCount; ++i)
                        for (int j = 0; j < outputCount; ++j)
                            hoGrads[i][j] += localHoGrad[i][j];
                }
                foreach (float[] localObGrad in threadObGrads)
                {
                    for (int o = 0; o < outputCount; ++o)
                        obGrads[o] += localObGrad[o];
                }
                foreach (float[][] localIhGrad in threadIhGrads)
                {
                    for (int i = 0; i < inputCount; ++i)
                        for (int h = 0; h < hiddenCount; ++h)
                            ihGrads[i][h] += localIhGrad[i][h];
                }
                foreach (float[] localHbGrad in threadHbGrads)
                {
                    for (int h = 0; h < hiddenCount; ++h)
                        hbGrads[h] += localHbGrad[h];
                }

                // Average the global gradients actross batch size
                for (int o = 0; o < outputCount; ++o)
                {
                    obGrads[o] /= batchSize;
                    for (int h = 0; h < hiddenCount; ++h)
                        hoGrads[h][o] /= batchSize;
                }
                for (int h = 0; h < hiddenCount; ++h)
                {
                    hbGrads[h] /= batchSize;
                    for (int i = 0; i < inputCount; ++i)
                        ihGrads[i][h] /= batchSize;
                }

                // update input-to-hidden weights
                for (int i = 0; i < inputCount; ++i)
                {
                    for (int h = 0; h < hiddenCount; ++h)
                    {
                        float delta = ihGrads[i][h] * learningRate;
                        hiddenInputWeights[h][i] += delta; // would be -= if (o-t)
                        hiddenInputWeights[h][i] += ihPrevWeightsDelta[i][h] * momentum;
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

            sw.Stop();
            Console.WriteLine($"{numBatches} batches completed in {sw.ElapsedMilliseconds}ms");
            totalTime += sw.ElapsedMilliseconds;
        }
        Console.WriteLine($"Average batch proccessing time: {totalTime / maxEpochs}ms");
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
        return (float)(1 - (tanh * tanh));
    }

    /// <summary>
    /// Element-wise MeanSquaredError between lists of expected and actual vectors.
    /// </summary>
    /// <param name="expected"></param>
    /// <param name="actual"></param>
    /// <returns>List of Mean Squared error for each output node.</returns>
    private static float[] AggregateMeanSquaredError(float[][] expected, float[][] actual)
    {

        float[][] squaredErrors = actual.Zip(expected, (Xl, Yl) => Xl.Zip(Yl, (x, y) => (x - y) * (x - y)).ToArray()).ToArray();
        int n = expected.Count();
        int m = squaredErrors[0].Count();
        float[] sumSquaredErrors = Enumerable.Range(0, m)
            .Select(j => squaredErrors.Sum(sublist => sublist[j] / n))
            .ToArray();
        return sumSquaredErrors;
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

    private static void ClearMatrix(float[][] matrix)
    {
        foreach (float[] row in matrix)
        {
            Array.Clear(row, 0, row.Length);
        }
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

    /// <summary>
    /// Loads a pre-trained neural network from a `*.tinn` file.
    /// </summary>
    /// <param name="path">An absolute or a relative path to the `*.tinn` file.</param>
    /// <param name="seed">A seed for random generator to produce predictable results.</param>
    /// <returns>An instance of a pre-trained <see cref="TinyNeuralNetwork"/>.</returns>
    public static NeuralNetwork Load(string path, int seed = default)
    {
        using var reader = new StreamReader(path);
        var metaData = ReadLine();
        var counts = metaData.Split(' ').Select(int.Parse).ToList();
        var inputCount = counts[0];
        var hiddenCount = counts[1];
        var outputCount = counts[2];

        float[][] hiWeights = MakeMatrix(hiddenCount, inputCount);
        float[] hiddenBiases = new float[hiddenCount];
        float[][] hoWeights = MakeMatrix(hiddenCount, outputCount);
        float[] outBiases = new float[outputCount];

        hiddenBiases = ReadLine().Split(' ').Select(float.Parse).ToArray();
        for (var h = 0; h < hiddenCount; h++)
        {
            hiWeights[h] = ReadLine().Split(' ').Select(float.Parse).ToArray();
        }
        outBiases = ReadLine().Split(' ').Select(float.Parse).ToArray();
        for (var h = 0; h < hiddenCount; h++)
        {
            hoWeights[h] = ReadLine().Split(' ').Select(float.Parse).ToArray();
        }

        return new NeuralNetwork(inputCount, outputCount, hiWeights, hiddenBiases, hoWeights, outBiases, seed);

        string ReadLine()
        {
            return reader.ReadLine() ?? throw new ArgumentException($"Corrupted file '{path ?? ""}', missing data.");
        }
    }
    /// <summary>
    /// Saves a trained neural network to a file.
    /// </summary>
    /// <param name="path">An absolute or a relative path to the neural network file.</param>
    public void Save(string path, bool backup = false)
    {
        if (path == null) throw new ArgumentNullException("path");
        if (backup && File.Exists(path))
        {
            string directory = Path.GetDirectoryName(path);
            if (directory == null) throw new DirectoryNotFoundException("path");
            string extension = Path.GetExtension(path);
            string filenameWithoutExtension = Path.GetFileNameWithoutExtension(path);
            string newFilename = $"{filenameWithoutExtension}_{DateTime.Now:yyyy-MM-dd_HH-mm-ss-fff}{extension}";
            string newFilepath = Path.Combine(directory, newFilename);

            File.Move(path, newFilepath);
        }
        using var writer = new FormattingStreamWriter(path, CultureInfo.InvariantCulture);
        writer.WriteLine($"{inputCount} {hiddenCount} {outputCount}");

        // Write Hidden Layer Biases
        writer.WriteLine(String.Join(' ', hiddenBiases));
        // Write Hidden Layer Weights (1 line per hidden neuron)
        foreach (var hiddenInputWeight in hiddenInputWeights)
        {
            writer.WriteLine(String.Join(' ', hiddenInputWeight));
        }
        // Write Output Layer Biases
        writer.WriteLine(String.Join(' ', outputBiases));
        // Write Output Layer Weights (1 line per output neuron)
        foreach (var hiddenOutputWeight in hiddenOutputWeights)
        {
            writer.WriteLine(String.Join(' ', hiddenOutputWeight));
        }
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

