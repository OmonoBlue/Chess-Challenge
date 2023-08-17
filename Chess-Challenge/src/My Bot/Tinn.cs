using System;
using System.IO;


public class Tinn
{
    private float[] w;     // All the weights.
    private float[] x;     // Hidden to output layer weights.
    private float[] b;     // Biases.
    private float[] h;     // Hidden layer.
    private float[] o;     // Output layer.
    private int nb;        // Number of biases - always two - Tinn only supports a single hidden layer.
    private int nw;        // Number of weights.
    private int nips;      // Number of inputs.
    private int nhid;      // Number of hidden neurons.
    private int nops;      // Number of outputs.

    // Computes error.
    private static float Err(float a, float b)
    {
        return 0.5f * (a - b) * (a - b);
    }

    // Returns partial derivative of error function.
    private static float PdErr(float a, float b)
    {
        return a - b;
    }

    // Computes total error of target to output.
    private static float TotErr(float[] tg, float[] o)
    {
        float sum = 0.0f;
        for (int i = 0; i < o.Length; i++)
            sum += Err(tg[i], o[i]);
        return sum;
    }

    // Activation function.
    private static float Act(float a)
    {
        return 1.0f / (1.0f + (float)Math.Exp(-a));
    }

    // Returns partial derivative of activation function.
    private static float PdAct(float a)
    {
        return a * (1.0f - a);
    }

    // Returns floating point random from 0.0 - 1.0.
    private static float Frand()
    {
        Random rand = new Random();
        return (float)rand.NextDouble();
    }

    // Performs back propagation.
    private void BProp(float[] input, float[] target, float rate)
    {
        for (int i = 0; i < nhid; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < nops; j++)
            {
                float a = PdErr(o[j], target[j]);
                float b = PdAct(o[j]);
                sum += a * b * x[j * nhid + i];
                x[j * nhid + i] -= rate * a * b * h[i];
            }
            for (int j = 0; j < nips; j++)
                w[i * nips + j] -= rate * sum * PdAct(h[i]) * input[j];
        }
    }

    // Performs forward propagation.
    private void FProp(float[] input)
    {
        for (int i = 0; i < nhid; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < nips; j++)
                sum += input[j] * w[i * nips + j];
            h[i] = Act(sum + b[0]);
        }
        for (int i = 0; i < nops; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < nhid; j++)
                sum += h[j] * x[i * nhid + j];
            o[i] = Act(sum + b[1]);
        }
    }

    // Randomizes tinn weights and biases.
    private void WBrand()
    {
        Random rand = new Random();
        for (int i = 0; i < nw; i++)
            w[i] = Frand() - 0.5f;
        for (int i = 0; i < nb; i++)
            b[i] = Frand() - 0.5f;
    }

    // Returns an output prediction given an input.
    public float[] Predict(float[] input)
    {
        FProp(input);
        return o;
    }

    // Trains a neural network with an input and target output with a learning rate. 
    // Returns target to output error.
    public float Train(float[] input, float[] target, float rate)
    {
        FProp(input);
        BProp(input, target, rate);
        return TotErr(target, o);
    }

    // Constructs a neural network with a specified number of inputs, hidden neurons, and outputs.
    public Tinn(int inputCount, int hiddenCount, int outputCount)
    {
        nb = 2;  // Tinn only supports one hidden layer so there are two biases.
        nw = hiddenCount * (inputCount + outputCount);
        w = new float[nw];
        x = new float[hiddenCount * outputCount];
        b = new float[nb];
        h = new float[hiddenCount];
        o = new float[outputCount];
        nips = inputCount;
        nhid = hiddenCount;
        nops = outputCount;
        WBrand();
    }

    // Saves the neural network to disk.
    public void Save(string path)
    {
        using (StreamWriter writer = new StreamWriter(path))
        {
            // Save header
            writer.WriteLine($"{nips} {nhid} {nops}");
            // Save biases
            foreach (float bias in b)
                writer.WriteLine(bias);
            // Save weights
            foreach (float weight in w)
                writer.WriteLine(weight);
        }
    }

    // Loads the neural network from disk.
    public static Tinn Load(string path)
    {
        using (StreamReader reader = new StreamReader(path))
        {
            // Load header
            string[] headerParts = reader.ReadLine().Split(' ');
            int nips = int.Parse(headerParts[0]);
            int nhid = int.Parse(headerParts[1]);
            int nops = int.Parse(headerParts[2]);

            Tinn t = new Tinn(nips, nhid, nops);

            // Load biases
            for (int i = 0; i < t.nb; i++)
                t.b[i] = float.Parse(reader.ReadLine());

            // Load weights
            for (int i = 0; i < t.nw; i++)
                t.w[i] = float.Parse(reader.ReadLine());

            return t;
        }
    }

    // Helper function to print an array of floats.
    private static void PrintArray(float[] arr, string label)
    {
        Console.WriteLine($"{label}:");
        foreach (float value in arr)
            Console.Write($"{value:F4} ");
        Console.WriteLine();
        Console.WriteLine(new string('-', 40));  // Separator for clarity
    }

    // Prints the weights and biases in a nicely formatted way.
    public void PrintWeightsAndBiases()
    {
        PrintArray(w, "Weights");
        PrintArray(b, "Biases");
    }

}

