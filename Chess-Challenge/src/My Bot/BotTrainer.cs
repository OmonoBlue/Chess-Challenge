using ChessChallenge.API;
using ChessChallenge.Application;
using ChessChallenge.Chess;
using System;
using CsvHelper;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using CsvHelper.Configuration;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Chess_Challenge.src.My_Bot
{
    public class FenEvaluation
    {
        public string FEN { get; set; }
        public string Evaluation { get; set; }
    }

    public class BotTrainer
    {
        const string trainingPath = "D:\\Documents\\_Programming\\C# Projects\\Chess-Challenge\\Chess-Challenge\\src\\My Bot\\training\\chessData.csv";
        const string testDataPath = "D:\\Documents\\_Programming\\C# Projects\\Chess-Challenge\\Chess-Challenge\\src\\My Bot\\training\\tactic_evals.csv";
        public const string modelPath = "D:\\Documents\\_Programming\\C# Projects\\Chess-Challenge\\Chess-Challenge\\src\\My Bot\\models\\savedmodel.tinn";

        private static string[,] fenEvalArray;
        private static Random random = new Random();
        private const int numToLoad = 10000;
        public static void Main(string[] args)
        {
            NeuralNetwork neuralNet = NeuralNetwork.Load(modelPath);
/*            TrainMyNetwork(neuralNet, trainingPath, numToLoad, 128);
            Console.WriteLine("Saving model...");*/
            neuralNet.Save(modelPath, true);
            return;
            var randomPair = GetRandomFENEvalPair();
            ChessChallenge.API.Board board = ChessChallenge.API.Board.CreateBoardFromFEN(randomPair.Item1);
            float[] testinput = MyBot.getInputs(board);
            float testtarget = evalStr_to_float(randomPair.Item2);
            float prediction = neuralNet.PropogateForward(testinput)[0];
            float error = (testtarget - prediction) * (testtarget - prediction);
/*            totalError += error;*/
        
            Console.WriteLine(board.CreateDiagram(true, false));
            Console.WriteLine($"Random FEN: {randomPair.Item1}, Eval: {randomPair.Item2}");
            Console.WriteLine($"Target Eval: {testtarget}");
            Console.WriteLine($"Bot's Eval: {prediction}");
            Console.WriteLine($"Error: {error}");
            Console.WriteLine();
            /*TinyNeuralNetwork neuralNet = new(MyBot.NumInputs, MyBot.NumHiddenNeurons, 1, random.Next());

            TrainNetwork(neuralNet, trainingPath, 0.05f, 1024, 0.999f, 10000);

            Console.WriteLine("Saving model");
            neuralNet.Save(modelPath, true);

            Console.WriteLine("Loading testing dataset...");
            LoadCSVToArray(trainingPath);

            TestModel(neuralNet, 20);*/
            Console.WriteLine("Press anything");
            Console.ReadKey();
            Console.ReadKey();
        }

        public static void TrainMyNetwork(NeuralNetwork neuralNet, string datasetPath = trainingPath, int numDatapoints = 1000, int iterations = 128, float learnRate = 0.01f, float momentum = 0.9f)
        {
            Console.WriteLine("Loading dataset...");
            LoadCSVToArray(datasetPath, numToLoad);
            (float[], float[])[] data = Enumerable.Range(0, numToLoad).Select(pair => GetRandomFENEvalPair()).Select(pair => (MyBot.getInputs(ChessChallenge.API.Board.CreateBoardFromFEN(pair.Item1)), new float[] { evalStr_to_float(pair.Item2) })).ToArray();
            Console.WriteLine("Starting training...");
            neuralNet.Train(data, iterations, learnRate, momentum);
        }

        public static void TrainNetwork(TinyNeuralNetwork network, string datasetPath = trainingPath, float rate = 0.1f, int iterations = 512, float anneal = 0.999f, int batch = 100)
        {
            // Hyper Parameters.
            // Learning rate is annealed and thus not constant.
            // It can be fine tuned along with the number of hidden layers.
            // Feel free to modify the anneal rate.
            // The number of iterations can be changed for stronger training.

            LoadCSVToArray(datasetPath);

            for (int i = 0; i < iterations; i++)
            {
                float error = 0.0f;
                int skipped = 0;
                for (int j = 0; j < batch; j++)
                {
                    float[] input;
                    float[] target;
                    (string, string) currPair = GetRandomFENEvalPair();
                    try
                    {
                        input = MyBot.getInputs(ChessChallenge.API.Board.CreateBoardFromFEN(currPair.Item1));
                        target = new float[]{evalStr_to_float(currPair.Item2) };
                    } catch (Exception e)
                    {
                        Console.WriteLine($"Error {e}\ninput: {currPair}");
                        ++skipped;
                        continue;
                    }
                    error += network.Train(input, target, rate);
                }
                Console.WriteLine($"{i+1}/{iterations}: error {(double)error / (batch-skipped)} :: learning rate {(double)rate} {(skipped>0?$"Skipped{skipped}":"")}");
                rate *= anneal;
            }
        }

        public static void TestModel(int numTests = 10, string testModelPath = modelPath, string testPath = testDataPath)
        {
            Console.WriteLine("Loading model...");
            TinyNeuralNetwork testNet = TinyNeuralNetwork.Load(testModelPath);

            Console.WriteLine("Loading testing dataset...");
            LoadCSVToArray(testPath);
            TestModel(testNet, numTests);
        }

        public static void TestModel(TinyNeuralNetwork testNet, int numTests = 10) {
            float totalError = 0.0f;
            for (int i = 0; i < numTests; i++)
            {
                var randomPair = GetRandomFENEvalPair();
                ChessChallenge.API.Board board = ChessChallenge.API.Board.CreateBoardFromFEN(randomPair.Item1);
                float[] testinput = MyBot.getInputs(board);
                float testtarget = evalStr_to_float(randomPair.Item2);
                float prediction = testNet.Predict(testinput)[0];
                float error = testNet.GetTotalError(new float[] { testtarget });
                totalError += error;

                Console.Write(board.CreateDiagram(true, false));
                Console.WriteLine($"Random FEN: {randomPair.Item1}, Eval: {randomPair.Item2}");
                Console.WriteLine($"Target Eval: {testtarget}");
                Console.WriteLine($"Bot's Eval: {prediction}");
                Console.WriteLine($"Error: {error}");
                Console.WriteLine();
            }
            Console.WriteLine($"Average Error: {totalError / numTests}");
        }

        public static void LoadCSVToArray(string path, int amount = -1)
        {
            var records = new List<FenEvaluation>();

            using (var reader = new StreamReader(path))
            using (var csv = new CsvReader(reader, new CsvConfiguration(System.Globalization.CultureInfo.InvariantCulture)))
            {
                if (amount > 0) {
                    records = csv.GetRecords<FenEvaluation>().Take(amount).ToList();
                }
                else
                {
                    records = csv.GetRecords<FenEvaluation>().ToList();
                }

            }

            fenEvalArray = new string[records.Count, 2];
            for (int i = 0; i < records.Count; i++)
            {
                fenEvalArray[i, 0] = records[i].FEN;
                fenEvalArray[i, 1] = records[i].Evaluation;
            }
        }

        public static (string, string) GetRandomFENEvalPair()
        {
            int randomIndex = random.Next(0, fenEvalArray.GetLength(0));
            return (fenEvalArray[randomIndex, 0], fenEvalArray[randomIndex, 1]);
        }

        public static float evalStr_to_float(string evalStr, float alpha = 300)
        {
            if (evalStr.StartsWith("#-"))
            {
                return -1f;
            } else if (evalStr.StartsWith("#+"))
            {
                return 1f;
            }
            int eval = Int32.Parse(evalStr, System.Globalization.NumberStyles.Integer);
            return (float)Math.Tanh(eval / alpha);
        }
    }
}