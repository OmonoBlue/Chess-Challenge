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
using System.Runtime.InteropServices;
using CsvHelper.TypeConversion;
using System.Text.Json.Nodes;
using System.Text.Json;
using CsvHelper.Configuration.Attributes;
using System.Transactions;

namespace Chess_Challenge.src.My_Bot
{
    public class FenEvaluation
    {
        public string FEN { get; set; }
        [TypeConverter(typeof(EvaluationConverter))]
        public float Evaluation { get; set; }

        public static float evalStr_to_float(string evalStr, float alpha = 300)
        {
            if (evalStr.StartsWith("#-"))
            {
                return -1f;
            }
            else if (evalStr.StartsWith("#+"))
            {
                return 1f;
            }
            int eval = Int32.Parse(evalStr, System.Globalization.NumberStyles.Integer);
            return (float)Math.Tanh(eval / alpha);
        }
    }

    public class EvaluationConverter : DefaultTypeConverter
    {
        public override object ConvertFromString(string text, IReaderRow row, MemberMapData memberMapData)
        {
            return FenEvaluation.evalStr_to_float(text);
        }
    }

    public class BotTrainer
    {
        [DllImport("NNMethods.dll", EntryPoint = "test", CallingConvention = CallingConvention.Cdecl)]
        private static extern int test(int val);


        const string trainingPath = "D:\\Documents\\_Programming\\C# Projects\\Chess-Challenge\\Chess-Challenge\\src\\My Bot\\training\\chessData.csv";
        const string testDataPath = "D:\\Documents\\_Programming\\C# Projects\\Chess-Challenge\\Chess-Challenge\\src\\My Bot\\training\\tactic_evals.csv";
        public const string modelPath = "D:\\Documents\\_Programming\\C# Projects\\Chess-Challenge\\Chess-Challenge\\src\\My Bot\\models\\savedmodel.nn";

        private static (string, float)[] fenEvalArray;
        private static Random random = new Random(1);
        private const int numToLoad = -1;
        public static void Main(string[] args)
        {
            NeuralNetwork neuralNet = new(MyBot.NumInputs, MyBot.NumHiddenNeurons, 1, 1);
            TrainMyNetwork(neuralNet, trainingPath, numToLoad, batchSize: 64, epochs: 64, learnRate: 0.05f, momentum: 0.9f, numThreads: 8);
            Console.WriteLine("Saving model...");
            neuralNet.Save(modelPath, true);

            var randomPair = GetRandomFENEvalPair();
            ChessChallenge.API.Board board = ChessChallenge.API.Board.CreateBoardFromFEN(randomPair.Item1);
            float[] testinput = MyBot.getInputs(board);
            float testtarget = randomPair.Item2;
            float prediction = 0;
            try
            {
                prediction = neuralNet.PropogateForward(testinput)[0];
            } catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
                
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

        public static void TestDLL()
        {
            try
            {
                int result = test(4);
                Console.WriteLine("Result: " + result);
            }
            catch (DllNotFoundException e)
            {
                Console.WriteLine("DLL not found: " + e.Message);
            }
            catch (Exception e)
            {
                Console.WriteLine("An error occurred: " + e.Message);
            }
        }
        public static void TrainMyNetwork(NeuralNetwork neuralNet, string datasetPath = trainingPath, int numDatapoints = 1000, int batchSize = 32, int epochs = 12, float learnRate = 0.01f, float momentum = 0.9f, int numThreads = 8)
        {
            Console.WriteLine("Loading dataset...");
            LoadCSVToArray(datasetPath, numDatapoints);
            Console.WriteLine("Parsing dataset...");
            (float[], float[])[] data = fenEvalArray.Select(pair => (MyBot.getInputs(ChessChallenge.API.Board.CreateBoardFromFEN(pair.Item1)), new float[] { pair.Item2 })).ToArray();
            Console.WriteLine("Starting training...");
            neuralNet.Train(data, batchSize, epochs, learnRate, momentum, numThreads);
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
                    (string, float) currPair = GetRandomFENEvalPair();
                    try
                    {
                        input = MyBot.getInputs(ChessChallenge.API.Board.CreateBoardFromFEN(currPair.Item1));
                        target = new float[]{currPair.Item2};
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
                float testtarget = randomPair.Item2;
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
                csv.Read();
                csv.ReadHeader();

                int rowCount = 0;
                int skipped = 0;
                while (csv.Read()) // Read the CSV file line by line
                {
                    if (amount > 0 && rowCount - skipped >= amount) break;
                    
                    try
                    {
                        records.Add(csv.GetRecord<FenEvaluation>());
                        rowCount++;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error in row number {rowCount} of csv file. Total skipped: {++skipped}");
                    }
                }
            }

            fenEvalArray = new (string, float)[records.Count];
            for (int i = 0; i < records.Count; i++)
            {
                fenEvalArray[i] = (records[i].FEN, records[i].Evaluation);
            }
        }

        public static (string, float) GetRandomFENEvalPair()
        {
            int randomIndex = random.Next(0, fenEvalArray.GetLength(0));
            return fenEvalArray[randomIndex];
        }

        
    }
}