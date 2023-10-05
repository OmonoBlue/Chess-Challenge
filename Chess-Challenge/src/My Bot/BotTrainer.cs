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
using System.Dynamic;
using System.IO.Compression;

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

    public class InputOutputPair
    {
        [TypeConverter(typeof(BotInputConverter))]
        public float[] Input { get; set; }
        public float Evaluation { get; set; }
    }

    public class BotInputConverter : DefaultTypeConverter
    {
        private ByteArrayConverter converter = new ByteArrayConverter(ByteArrayConverterOptions.Base64);
        
        public override object? ConvertFromString(string text, IReaderRow row, MemberMapData memberMapData)
        {
            byte[] byteArray = Decompress((byte[])converter.ConvertFromString(text, row, memberMapData));
            float[] floatArray = new float[byteArray.Length / sizeof(float)];
            Buffer.BlockCopy(byteArray, 0, floatArray, 0, byteArray.Length);
            return floatArray;
        }

        public override string? ConvertToString(object? value, IWriterRow row, MemberMapData memberMapData)
        {
            if (value.GetType() == typeof(float[]))
            {
                float[] floatArray = (float[])value;
                byte[] byteArray = new byte[floatArray.Length * sizeof(float)];
                Buffer.BlockCopy(floatArray, 0, byteArray, 0, byteArray.Length);
                return converter.ConvertToString(Compress(byteArray), row, memberMapData);
            }
            return base.ConvertToString(value, row, memberMapData);
        }

        public static byte[] Compress(byte[] bytes)
        {
            using (var memoryStream = new MemoryStream())
            {
                using (var gzipStream = new GZipStream(memoryStream, CompressionLevel.Optimal))
                {
                    gzipStream.Write(bytes, 0, bytes.Length);
                }
                return memoryStream.ToArray();
            }
        }

        public static byte[] Decompress(byte[] bytes)
        {
            using (var memoryStream = new MemoryStream(bytes))
            {

                using (var outputStream = new MemoryStream())
                {
                    using (var decompressStream = new GZipStream(memoryStream, CompressionMode.Decompress))
                    {
                        decompressStream.CopyTo(outputStream);
                    }
                    return outputStream.ToArray();
                }
            }
        }
    }


    public class BotTrainer
    {
        [DllImport("NNMethods.dll", EntryPoint = "test", CallingConvention = CallingConvention.Cdecl)]
        private static extern int test(int val);

        const string trainingPath = "D:\\Documents\\_Programming\\C# Projects\\Chess-Challenge\\Chess-Challenge\\src\\My Bot\\training\\chessData.csv";
        const string proccessedTrainingPath = "D:\\Documents\\_Programming\\C# Projects\\Chess-Challenge\\Chess-Challenge\\src\\My Bot\\training\\chessData-Proccessed.csv";
        const string testDataPath = "D:\\Documents\\_Programming\\C# Projects\\Chess-Challenge\\Chess-Challenge\\src\\My Bot\\training\\random_evals.csv";
        const string tacticDataPath = "D:\\Documents\\_Programming\\C# Projects\\Chess-Challenge\\Chess-Challenge\\src\\My Bot\\training\\tactic_evals.csv";
        public const string modelPath = "D:\\Documents\\_Programming\\C# Projects\\Chess-Challenge\\Chess-Challenge\\src\\My Bot\\models\\savedmodel.nn";

        private static (string, float)[] fenEvalArray;
        private static (float[], float[])[] trainingData;
        private static Random random = new Random(1);
        private const int numToLoad = 5000000;
        public static void Main(string[] args)
        {
            Console.WriteLine("Converting...");
            ConvertToTrainingCSV(trainingPath, proccessedTrainingPath);
            return;
            LoadCSVToTrainingArray(proccessedTrainingPath, numToLoad);

            MyNeuralNetwork neuralNet = new(MyBot.NumInputs, MyBot.NumHiddenNeurons, 1);
            TrainMyNetwork(neuralNet, batchSize: 64, epochs: 16, learnRate: 0.03f, momentum: 0.9f, numThreads: 8);
            Console.WriteLine("Saving model...");
            neuralNet.Save(modelPath, true);

            LoadCSVToFenEvalArray(testDataPath, 100);
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


        public static void TrainMyNetwork(MyNeuralNetwork neuralNet, string datasetPath = trainingPath, int numDatapoints = 1000, int batchSize = 32, int epochs = 12, float learnRate = 0.01f, float momentum = 0.9f, int numThreads = 8)
        {
            Console.WriteLine("Loading dataset...");
            LoadCSVToFenEvalArray(datasetPath, numDatapoints);
            Console.WriteLine("Parsing dataset...");
            LoadTrainingDataFromFenEvalArray();
            TrainMyNetwork(neuralNet, batchSize, epochs, learnRate, momentum, numThreads);
        }

        /// <summary>
        /// Trains neural network using data in trainingData array. Assumes trainingData is already loaded
        /// </summary>
        /// <param name="neuralNet"></param>
        /// <param name="batchSize"></param>
        /// <param name="epochs"></param>
        /// <param name="learnRate"></param>
        /// <param name="momentum"></param>
        /// <param name="numThreads"></param>
        public static void TrainMyNetwork(MyNeuralNetwork neuralNet, int batchSize = 32, int epochs = 12, float learnRate = 0.01f, float momentum = 0.9f, int numThreads = 8)
        {
            Console.WriteLine("Starting training...");
            neuralNet.Train(trainingData, batchSize, epochs, learnRate, momentum, numThreads);
        }

        public static void LoadTrainingDataFromFenEvalArray()
        {
            trainingData = fenEvalArray.AsParallel().Select(pair => (MyBot.getInputs(ChessChallenge.API.Board.CreateBoardFromFEN(pair.Item1)), new float[] { pair.Item2 })).ToArray();
        }

        public static void LoadCSVToTrainingArray(string inPath = proccessedTrainingPath, int amount = -1)
        {
            var records = new Queue<InputOutputPair>();
            int rowCount = 0;
            int skipped = 0;
            using (var reader = new StreamReader(inPath))
            using (var csv = new CsvReader(reader, new CsvConfiguration(System.Globalization.CultureInfo.InvariantCulture)))
            {
                csv.Read();
                csv.ReadHeader();
                while (csv.Read()) // Read the CSV file line by line
                {
                    if (amount > 0 && rowCount - skipped >= amount) break;

                    try
                    {
                        records.Enqueue(csv.GetRecord<InputOutputPair>());
                        rowCount++;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error in row number {rowCount} of csv file. Total skipped: {++skipped}");
                    }
                }
            }
            trainingData = records.AsParallel().Select(record => (record.Input, new float[] { record.Evaluation })).ToArray();
        }

        public static void LoadCSVToFenEvalArray(string path, int amount = -1)
        {
            var records = new Queue<FenEvaluation>();
            int rowCount = 0;
            int skipped = 0;
            using (var reader = new StreamReader(path))
            using (var csv = new CsvReader(reader, new CsvConfiguration(System.Globalization.CultureInfo.InvariantCulture)))
            {
                csv.Read();
                csv.ReadHeader();
                while (csv.Read()) // Read the CSV file line by line
                {
                    if (amount > 0 && rowCount - skipped >= amount) break;
                    
                    try
                    {
                        records.Enqueue(csv.GetRecord<FenEvaluation>());
                        rowCount++;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error in row number {rowCount} of csv file. Total skipped: {++skipped}");
                    }
                }
            }
            fenEvalArray = records.Select(record => (record.FEN, record.Evaluation)).ToArray();
        }


        public static void SaveTrainingCSV(string outpath, int amount = -1)
        {
            IEnumerable<InputOutputPair> inOutPairs = fenEvalArray.Select(pair => new InputOutputPair
            {
                Input = MyBot.getInputs(ChessChallenge.API.Board.CreateBoardFromFEN(pair.Item1)),
                Evaluation = pair.Item2
            });
            if (amount > 0)
            {
                inOutPairs = inOutPairs.Take(amount);
            }

            using (var writer = new StreamWriter(outpath))
            using (var csv = new CsvWriter(writer, new CsvConfiguration(System.Globalization.CultureInfo.InvariantCulture)))
            {
                csv.WriteRecords(inOutPairs);
            }
        }

        public static void ConvertToTrainingCSV(string inpath, string outpath, int amount = -1)
        {
            Console.WriteLine("Loading dataset...");
            LoadCSVToFenEvalArray(inpath, amount);
            Console.WriteLine($"Converting and writing {(amount > 1 ? amount+" ":"")}input-output pairs...");
            SaveTrainingCSV(outpath, amount);
            Console.WriteLine($"Done! Saved to {outpath}");
        }

        public static (string, float) GetRandomFENEvalPair()
        {
            int randomIndex = random.Next(0, fenEvalArray.GetLength(0));
            return fenEvalArray[randomIndex];
        }

        
    }
}