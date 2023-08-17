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

    static class BotTrainer
    {
        const string trainingPath = "D:\\Documents\\_Programming\\C# Projects\\Chess-Challenge\\Chess-Challenge\\src\\My Bot\\training\\chessData.csv";
        private static string[,] fenEvalArray;
        private static Random random = new Random();
        public static void Main(string[] args)
        {
            Console.WriteLine("Yo this worked pog");
            MyBot bot = new MyBot();
            LoadCSVToArray(trainingPath);
            var randomPair = GetRandomFENEvalPair();
            Console.WriteLine($"Random FEN: {randomPair.Item1}, Eval: {randomPair.Item2}");
            Console.WriteLine(evalStr_to_float(randomPair.Item2));
            Console.WriteLine("Press anything");
            Console.ReadKey();
        }

        public static void LoadCSVToArray(string path)
        {
            var records = new List<FenEvaluation>();

            using (var reader = new StreamReader(path))
            using (var csv = new CsvReader(reader, new CsvConfiguration(System.Globalization.CultureInfo.InvariantCulture)))
            {
                records = csv.GetRecords<FenEvaluation>().ToList();
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

        public static float evalStr_to_float(string evalStr, int alpha = 300)
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