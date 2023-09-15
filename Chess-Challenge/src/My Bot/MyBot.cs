using Chess_Challenge.src.My_Bot;
using ChessChallenge.API;
using Microsoft.CodeAnalysis;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

public class MyBot : IChessBot
{
    
    public static float[] PieceValues = { 0f, 0.125f, 0.25f, 0.375f, 0.625f, 0.75f, 1f };
    private NeuralNetwork neuralnet;
    private NeuralNetwork blackNeuralnet;
    public static int NumInputs = 776;
    public static int NumHiddenNeurons = 256;

    public MyBot()
    {
        try
        {
            neuralnet = NeuralNetwork.Load(BotTrainer.modelPath);
            blackNeuralnet = NeuralNetwork.Load("D:\\Documents\\_Programming\\C# Projects\\Chess-Challenge\\Chess-Challenge\\src\\My Bot\\models\\bigrandomodel.nn");
        }
        catch (Exception e)
        {
            Console.WriteLine($"{e} Model loading failed, using random model");
            neuralnet = new NeuralNetwork(NumInputs, NumHiddenNeurons, 1);
        }
        
        //neuralnet.PrintWeightsAndBiases();
        
    }

    //public static float[] getInputs(Board board)
    //{
    //    /* NN INPUTS, array of floats.
    //    * 
    //    * board coordinates are nomalized from (1-8)/(a-h) -> (0.125-1)
    //    *      eg. coordinates e4 (5,4) -> (0.625,0.5)
    //    *      
    //    * 1x currentTurn (0:black 1:white)
    //    * 1x moveCount (0, 100) ply mapped to -> (-1, 1)
    //    * 1x isInCheck (-1 black check, 1 white check, 0 neither)
    //    * 1x enPassantSquare. Target en passant target file mapped from (a-h) -> (0.125-1). -1 means no en passant
    //    * 2x whiteCastleRights (kingside 0/1, queenside 0/1)
    //    * 2x blackCastleRights (kingside 0/1, queenside 0/1)
    //    * 64x boardState
    //    * 
    //    * BOARDSTATE FORMAT:
    //    * each piece has float value. White is positive, black is negative
    //    * squares listed in order a1-h1, followed by a2-h2, etc.
    //    * Piece Values:
    //    * pawn: 0.125
    //    * knigt: 0.25
    //    * bishop: 0.375
    //    * rook: 0.625
    //    * queen: 0.75
    //    * king: 1
    //    * 
    //    * NN OUTPUTS:
    //    * Board evaluation number
    //    * +1 = white winning overwhelmingly
    //    * -1 = black winning overwhelmingly
    //    */
        
    //    string FENstring = board.GetFenString();
    //    string enPassantSquare = FENstring.Split(' ')[3];
        

    //    List<float> inputs = new()
    //    {
    //        board.IsWhiteToMove ? 1f : 0f,
    //        tanh_sigmoid(board.PlyCount, 60), // Squish move count between 0 and 1, Alpha set to 60, average game is around 80, so 40-60 should be midgame.
    //        board.IsInCheck() ? (board.IsWhiteToMove ? 1f : -1f) : 0f,
    //        enPassantSquare == "-" ? -1f : SquareCoordinates(new Square(enPassantSquare))[0] // get en passant rank number
    //    };

    //    // get castling rights
    //    bool[] castleRights = { board.HasKingsideCastleRight(true), board.HasQueensideCastleRight(true), board.HasKingsideCastleRight(false), board.HasQueensideCastleRight(false) };
    //    inputs.AddRange(castleRights.Select(x => x ? 1f : 0f));

    //    // get board piece info
    //    for (int i = 0; i < 64; i++)
    //    {
    //        Piece currPiece = board.GetPiece(new Square(i));
    //        inputs.Add(PieceValues[(int)currPiece.PieceType] * (currPiece.IsNull||currPiece.IsWhite?1f:-1f));
    //    }

    //    return inputs.ToArray();
    //}

    public static float[] getInputs(Board board)
    {
        /* NN INPUTS, array of floats.
        * 
        * board coordinates are nomalized from (1-8)/(a-h) -> (0.125-1)
        *      eg. coordinates e4 (5,4) -> (0.625,0.5)
        *      
        * 1x currentTurn (0:black 1:white)
        * 1x moveCount (0, 100) ply mapped to -> (-1, 1)
        * 1x isInCheck (0 if current turn is not in check, 1 if in check)
        * 1x enPassantSquare. Target en passant target file mapped from (a-h) -> (0.125-1). -1 means no en passant
        * 2x whiteCastleRights (kingside 0/1, queenside 0/1)
        * 2x blackCastleRights (kingside 0/1, queenside 0/1)
        * 768x boardState
        * 
        * BOARDSTATE FORMAT:
        * bitboards for each piece type (Pawn, Knight, Bishop, Rook, Queen, King). White first then Black.
        * 
        * NN OUTPUTS:
        * Board evaluation number
        * +1 = white winning overwhelmingly
        * -1 = black winning overwhelmingly
        */

        string FENstring = board.GetFenString();
        string enPassantSquare = FENstring.Split(' ')[3];


        List<float> inputs = new()
        {
            board.IsWhiteToMove ? 1f : 0f,
            tanh_sigmoid(board.PlyCount, 60), // Squish move count between 0 and 1, Alpha set to 60, average game is around 80, so 40-60 should be midgame.
            board.IsInCheck() ? 1f : 0f,
            enPassantSquare == "-" ? -1f : SquareCoordinates(new Square(enPassantSquare))[0] // get en passant rank number
        };

        // get castling rights
        bool[] castleRights = { board.HasKingsideCastleRight(true), board.HasQueensideCastleRight(true), board.HasKingsideCastleRight(false), board.HasQueensideCastleRight(false) };
        inputs.AddRange(castleRights.Select(x => x ? 1f : 0f));


        // get all white and black piece bitboards (from pawn to king)
        bool pieceCol = true;
        for (int i = 0; i < 2; ++i)
        {
            for (PieceType piece = PieceType.Pawn; piece <= PieceType.King; ++piece)
            {
                BitArray bits = new(BitConverter.GetBytes(board.GetPieceBitboard(piece, pieceCol)));
                foreach (bool bit in bits)
                {
                    inputs.Add(bit ? 1f : 0f);
                }
            }
            pieceCol = false;
        }
        return inputs.ToArray();
    }

    public static float[] SquareCoordinates(Square square)
    {
        return new float[2] { (square.File + 1) / 8.0f, (square.Rank + 1) / 8.0f };
    }

    public static float tanh_sigmoid(float num, float alpha = 300f)
    {
        return (float)Math.Tanh(num / alpha);
    }

    public float GetMoveScore(Board board, Move move)
    {
        board.MakeMove(move);
        if (board.IsInCheckmate())
        {
            board.UndoMove(move);
            return 100f * (board.IsWhiteToMove ? 1f : -1f); // Make eval stupid high for checkmate so bot always picks (or ignores) checkmate move
        }
        float eval = neuralnet.Predict(getInputs(board))[0];
        board.UndoMove(move);
        return eval;
    }

    public Move Think(Board board, Timer timer)
    {
        bool botIsWhite = board.IsWhiteToMove;
        //if (botIsWhite)
        //{
        //    neuralnet = blackNeuralnet;
        //}
        Console.WriteLine(botIsWhite);
        Move bestMove = Move.NullMove;
        float bestScore = -2f;
        foreach (Move move in board.GetLegalMoves())
        {
            float moveScore = GetMoveScore(board, move) * (botIsWhite?1:-1);
            Console.WriteLine($"{move} score: {moveScore}");
            if (moveScore > bestScore)
            {
                bestMove = move;
                bestScore = moveScore;
                if (bestScore == 100f)
                {
                    Console.WriteLine("ayo checkmate lmfao");
                    break;
                }
            }
        }
        /*float[] inputs = getInputs(board);
        var nnOut = neuralnet.Predict(inputs);
        foreach (float item in inputs)
        {
            Console.WriteLine(item.ToString());
        }
        Console.WriteLine();
        Console.WriteLine(nnOut[0].ToString());
        Console.WriteLine();
        Move[] moves = board.GetLegalMoves();*/

        return bestMove;
    }

   
}