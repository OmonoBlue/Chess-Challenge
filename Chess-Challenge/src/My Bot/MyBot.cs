using ChessChallenge.API;
using Microsoft.CodeAnalysis;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

public class MyBot : IChessBot
{
    /* NN INPUTS, array of floats.
     * 
     * board coordinates are nomalized from (1-8)/(a-h) -> (0-1)
     *      eg. coordinates e4 (5,4) -> (0.625,0.5)
     *      
     * 1x currentTurn (0:black 1:white)
     * 4x lastMove (x, y cooridnates)
     * 2x whiteCastleRights (kingside 0/1, queenside 0/1)
     * 2x blackCastleRights (kingside 0/1, queenside 0/1)
     * 64x boardState
     * 
     * BOARDSTATE FORMAT:
     * each piece has float value. White is positive, black is negative
     * squares listed in order a1-h1, followed by a2-h2, etc.
     * Piece Values:
     * pawn: 0.125
     * knigt: 0.25
     * bishop: 0.375
     * rook: 0.625
     * queen: 0.75
     * king: 1
     * 
     * NN OUTPUTS:
     * Board evaluation number
     * +1 = white winning overwhelmingly
     * -1 = black winning overwhelmingly
     */
    float[] PieceValues = { 0f, 0.125f, 0.25f, 0.375f, 0.625f, 0.75f, 1f };
    private Tinn neuralnet;
    private const int NumInputs = 73;
    private const int NumHiddenNeurons = 1024;

    public MyBot()
    {
        neuralnet = new Tinn(NumInputs, NumHiddenNeurons, 1);
        //neuralnet.PrintWeightsAndBiases();
        
    }

    private float[] getInputs(Board board)
    { 
        List<float> inputs = new();

        inputs.Add(board.IsWhiteToMove ? 1 : 0);
        if (board.GameMoveHistory.Length > 0)
        {
            Move lastMove = board.GameMoveHistory[board.GameMoveHistory.Length - 1];
            inputs.AddRange(SquareCoordinates(lastMove.StartSquare));
            inputs.AddRange(SquareCoordinates(lastMove.TargetSquare));
        } else
        {
            inputs.AddRange(new float[] { 0f, 0f, 0f, 0f });
        }
        bool[] castleRights = { board.HasKingsideCastleRight(true), board.HasQueensideCastleRight(true), board.HasKingsideCastleRight(false), board.HasQueensideCastleRight(false) };
        inputs.AddRange(castleRights.Select(x => x ? 1f : 0f));
        for (int i = 0; i < 64; i++)
        {
            Piece currPiece = board.GetPiece(new Square(i));
            inputs.Add(PieceValues[(int)currPiece.PieceType] * (currPiece.IsNull||currPiece.IsWhite?1:-1));
        }
        return inputs.ToArray();
    }

    private float[] SquareCoordinates(Square square)
    {
        return new float[2] { square.File / 7.0f, square.Rank / 7.0f };
    }

    public static float stockfish_eval_to_float(int eval, int alpha = 300)
    {
        return (float)Math.Tanh(eval / alpha);
    }

    public Move Think(Board board, Timer timer)
    {
        var nnOut = neuralnet.Predict(getInputs(board));
        foreach (var item in nnOut)
        {
            Console.WriteLine(item.ToString());
        }
        

        Move[] moves = board.GetLegalMoves();
        return moves[0];
    }
}