#!/usr/bin/env python3
"""
Chess Reasoning Demo for VMEvalKit

This script demonstrates how to use the python-chess library to create
chess reasoning evaluation tasks for video models.

Usage:
    python examples/chess_reasoning_demo.py
"""

import sys
import os

# Add python-chess to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'submodules', 'python-chess'))

import chess
import chess.svg
import chess.pgn


def demo_basic_chess_operations():
    """Demonstrate basic chess board operations."""
    print("=== Basic Chess Operations ===")
    
    # Create a board
    board = chess.Board()
    print("Starting position:")
    print(board)
    print(f"FEN: {board.fen()}")
    print()
    
    # Show legal moves
    legal_moves = list(board.legal_moves)[:10]  # First 10 moves
    print("First 10 legal moves:")
    for move in legal_moves:
        print(f"  {board.san(move)} ({move.uci()})")
    print()
    
    # Make some moves
    print("Playing Scholar's Mate:")
    moves = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7"]
    
    for move_san in moves:
        move = board.push_san(move_san)
        print(f"  {move_san} -> {board.fen()}")
        
        # Check game state
        if board.is_checkmate():
            print(f"  Checkmate! {'White' if board.turn == chess.BLACK else 'Black'} wins!")
            break
    
    print()


def demo_chess_puzzle_creation():
    """Demonstrate creating chess puzzles from EPD format."""
    print("=== Chess Puzzle Creation ===")
    
    # Example tactical puzzle - White to move and win  
    epd = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - bm Ng5; id \"Fork Attack\";"
    
    try:
        board, epd_info = chess.Board.from_epd(epd)
        
        print("Puzzle position:")
        print(board)
        print(f"FEN: {board.fen()}")
        
        # Extract puzzle information
        best_moves = epd_info.get('bm', [])
        puzzle_id = epd_info.get('id', 'Unknown')
        
        print(f"Puzzle ID: {puzzle_id}")
        print(f"Best move(s): {[board.san(move) for move in best_moves]}")
        print()
        
        # Show why this move works
        if best_moves:
            best_move = best_moves[0]
            print(f"After {board.san(best_move)}:")
            board.push(best_move)
            print(board)
            print()
            
    except Exception as e:
        print(f"Error processing EPD: {e}")


def demo_endgame_position():
    """Demonstrate endgame position analysis."""
    print("=== Endgame Position Analysis ===")
    
    # Classic king and pawn endgame
    fen = "8/8/8/8/3k4/8/3K1P2/8 w - - 0 1"
    board = chess.Board(fen)
    
    print("King and Pawn endgame:")
    print(board)
    print(f"FEN: {fen}")
    
    # Show key squares concept
    print("White to move - can White promote the pawn?")
    print("Key analysis points:")
    print("- Opposition: Who has the opposition?")
    print("- Key squares: Can White's king reach the key squares?")
    print("- Pawn race: Can Black's king catch the pawn?")
    print()


def demo_chess_reasoning_task():
    """Demonstrate a complete chess reasoning task for video models."""
    print("=== Chess Reasoning Task Example ===")
    
    # Create a tactical position - discovered attack
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 6"
    board = chess.Board(fen)
    
    print("TASK: Find the winning tactical sequence")
    print("INPUT IMAGE: Chess position shown below")
    print(board)
    print()
    
    print("TEXT PROMPT: 'White to move. Find the tactic that wins material.'")
    print()
    
    print("EXPECTED VIDEO SOLUTION:")
    print("1. Show the current position")
    print("2. Highlight the key pieces involved in the tactic")
    print("3. Demonstrate the winning move sequence:")
    
    # Solution: Ng5! (attacking h7 and f7, creating tactical threats)
    solution_moves = ["Ng5"]  # This attacks f7 and h7, creating multiple threats
    
    for i, move_san in enumerate(solution_moves, 1):
        move = board.parse_san(move_san)
        print(f"   {i}. {move_san} - {get_move_explanation(board, move)}")
        board.push(move)
        
    print("4. Show the final position with material advantage")
    print()
    
    print("EVALUATION CRITERIA:")
    print("- ✓ Correctly identifies the tactical motif")
    print("- ✓ Plays legal moves only") 
    print("- ✓ Demonstrates the complete sequence")
    print("- ✓ Shows clear material/positional gain")
    print()


def get_move_explanation(board, move):
    """Get a simple explanation for a move."""
    piece = board.piece_at(move.from_square)
    if not piece:
        return "unknown move"
    
    piece_name = chess.piece_name(piece.piece_type).title()
    from_square = chess.square_name(move.from_square)
    to_square = chess.square_name(move.to_square)
    
    explanation = f"{piece_name} from {from_square} to {to_square}"
    
    # Check for special moves
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured:
            explanation += f", captures {chess.piece_name(captured.piece_type)}"
    
    if board.gives_check(move):
        explanation += ", gives check"
        
    return explanation


def demo_game_analysis():
    """Demonstrate analyzing a complete game."""
    print("=== Complete Game Analysis ===")
    
    # Load a famous game from the included PGN files
    pgn_path = os.path.join(os.path.dirname(__file__), '..', 'submodules', 'python-chess', 'data', 'pgn', 'molinari-bordais-1979.pgn')
    
    try:
        with open(pgn_path, 'r') as pgn_file:
            game = chess.pgn.read_game(pgn_file)
            
        if game:
            print(f"Game: {game.headers.get('White', '?')} vs {game.headers.get('Black', '?')}")
            print(f"Result: {game.headers.get('Result', '?')}")
            print(f"Date: {game.headers.get('Date', '?')}")
            print()
            
            print("Game moves:")
            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                if i % 2 == 0:
                    move_number = (i // 2) + 1
                    print(f"{move_number}. {board.san(move)}", end="")
                else:
                    print(f" {board.san(move)}")
                board.push(move)
            print()
            print()
            
            print("Final position:")
            print(board)
            
            if board.is_checkmate():
                print("Checkmate!")
            elif board.is_stalemate():
                print("Stalemate!")
                
    except FileNotFoundError:
        print(f"PGN file not found: {pgn_path}")
    except Exception as e:
        print(f"Error reading game: {e}")


def main():
    """Run all chess reasoning demonstrations."""
    print("Chess Reasoning Demo for VMEvalKit")
    print("=" * 50)
    print()
    
    demo_basic_chess_operations()
    demo_chess_puzzle_creation()
    demo_endgame_position()
    demo_chess_reasoning_task()
    demo_game_analysis()
    
    print("=" * 50)
    print("Demo completed!")
    print()
    print("Next steps for VMEvalKit integration:")
    print("1. Create chess task generator in vmevalkit/tasks/chess_reasoning.py")
    print("2. Implement board image generation for input images")
    print("3. Add move sequence validation for video outputs")
    print("4. Create comprehensive chess puzzle database")
    print("5. Integrate with existing VMEvalKit evaluation pipeline")


if __name__ == "__main__":
    main()
