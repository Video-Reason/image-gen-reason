# Python-Chess Integration for VMEvalKit

## Overview

The `python-chess` library provides comprehensive chess functionality that will enable VMEvalKit to create sophisticated chess reasoning evaluation tasks. This library allows us to generate chess problems where video models must analyze board positions and demonstrate reasoning by generating videos that show solutions.

## Key Capabilities for Video Reasoning Evaluation

### 1. Board Representation & Manipulation

**Core Features:**
- Complete chess board representation with all pieces
- Legal move generation and validation
- Game state detection (checkmate, stalemate, draws)
- Support for chess variants (Chess960, King of the Hill, etc.)

**VMEvalKit Applications:**
- Generate diverse chess positions for evaluation
- Validate that video model solutions are legal moves
- Create graduated difficulty problems from beginner to master level

```python
import chess

# Create starting position
board = chess.Board()

# Generate legal moves
legal_moves = list(board.legal_moves)

# Apply moves and check game states
board.push_san("e4")
board.push_san("e5")
is_checkmate = board.is_checkmate()
```

### 2. Chess Notation Formats

**Supported Formats:**
- **FEN (Forsyth-Edwards Notation)**: Complete board position encoding
- **PGN (Portable Game Notation)**: Full game records with metadata
- **EPD (Extended Position Description)**: Position with analytical annotations
- **SAN (Standard Algebraic Notation)**: Human-readable move notation

**VMEvalKit Applications:**
- Store chess problems in standardized formats
- Create problem databases with varying difficulty
- Generate text prompts with chess notation for models

```python
# FEN for board positions
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
board = chess.Board(fen)

# SAN for moves
move = board.parse_san("Nf3")
san_notation = board.san(move)
```

### 3. Visual Board Rendering

**Capabilities:**
- **SVG Generation**: High-quality scalable board images
- **ASCII Output**: Terminal-friendly board display
- **Customizable Styling**: Colors, piece sets, board orientation

**VMEvalKit Applications:**
- Generate input images for video models showing chess positions
- Create consistent visual representations for evaluation
- Render board states at different points in solutions

```python
import chess.svg

# Generate SVG image of board position
board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3")
svg = chess.svg.board(board)

# ASCII representation
print(board)
```

### 4. Chess Problem Types for Video Reasoning

**Tactical Problems:**
- **Checkmate in N**: Find forced checkmate sequences
- **Pin and Skewer**: Exploit piece relationships
- **Fork and Discovery**: Multi-piece attacks
- **Endgame Studies**: Precise technique required

**Strategic Evaluation:**
- **Best Move Selection**: Choose optimal moves from positions
- **Plan Recognition**: Show multi-move strategic ideas
- **Positional Assessment**: Evaluate position strengths/weaknesses

### 5. EPD Test Suites

The library includes extensive test suites:
- **Bratko-Kopec Test**: 24 tactical positions
- **Endgame Studies**: Tablebase-verified positions  
- **Positional Tests**: Strategic evaluation problems

**VMEvalKit Integration:**
```python
# Load EPD test suite
with open("submodules/python-chess/data/endgame.epd") as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            board, epd_info = chess.Board.from_epd(line)
            # Extract best moves, avoid moves, etc.
            best_moves = epd_info.get('bm', [])
            avoid_moves = epd_info.get('am', [])
```

### 6. Engine Integration

**Features:**
- UCI/XBoard engine communication
- Position analysis with depth/time limits
- Move evaluation and principal variation extraction

**VMEvalKit Applications:**
- Verify video model solutions against engine analysis
- Generate ground truth for chess problem difficulty rating
- Create reference solutions for comparison

```python
import chess.engine

# Analyze position with Stockfish
engine = chess.engine.SimpleEngine.popen_uci("stockfish")
board = chess.Board("1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1")
result = engine.play(board, chess.engine.Limit(time=2.0))
best_move = result.move
```

### 7. Endgame Tablebase Support

**Capabilities:**
- **Syzygy Tablebase**: Perfect endgame play up to 7 pieces
- **Gaviota Tablebase**: Distance-to-mate information
- **Perfect Play Verification**: Absolute correctness for endings

**VMEvalKit Applications:**
- Create endgame problems with guaranteed correct solutions
- Evaluate video model accuracy in theoretical positions
- Generate learning sequences showing optimal play

```python
import chess.syzygy

# Probe endgame tablebase
tablebase = chess.syzygy.open_tablebase("submodules/python-chess/data/syzygy/regular")
board = chess.Board("8/2K5/4B3/3N4/8/8/4k3/8 b - - 0 1")
dtz = tablebase.probe_dtz(board)  # Distance to zeroing move
```

## Proposed Chess Reasoning Tasks for VMEvalKit

### Task 1: Tactical Pattern Recognition
**Input**: Chess position image with tactical motif
**Prompt**: "Show the winning tactical sequence"
**Expected Output**: Video demonstrating the tactical solution with piece movements

### Task 2: Checkmate Sequence
**Input**: Position where checkmate is possible
**Prompt**: "Find checkmate in 3 moves"
**Expected Output**: Video showing the complete checkmate sequence

### Task 3: Endgame Technique
**Input**: Theoretical endgame position
**Prompt**: "Demonstrate the winning technique"
**Expected Output**: Video showing optimal endgame play

### Task 4: Opening Principles
**Input**: Early game position
**Prompt**: "Show the best developing moves"
**Expected Output**: Video demonstrating sound opening development

### Task 5: Strategic Planning
**Input**: Complex middlegame position
**Prompt**: "Execute a winning plan"
**Expected Output**: Video showing multi-move strategic execution

## Integration Architecture

### Chess Task Generator
```python
class ChessTaskGenerator:
    def __init__(self):
        self.board = chess.Board()
        self.engine = None
        
    def generate_tactical_puzzle(self, difficulty='medium'):
        """Generate tactical puzzle with specified difficulty"""
        pass
        
    def create_position_image(self, board_state):
        """Create SVG/PNG image of chess position"""
        pass
        
    def validate_solution(self, video_moves, expected_solution):
        """Validate video model's solution against correct moves"""
        pass
```

### Evaluation Metrics
- **Move Accuracy**: Percentage of correct moves
- **Sequence Completeness**: Full solution demonstrated
- **Legal Move Validation**: All moves are chess-legal
- **Tactical Pattern Recognition**: Correct motif identification
- **Strategic Understanding**: Long-term plan execution

## Data Assets Available

The submodule includes rich datasets:
- **Historical Games**: Famous games (Kasparov vs Deep Blue, etc.)
- **Puzzle Collections**: Bratko-Kopec, endgame studies
- **Opening Books**: Polyglot format opening databases
- **Tablebase Data**: Perfect endgame play references

## Installation & Setup

The library is already integrated as a submodule. To use:

```python
import sys
sys.path.append('/Users/access/VMEvalKit/submodules/python-chess')
import chess
import chess.svg
import chess.engine
```

## Next Steps for VMEvalKit Integration

1. **Create Chess Task Module**: `vmevalkit/tasks/chess_reasoning.py`
2. **Generate Test Dataset**: Convert EPD puzzles to video reasoning tasks
3. **Implement Evaluation Pipeline**: Validate model solutions
4. **Add Chess Visualization**: Generate consistent board images
5. **Engine Integration**: Set up Stockfish for solution verification

This integration will significantly expand VMEvalKit's reasoning evaluation capabilities, adding a rich domain of logical, tactical, and strategic thinking that's perfect for testing video model intelligence.
