# Abstract (Short Version - 250 words)

## Video Models Start to Solve Chess, Maze, Sudoku, Mental Rotation, and Raven's Matrices

Recent advances in video generation have produced visually compelling models, yet their capacity for visual reasoning—generating videos that demonstrate solutions to cognitive tasks—remains unexplored. We introduce VMEvalKit, a systematic framework evaluating whether video generation models can reason through five fundamental cognitive challenges: chess puzzles (strategic planning), maze navigation (spatial pathfinding), Sudoku (logical deduction), 3D mental rotation (spatial transformation), and Raven's matrices (abstract pattern recognition).

We provide the first evidence that state-of-the-art video models exhibit measurable reasoning capabilities. Evaluating 40 models across 11 families (Sora, Veo, Luma, Runway, and open-source alternatives), we find leading models achieve >60% success rates on reasoning tasks, with top performers solving complex tactics at near-human levels. This represents a fundamental shift from video models as pure generators to visual problem-solvers.

Our evaluation paradigm centers on "Task Pairs" (initial state, solution state, instruction), enabling objective automated evaluation by comparing generated endpoints against ground truth. Statistical validation shows automated GPT-4O evaluation strongly correlates with human judgment (r=0.949, κ=0.867, p=0.051), converging at 25 samples—demonstrating robustness for large-scale evaluation.

We release VMEvalKit, a modular open-source framework supporting extensible model integration (40 models), standardized task definition, and automated evaluation. The framework's clean architecture enables researchers to add cognitive tasks and models without modifying core infrastructure, facilitating collaborative benchmarking at scale. By demonstrating that objective automated evaluation is feasible, we establish foundations for reinforcement learning approaches to improve video model reasoning. Our work reveals that video generation is transitioning from photorealism to problem-solving, opening new frontiers where models must demonstrate understanding of logical, spatial, and strategic principles.

