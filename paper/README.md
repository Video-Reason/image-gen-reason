# Papers

This folder contains papers and research publications created from this VMEvalKit codebase project.

## Abstract Files

We provide three versions of the abstract for "Video Models Start to Solve Chess, Maze, Sudoku, Mental Rotation, and Raven's Matrices":

- **ABSTRACT.md** - Extended abstract (~600 words) for ArXiv, technical reports, and detailed documentation
- **ABSTRACT_SHORT.md** - Standard academic abstract (~250 words) for conference/journal submissions
- **ABSTRACT_WITH_FINDINGS.md** - Detailed technical summary (~1000 words) with bullet points for presentations and project pages
- **ABSTRACT_GUIDE.md** - Guide explaining when to use each version

Choose the appropriate version based on your submission requirements. See `ABSTRACT_GUIDE.md` for detailed recommendations.

## Main Paper Contributions

The paper demonstrates five key contributions:

1. **Emergent Reasoning**: First systematic evidence that video generation models exhibit measurable reasoning capabilities (40 models, >60% success rates)
2. **Validated Paradigm**: Task Pair methodology with automated evaluation validated against humans (r=0.949, κ=0.867)
3. **Extensible Framework**: Open-source VMEvalKit supporting 40 models and 5 cognitive tasks with modular architecture
4. **Robust Automation**: GPT-4O evaluation statistically equivalent to human judgment, enabling scale
5. **RL Foundation**: Objective automated evaluation opens paths for reinforcement learning improvements

## Assets

The `assets/` subfolder contains materials for public appearances and presentation purposes related to these publications:

- **Example videos**: Score=5 perfect solutions from top models (Sora, Veo 3.1)
- **12-frame decompositions**: Temporal progression visualizations (PNG + EPS formats)
- **Task examples**: Representative problems from each cognitive domain
- **Ground truth data**: Input images, solution images, and prompts

See `assets/README.md` for complete documentation of available materials.
