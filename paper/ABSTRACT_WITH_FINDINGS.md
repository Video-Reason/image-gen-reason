# Abstract with Key Findings

## Video Models Start to Solve Chess, Maze, Sudoku, Mental Rotation, and Raven's Matrices

### Core Contribution

We introduce VMEvalKit, a systematic framework for evaluating reasoning capabilities in video generation models through five cognitive tasks: chess puzzles, maze navigation, Sudoku solving, 3D mental rotation, and Raven's Progressive Matrices. Unlike traditional video generation benchmarks that assess visual quality, we test whether models can generate videos demonstrating correct solutions to visual reasoning problems—requiring models to transition from initial problem states to accurate solution states.

### Main Contributions & Findings

**1. Emergent Reasoning in Video Models**
- **40 models evaluated** across 11 families (Sora, Veo, Luma, Runway, LTX-Video, HunyuanVideo, etc.)
- **Leading models achieve >60% success rates** on reasoning tasks requiring strategic planning, spatial pathfinding, logical deduction, 3D spatial reasoning, and abstract pattern recognition
- **Top performers (Sora, Veo 3.1)** demonstrate near-human performance on complex chess tactics (mate-in-1) and abstract pattern completion
- **Performance hierarchy emerges**: Commercial frontier models > Mid-tier APIs > Open-source models, suggesting reasoning scales with model capacity
- **Task difficulty varies**: Sudoku (easiest) → Raven → Rotation → Maze → Chess (hardest), revealing distinct cognitive capabilities

**2. Validated Evaluation Paradigm**
- **"Task Pair" methodology**: (initial_state.png, final_state.png, instruction_prompt.txt) enables objective evaluation
- **Automated GPT-4O evaluation validated against human judgment** across 450 paired evaluations:
  - Pearson correlation: **r = 0.949** (p < 0.001)
  - Cohen's kappa: **κ = 0.867** (substantial agreement)
  - Paired t-test: **p = 0.051** (no significant difference)
  - Wilcoxon test: **p = 0.067** (non-parametric confirmation)
  - **Convergence at n ≥ 25 samples** (methods become statistically equivalent)
- **Task-specific validation**:
  - Chess: r = 0.985 (near-perfect agreement)
  - Sudoku: r = 0.997 (near-perfect agreement)
  - Maze: r = 0.912 (strong agreement)
  - Rotation: r = 0.933 (strong agreement)
  - Raven: r = 0.891 (strong agreement)
- **Evaluation cost reduction**: Automated eval enables 100× faster evaluation than human annotation while maintaining validity

**3. Extensible Open-Source Framework**
- **Modular architecture** with dynamic model loading (no core modifications needed for new models)
- **Standardized Task Pair interface** enables adding new cognitive domains via simple registry entry
- **11 model families supported**: Commercial APIs (Luma, Veo, Runway, Sora, WaveSpeed) + Open-source (LTX-Video, HunyuanVideo, VideoCrafter, DynamiCrafter)
- **Complete inference pipeline**: Dataset generation → Video inference → Automated evaluation → Statistical analysis
- **Resume capability**: Automatic skip of completed evaluations for interrupted experiments
- **Web dashboard**: Real-time visualization of model performance across tasks
- **Public dataset**: 450+ task pairs across 5 cognitive domains with ground truth solutions

**4. Robust Automated Evaluation at Scale**
- **Statistical equivalence** between automated and human evaluation enables scaling
- **Task-specific prompts** ensure domain-appropriate evaluation criteria
- **1-5 scoring scale** with clear rubric (1=completely wrong, 5=perfect solution)
- **Final frame comparison** provides objective ground truth matching
- **Bootstrap confidence intervals** confirm evaluation stability (95% CI: [-0.100, -0.002])
- **Low evaluation variance** across repeated trials indicates reliability
- **Production-ready**: Successfully evaluated 450+ videos with consistent results

**5. Foundation for Reinforcement Learning**
- **Clear reward signal**: Solution correctness scores (1-5) enable outcome-based RL
- **Ground truth targets**: Task pairs provide supervised fine-tuning data
- **Automated feedback loop**: Models can be iteratively improved without human annotation
- **Multi-task learning**: 5 diverse reasoning domains support transfer learning research
- **Difficulty scaling**: Easy/medium/hard variants enable curriculum learning
- **Standardized benchmark**: Enables measuring improvements over time

### Key Insights

1. **Video models possess latent reasoning capabilities** that emerge under appropriate evaluation paradigms
2. **Reasoning quality scales with model sophistication**: Frontier models significantly outperform smaller alternatives
3. **Task-specific performance patterns** reveal models excel at different cognitive capabilities (e.g., logical deduction easier than strategic planning)
4. **Automated evaluation validity** opens the door to large-scale iterative improvement cycles
5. **Open framework enables collaborative progress**: Researchers can contribute new models, tasks, and evaluation methods

### Implications

Our work establishes that video generation is transitioning from **photorealism to problem-solving**. By demonstrating measurable reasoning capabilities and providing validated infrastructure for assessment, we open new research directions:

- **Reinforcement learning for video reasoning**: Using automated rewards to improve problem-solving
- **Multi-modal reasoning benchmarks**: Extending to more complex cognitive tasks
- **Model interpretability**: Understanding how video models represent and manipulate reasoning states
- **Transfer learning**: Leveraging reasoning capabilities across different visual domains
- **Human-AI collaboration**: Developing video models as reasoning assistants

VMEvalKit provides both empirical evidence of emergent video reasoning and the infrastructure to systematically measure, compare, and enhance these capabilities. We release the complete framework, dataset, and evaluation tools to enable the research community to advance video models from passive generators to active visual problem-solvers.

---

**Resources:**
- **Code**: https://github.com/yourusername/VMEvalKit
- **Dataset**: 450+ task pairs with ground truth (5 cognitive domains)
- **Models**: Configurations for 40 video generation models
- **Evaluation**: Human and automated (GPT-4O) evaluation pipelines
- **Analysis**: Statistical validation scripts and visualization tools
- **Documentation**: Complete guides for adding models and tasks

