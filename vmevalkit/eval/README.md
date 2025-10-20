# VMEvalKit Evaluation Module

This module provides evaluation methods for assessing video generation models' reasoning capabilities.

## Available Evaluators

### 1. Human Evaluator
Interactive web interface for human annotation of generated videos.

**Features:**
- Gradio-based web interface
- Side-by-side display of input and generated video
- Structured evaluation criteria
- Progress tracking
- Export results as JSON

**Usage:**
```bash
# Run human evaluation interface
python -m vmevalkit.runner.evaluate human --annotator "Your Name" --port 7860

# With public share link
python -m vmevalkit.runner.evaluate human --annotator "Your Name" --share
```

### 2. GPT-4O Evaluator
Automatic evaluation using OpenAI's GPT-4O vision model.

**Features:**
- Automatic frame extraction from videos
- Multi-frame analysis
- Task-specific evaluation prompts
- Batch processing of all models
- Detailed scoring and explanations

**Usage:**
```bash
# Evaluate all models
python -m vmevalkit.runner.evaluate gpt4o

# Evaluate specific models
python -m vmevalkit.runner.evaluate gpt4o --models luma-ray-2 openai-sora-2

# Custom settings
python -m vmevalkit.runner.evaluate gpt4o --max-frames 16 --temperature 0.2
```

**Requirements:**
- Set `OPENAI_API_KEY` environment variable
- Install dependencies: `opencv-python`, `httpx`

## Evaluation Criteria

Both evaluators focus on a single criterion:

**Solution Correctness**: 
- **Correct**: The solution is fully correct
- **Incorrect**: The solution is wrong  
- **Partially Correct**: The solution has some correct elements but is not fully correct

## Output Structure

Evaluations are saved in `data/evaluations/` following this structure:
```
data/evaluations/
├── pilot_experiment/
│   ├── luma-ray-2/
│   │   ├── chess_task/
│   │   │   ├── chess_0000/
│   │   │   │   ├── human-eval.json
│   │   │   │   └── gpt-4o-eval.json
│   │   │   └── ...
│   │   └── <evaluator>_summary.json
│   └── <evaluator>_all_models.json
```

## Custom Evaluators

To create a custom evaluator:

1. Inherit from `BaseEvaluator`
2. Implement `evaluate_single()` method
3. Optionally override `_calculate_summary()`

Example:
```python
from vmevalkit.eval import BaseEvaluator

class MyEvaluator(BaseEvaluator):
    def evaluate_single(self, model_name, task_type, task_id, video_path, question_data):
        # Your evaluation logic here
        return {
            "solution_correctness": "Correct",  # or "Incorrect" or "Partially Correct"
            "explanation": "The solution correctly solves the task"
        }
```

## API Reference

### BaseEvaluator
Base class providing common evaluation functionality.

**Methods:**
- `evaluate_model(model_name)`: Evaluate all tasks for a model
- `evaluate_all_models()`: Evaluate all models in experiment
- `_save_results()`: Save evaluation results
- `_calculate_summary()`: Calculate summary statistics

### HumanEvaluator
Gradio-based interface for human annotation.

**Methods:**
- `launch_interface(share, port)`: Start web interface
- `get_evaluation_summary()`: Get pandas DataFrame of results

### GPT4OEvaluator
Automatic evaluation using GPT-4O.

**Methods:**
- `extract_frames(video_path, num_frames)`: Extract video frames
- `create_evaluation_prompt(task_type)`: Generate task-specific prompts
- `evaluate_single()`: Evaluate one video

## Tips

1. **For Human Evaluation:**
   - Use consistent annotator names for tracking
   - Complete evaluations in one session when possible
   - Add detailed comments for edge cases

2. **For GPT-4O Evaluation:**
   - Adjust `max_frames` based on video length
   - Use lower temperature (0.1-0.3) for consistency
   - Monitor API costs for large experiments

3. **General:**
   - Run evaluations after all inference is complete
   - Compare human and GPT-4O results for validation
   - Export summaries for further analysis
