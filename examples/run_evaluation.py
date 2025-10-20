#!/usr/bin/env python3
"""Example script demonstrating how to use VMEvalKit evaluation system."""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vmevalkit.eval import HumanEvaluator, GPT4OEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_human_evaluation():
    """Example of running human evaluation."""
    print("\n=== Human Evaluation Example ===")
    
    # Create evaluator
    evaluator = HumanEvaluator(
        experiment_name="pilot_experiment",
        annotator_name="John Doe"
    )
    
    # Launch interface (this will block until closed)
    print("Launching human evaluation interface...")
    print("Open http://localhost:7860 in your browser")
    evaluator.launch_interface(port=7860)


def example_gpt4o_evaluation():
    """Example of running GPT-4O evaluation."""
    print("\n=== GPT-4O Evaluation Example ===")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY environment variable")
        return
    
    # Create evaluator
    evaluator = GPT4OEvaluator(
        experiment_name="pilot_experiment",
        max_frames=8,
        temperature=0.1
    )
    
    # Example 1: Evaluate a specific model
    print("\nEvaluating specific model: luma-ray-2")
    try:
        results = evaluator.evaluate_model("luma-ray-2")
        print(f"Evaluation complete. Summary: {results['summary']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Evaluate all models
    print("\nEvaluating all models...")
    all_results = evaluator.evaluate_all_models()
    
    # Print summary for each model
    for model_name, results in all_results.items():
        if "summary" in results:
            print(f"\n{model_name}:")
            print(f"  - Total tasks: {results['summary']['total_tasks']}")
            print(f"  - Average score: {results['summary'].get('average_score', 0):.2f}")


def example_custom_evaluation():
    """Example of using the evaluation API programmatically."""
    print("\n=== Custom Evaluation Example ===")
    
    from vmevalkit.eval import BaseEvaluator
    
    class SimpleEvaluator(BaseEvaluator):
        """A simple custom evaluator for demonstration."""
        
        def evaluate_single(self, model_name, task_type, task_id, video_path, question_data):
            # Simple evaluation logic
            import random
            
            score = random.uniform(3, 5)  # Random score for demo
            
            return {
                "scores": {
                    "overall": score
                },
                "binary_evaluations": {
                    "passes": score >= 4
                },
                "overall_score": score,
                "status": "completed"
            }
    
    # Use the custom evaluator
    evaluator = SimpleEvaluator(experiment_name="pilot_experiment")
    
    # Evaluate a model
    print("Running custom evaluation...")
    results = evaluator.evaluate_model("luma-ray-2")
    print(f"Custom evaluation complete: {results['summary']}")


def main():
    """Main function to run examples."""
    print("VMEvalKit Evaluation Examples")
    print("=" * 40)
    
    # Check if pilot_experiment exists
    if not Path("data/outputs/pilot_experiment").exists():
        print("Error: pilot_experiment not found. Please run inference first.")
        return
    
    # Run examples based on command line argument
    if len(sys.argv) > 1:
        if sys.argv[1] == "human":
            example_human_evaluation()
        elif sys.argv[1] == "gpt4o":
            example_gpt4o_evaluation()
        elif sys.argv[1] == "custom":
            example_custom_evaluation()
        else:
            print(f"Unknown example: {sys.argv[1]}")
            print("Usage: python run_evaluation.py [human|gpt4o|custom]")
    else:
        print("Usage: python run_evaluation.py [human|gpt4o|custom]")
        print("\nAvailable examples:")
        print("  human  - Launch human evaluation interface")
        print("  gpt4o  - Run GPT-4O automatic evaluation")
        print("  custom - Demonstrate custom evaluator")


if __name__ == "__main__":
    main()
