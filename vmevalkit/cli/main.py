import argparse
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any
from vmevalkit.runner.retriever import Retriever
from vmevalkit.runner.inference import InferenceRunner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', help='Train config file path')
    return parser.parse_args()

def discover_tasks_from_folders(questions_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Discover all tasks by scanning folder structure.
    
    Args:
        questions_dir: Path to questions directory
        
    Returns:
        Dictionary mapping domain to list of task dictionaries
    """
    print(f"🔍 Discovering tasks from folders: {questions_dir}")
    
    tasks_by_domain = {}
    total_tasks = 0
    
    # Scan each domain folder
    for domain_dir in sorted(questions_dir.glob("*_task")):
        if not domain_dir.is_dir():
            continue
            
        domain = domain_dir.name.replace("_task", "")
        domain_tasks = []
        
        print(f"   📁 Scanning {domain_dir.name}/")
        
        # Scan each question folder in this domain  
        for question_dir in sorted(domain_dir.glob(f"{domain}_*")):
            if not question_dir.is_dir():
                continue
                
            task_id = question_dir.name
            
            # Look for required files
            prompt_file = question_dir / "prompt.txt"
            first_image = question_dir / "first_frame.png"
            final_image = question_dir / "final_frame.png"
            
            # Required files check
            if not prompt_file.exists():
                print(f"      ⚠️  Skipping {task_id}: Missing prompt.txt")
                continue
                
            if not first_image.exists():
                print(f"      ⚠️  Skipping {task_id}: Missing first_frame.png")
                continue
            
            # Load prompt
            prompt_text = prompt_file.read_text().strip()
            
            # Create task dictionary
            task = {
                "id": task_id,
                "domain": domain,
                "task_category": domain,
                "prompt": prompt_text,
                "first_image_path": str(first_image.absolute()),
                "final_image_path": str(final_image.absolute()) if final_image.exists() else None
            }
            
            # Load supplemental metadata from JSON if exists
            metadata_file = question_dir / "question_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    supplemental_metadata = json.load(f)
                
                # Add supplemental data but don't override core fields
                for key, value in supplemental_metadata.items():
                    if key not in task:
                        task[key] = value
            
            domain_tasks.append(task)
            
        print(f"      ✅ Found {len(domain_tasks)} tasks in {domain}")
        tasks_by_domain[domain] = domain_tasks
        total_tasks += len(domain_tasks)
    
    print(f"\n📊 Discovery Summary:")
    print(f"   Total tasks: {total_tasks}")
    for domain, tasks in tasks_by_domain.items():
        print(f"   {domain.title()}: {len(tasks)} tasks")
    
    return tasks_by_domain

def main():
    args = parse_args()
    
    if not args.config:
        print("Error: Config file path is required")
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Step 1: Retrieve tasks
    dataset_config_list = config_dict.get('datasets', [])
    questions_dir = None
    
    for dataset_config in dataset_config_list:
        retriever = Retriever(
            dataset_config=dataset_config,
        )
        retriever.retrieve_tasks()
        # Get output path from retriever
        if questions_dir is None:
            questions_dir = Path(retriever.output_path)
    
    if questions_dir is None:
        questions_dir = Path("data/questions")
    
    # Step 2: Discover all tasks from folders
    tasks_by_domain = discover_tasks_from_folders(questions_dir)
    
    # Step 3: Get models from config
    models_config = config_dict.get('models', [])
    if not models_config:
        print("⚠️  No models specified in config. Skipping inference.")
        return
    
    model_names = [model.get('model') for model in models_config if model.get('model')]
    if not model_names:
        print("⚠️  No valid model names found in config. Skipping inference.")
        return
    
    print(f"\n🤖 Models to run: {model_names}")
    
    # Step 4: Setup output directory
    output_dir = Path(config_dict.get('output_dir', 'data/outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 5: Run inference for each model and task
    runner = InferenceRunner(output_dir=str(output_dir))
    
    total_inferences = 0
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"🎨 Running model: {model_name}")
        print(f"{'='*60}")
        
        for domain, tasks in tasks_by_domain.items():
            print(f"\n📂 Processing domain: {domain} ({len(tasks)} tasks)")
            
            for task in tasks:
                task_id = task["id"]
                image_path = task["first_image_path"]
                prompt = task["prompt"]
                
                print(f"\n  🎨 Generating: {task_id} with {model_name}")
                print(f"     Image: {image_path}")
                print(f"     Prompt: {prompt[:80]}...")
                
                # Check if image exists
                if not Path(image_path).exists():
                    print(f"     ⚠️  Skipping: Image not found")
                    continue
                
                # Run inference
                result = runner.run(
                    model_name=model_name,
                    image_path=image_path,
                    text_prompt=prompt,
                    question_data=task,
                )
                
                if result.get("status") == "success":
                    total_inferences += 1
                    print(f"     ✅ Success: {result.get('inference_dir')}")
                else:
                    print(f"     ❌ Failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n{'='*60}")
    print(f"✅ Inference complete!")
    print(f"   Total successful inferences: {total_inferences}")
    print(f"   Output directory: {output_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
