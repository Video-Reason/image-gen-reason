# Dataset Creation Framework for VMEvalKit

### Task Structure Pattern
All datasets follow the **First Frame → Final Frame** pattern:
- **First Frame**: Shows the initial problem state
- **Final Frame**: Shows the solved/target state  
- **Video Task**: Model must generate video showing the transition
- **Text Prompt**: Provides instructions for the reasoning task

---

## Fundamental Data Unit: Task Pair

**The basic unit of all VMEvalKit datasets is a "Task Pair"**, consisting of:

- **Initial state image**: Visual representation of the problem to solve
- **Final state image**: Visual representation of the solved/target state
- **Text prompt**: Natural language instructions for the video model
- **Metadata**: Task-specific information (difficulty, generation method, etc.)

This consistent structure enables unified evaluation across all reasoning domains.

---

## Dataset Framework Structure

### 1. Directory Organization

Every new dataset must follow this exact structure:

```
vmevalkit/tasks/{task_name}_task/
├── __init__.py                    # Module initialization
├── {task_name}_reasoning.py       # Main generation script
├── {TASK_NAME}.md                 # Documentation
└── [additional_modules.py]        # Optional helper modules

data/
├── questions/
│   ├── vmeval_dataset.json           # Master dataset manifest (all domains)
│   ├── {domain}_task/                 # Domain-specific folders
│   │   └── {question_id}/            # Per-question folders
│   │       ├── first_frame.png      # Initial state image
│   │       ├── final_frame.png      # Solution state image
│   │       ├── prompt.txt           # Text instruction
│   │       └── question_metadata.json  # Question-specific metadata
│   └── ...                           # Other domain folders
└── outputs/                          # Model-generated videos
```

### 2. Naming Conventions

- **Task folder**: `{task_name}_task` (e.g., `maze_task`, `chess_task`)
- **Main script**: `{task_name}_reasoning.py` 
- **Documentation**: `{TASK_NAME}.md` (uppercase)
- **Master dataset**: `vmeval_dataset.json` (contains all domains)
- **Domain folders**: `{domain}_task/` (e.g., `chess_task/`, `maze_task/`)
- **Question folders**: `{domain}_{id:04d}/` (e.g., `chess_0000/`, `maze_0000/`)
- **Image files**: Standardized as `first_frame.png` and `final_frame.png`
- **Prompt file**: `prompt.txt`
- **Metadata file**: `question_metadata.json`

---

## Required Dataset Format

### JSON Structure

Every dataset must follow this exact JSON structure:

```json
{
  "name": "vmeval_dataset",
  "description": "VMEvalKit video reasoning evaluation dataset (X task pairs)",
  "created_at": "ISO timestamp",
  "total_pairs": X,
  "generation_info": {
    "domains": {
      "chess": { "count": N, "description": "..." },
      "maze": { "count": N, "description": "..." },
      // Other domains...
    }
  },
  "pairs": [
    {
      "id": "{domain}_{id:04d}",
      "prompt": "Task instruction for the video model",
      "first_image_path": "{domain}_task/{question_id}/first_frame.png",
      "final_image_path": "{domain}_task/{question_id}/final_frame.png", 
      "task_category": "TaskType",
      "domain": "{domain}",
      "{domain}_data": {
        "generation_method": "description_of_method",
        // Task-specific metadata
      },
      "difficulty": "easy|medium|hard",
      // Additional task-specific fields
      "created_at": "ISO timestamp"
    }
  ]
}
```

### Required Fields

1. **Top Level**:
   - `name`: Dataset identifier
   - `description`: Human-readable description with task count
   - `pairs`: Array of task pairs

2. **Task Pair Level**:
   - `id`: Unique identifier (`{task_name}_####`)
   - `prompt`: Instructions for video model
   - `first_image_path`: Path to initial state image
   - `final_image_path`: Path to solution state image
   - `task_category`: Category name for the task type
   - `domain`: One of `chess`, `maze`, `raven`, `rotation` (used for aggregation)
   - `{task_name}_data`: Task-specific metadata object
   - `difficulty`: Standardized difficulty level
   - `created_at`: ISO timestamp

### Image Requirements

- **Format**: PNG files (required for consistency)
- **Location**: Inside per-question folders: `{domain}_task/{question_id}/`
- **Naming**: Standardized as `first_frame.png` and `final_frame.png`
- **Size**: Recommended 400x400 pixels (configurable)
- **Content**: Clear, unambiguous visual representation
- **Pairs**: Each task must have exactly one first and one final frame

---

## Implementation Template

### 1. Main Generation Script Template

Create `{task_name}_reasoning.py` following this structure:

```python
#!/usr/bin/env python3
"""
{Task Name} Reasoning Task for VMEvalKit

[Task description and purpose]

Author: VMEvalKit Team
"""

import os
import json
import random
from typing import List, Dict, Any
from datetime import datetime

class {TaskName}Generator:
    """Self-contained {task_name} task generator."""
    
    def __init__(self):
        self.generated_positions = []
        
    def generate_tasks(self, num_tasks: int = 50) -> List[Dict[str, Any]]:
        """Generate {task_name} tasks using built-in templates."""
        print(f"🎯 Generating {num_tasks} {task_name} tasks...")
        
        # Implementation: Generate your specific tasks here
        # Must return list of task dictionaries
        
        return self.generated_positions[:num_tasks]

def generate_task_images(task_data: Dict[str, Any], output_dir: str) -> tuple:
    """
    Generate first and final frame images for a task.
    
    Returns:
        (first_image_path, final_image_path)
    """
    # Implementation: Create PNG images showing initial and final states
    pass

def create_task_pair(task_data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Create a task pair in VMEvalKit format."""
    
    # Generate per-question folder structure
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    question_dir = os.path.join(base_dir, f"data/questions/{domain}_task/{task_id}")
    os.makedirs(question_dir, exist_ok=True)
    
    # Generate images with standardized names
    first_image_path = os.path.join(question_dir, "first_frame.png")
    final_image_path = os.path.join(question_dir, "final_frame.png")
    generate_task_images(task_data, question_dir)
    
    # Save prompt to text file
    with open(os.path.join(question_dir, "prompt.txt"), 'w') as f:
        f.write(prompt)
    
    # Generate prompt
    prompt = generate_prompt(task_data)
    
    # Create task pair
    # Create task pair metadata
    task_metadata = {
        "id": task_id,
        "prompt": prompt,
        "first_image_path": f"{domain}_task/{task_id}/first_frame.png",  # Relative path
        "final_image_path": f"{domain}_task/{task_id}/final_frame.png",  # Relative path
        "task_category": "TaskType",
        "domain": domain,  # Important: specify the domain
        f"{domain}_data": {
            "generation_method": "method_description",
            # Add task-specific metadata
        },
        "difficulty": task_data.get("difficulty", "easy"),
        "created_at": datetime.now().isoformat()
    }
    
    # Save metadata to question folder
    with open(os.path.join(question_dir, "question_metadata.json"), 'w') as f:
        json.dump(task_metadata, f, indent=2)
    
    return task_metadata

def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    """Create complete dataset - standard interface for all domains."""
    
    print(f"🎯 Creating {task_name} dataset with {num_samples} samples...")
    
    # Generate tasks
    generator = {TaskName}Generator()
    tasks = generator.generate_tasks(num_samples)
    
    # Create task pairs
    pairs = []
    for i, task_data in enumerate(tasks):
        task_id = f"{task_name}_{i:04d}"
        pair = create_task_pair(task_data, task_id)
        pairs.append(pair)
        print(f"✅ Created task {task_id}")
    
    # Return dataset dictionary (standard format for all domains)
    dataset = {
        "name": f"{task_name}_tasks",
        "description": f"{Task_name} reasoning tasks for video model evaluation ({len(pairs)} pairs)",
        "pairs": pairs
    }
    
    # Save master dataset (combining all domains)
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    dataset_dir = os.path.join(base_dir, "data/questions")
    os.makedirs(dataset_dir, exist_ok=True)
    output_path = os.path.join(dataset_dir, "vmeval_dataset.json")
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Saved dataset: {output_path}")
    return dataset

def main():
    """Generate dataset."""
    dataset = create_dataset(num_samples=50)
    print(f"🚀 {task_name} reasoning dataset ready!")

if __name__ == "__main__":
    main()
```

### 2. Module Initialization Template

Create `__init__.py`:

```python
"""
{Task Name} Task Module for VMEvalKit

[Brief description of task capabilities]
"""

from .{task_name}_reasoning import (
    {TaskName}Generator,
    create_dataset,
    # Add other exports
)

__all__ = [
    '{TaskName}Generator',
    'create_dataset',
    # List all exports
]

__version__ = "1.0.0"
```

### 3. Documentation Template

Create `{TASK_NAME}.md` following the structure of existing task documentation.