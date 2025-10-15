# RAVEN Progressive Matrix Reasoning Task Documentation

## Overview

The RAVEN Progressive Matrix Reasoning Task evaluates video generation models' ability to demonstrate **abstract visual reasoning** and **pattern completion** by generating videos that show the logical process of completing Raven's Progressive Matrices (RPM). This task tests analogical reasoning, relational understanding, and rule-based pattern recognition capabilities.

Note: This task now uses a lightweight local RAVEN-like generator implemented inside this repository (no external submodule). It reproduces core RPM mechanics with a simplified, robust rule set.

## Task Structure

### Core Concept
- **First Frame**: Shows 8 panels of a 3×3 Progressive Matrix with the 9th panel missing (marked with "?")  
- **Final Frame**: Shows the complete 3×3 matrix with the correct 9th panel filled in
- **Video Task**: Model must generate video showing the reasoning process to determine the missing panel
- **Text Prompt**: Provides configuration-specific instructions and reasoning hints

## Configuration Types

The current local generator supports three robust configurations:

### 1. Center Configuration
- **Description**: Single centered element with systematic transformations
- **Pattern**: Elements arranged around central focus
- **Example Prompt**: "Complete this center-focused pattern matrix"

### 2. 2×2Grid Configuration  
- **Description**: Four-element grid patterns with systematic relationships
- **Pattern**: 2×2 arrangements with rule-based variations
- **Example Prompt**: "Complete this 2×2 grid pattern matrix"

### 3. 3×3Grid Configuration
- **Description**: Nine-element grid patterns with multi-level rule applications
- **Pattern**: 3×3 arrangements with systematic transformations  
- **Example Prompt**: "Complete this 3×3 grid pattern matrix"

## Rule Types

**1. Constant Rules**: Attributes remain unchanged across panels
**2. Progression Rules**: Systematic step-by-step changes (±1 steps for Number/Position, +1 for Type/Size/Color)

**Supported Attributes**: Number, Position, Type, Size, Color  
**Rule Sampling**: Each matrix uses exactly 2 rules - one primary (Number OR Position) + one secondary (Type/Size/Color)

**Note**: *Only 2 core rule types implemented for reliability and Python 3 compatibility.*

## Data Structure

```python
task_pair = {
    "id": "raven_0001",
    "prompt": "Complete this center-focused pattern matrix. Show what goes in the missing panel.",
    "first_image_path": "/tmp/raven_0001_first.png",        # Incomplete matrix (8 panels + empty)
    "final_image_path": "/tmp/raven_0001_final.png",        # Complete matrix (all 9 panels)
    "task_category": "Center",                              # "Center", "2x2Grid", "3x3Grid"
    "configuration_type": "Center",                         # Same as task_category
    "raven_data": {
        "generation_method": "RAVEN Progressive Matrix Generator",
        "configuration": "center_single",                   # Internal config name
        "matrix_size": "160x160",
        "pattern_type": "Progressive Matrix"
    },
    "created_at": "2025-10-15T10:30:45.123456"
}
```

## Visual Representation

### First Frame (Incomplete Matrix)
```
┌─────┬─────┬─────┐
│ P₁  │ P₂  │ P₃  │  
├─────┼─────┼─────┤
│ P₄  │ P₅  │ P₆  │  
├─────┼─────┼─────┤
│ P₇  │ P₈  │     │  ← Missing panel
└─────┴─────┴─────┘
```

### Final Frame (Complete Matrix)  
```
┌─────┬─────┬─────┐
│ P₁  │ P₂  │ P₃  │  
├─────┼─────┼─────┤
│ P₄  │ P₅  │ P₆  │  
├─────┼─────┼─────┤
│ P₇  │ P₈  │ P₉  │  ← Completed panel
└─────┴─────┴─────┘
```

## Pattern Complexity & Evaluation

### Complexity Levels
- **Center**: Simplest - single position, focus on attribute changes
- **2×2Grid**: Medium - 4 positions, spatial relationships between elements  
- **3×3Grid**: Most Complex - 9 positions, complex spatial patterns

### Video Reasoning Assessment
**Core Requirements**: Video must show systematic analysis, pattern recognition, logical progression, and correct completion
**Quality Metrics**: Clarity, completeness, accuracy, coherence of reasoning demonstration

## Usage Examples

```python
from vmevalkit.tasks.raven_task import create_dataset
from vmevalkit.runner.create_dataset import main as create_main
from vmevalkit.runner.inference import main as inference_main

# Generate dataset
dataset = create_dataset(num_samples=50)
create_main(task_type="raven_task", num_samples=100)

# Run evaluation  
inference_main(model_name="your_model", dataset_path="data/questions/raven_task/raven_tasks.json")
```

## Why Video?

Unlike traditional Progressive Matrix tests requiring only the **final answer**, this task evaluates the **reasoning process**: video must show systematic analysis → pattern recognition → rule application → correct completion.

**Input**: Incomplete 3×3 matrix + text prompt  
**Output**: Video demonstrating step-by-step reasoning to complete the missing panel

| Task | Reasoning Type | Video Requirement |
|------|----------------|-------------------|  
| **RAVEN** | **Abstract Pattern Recognition** | Show rule discovery and application |
| **Maze** | **Spatial Navigation** | Show path planning and execution |
| **Chess** | **Strategic Planning** | Show move calculation and decisions |


## Technical Implementation

**Architecture**: Self-contained local implementation (`local_raven/`) with Entity→Layout→Component→Structure→Root hierarchy

**Generation Process**:
1. Choose configuration layout (center_single/distribute_four/distribute_nine)  
2. Sample 2 rules per component (primary + secondary attribute)
3. Apply rules systematically: Panel₂ = Rule(Panel₁), Panel₃ = Rule(Panel₂)
4. Render to 160×160 pixel images using OpenCV (black shapes on white background)

**Visual Output**: 480×480 pixel 3×3 grid PNG images with 2px black borders

## Design Trade-offs

**Simplified Implementation**: Uses only 2 rule types (vs. 4 in original RAVEN) and 3 configurations (vs. 7) for 100% generation reliability and Python 3 compatibility.

**Current Limitations**: Grayscale visuals, basic geometric shapes, fixed 3×3 structure - optimized for clear video reasoning evaluation over maximum complexity.

## Conclusion

RAVEN evaluates **abstract visual reasoning** in video models through Progressive Matrix completion, testing pattern recognition, analogical reasoning, and rule application. Unlike static tests, it requires **process demonstration** through video, evaluating genuine reasoning vs. superficial pattern matching.

**Core Value**: Tests higher-order reasoning capabilities beyond visual generation quality.
