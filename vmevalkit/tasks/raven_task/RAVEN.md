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

### Task Pair Structure
Each RAVEN task is represented as a dictionary with the following structure:

```python
task_pair = {
    "id": str,                      # "raven_0001"
    "prompt": str,                  # Configuration-specific instruction
    "first_image_path": str,        # Incomplete matrix (8 panels + empty)
    "final_image_path": str,        # Complete matrix (all 9 panels)
    "task_category": str,           # Configuration display name ("Center", "2x2Grid", "3x3Grid")
    "configuration_type": str,      # Same as task_category
    "raven_data": Dict[str, Any],   # Generation metadata (see below)
    "created_at": str              # ISO timestamp
}
```

### RAVEN Data Structure
The `raven_data` field contains generation metadata (simplified structure):

```python
raven_data = {
    "generation_method": "RAVEN Progressive Matrix Generator",
    "configuration": str,           # Internal config name ("center_single", "distribute_four", "distribute_nine")
    "matrix_size": "160x160",       # Panel dimensions
    "pattern_type": "Progressive Matrix"
}
```

### Example Task Pair

```python
{
    "id": "raven_0001",
    "prompt": "Complete this center-focused pattern matrix. Show what goes in the missing panel.",
    "first_image_path": "/tmp/raven_0001_first.png",
    "final_image_path": "/tmp/raven_0001_final.png", 
    "task_category": "Center",
    "configuration_type": "Center",
    "raven_data": {
        "generation_method": "RAVEN Progressive Matrix Generator",
        "configuration": "center_single",
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

## Pattern Complexity

The local RAVEN generator creates patterns with varying complexity based on rule combinations:

### Pattern Types

#### Single-Rule Patterns
- **Description**: Apply the same rule across all attributes (e.g., all Constant or all Progression)
- **Complexity**: Simpler patterns with consistent transformations
- **Example**: Number increases by 1 each panel, all other attributes stay constant
- **Reasoning**: Direct pattern recognition and application

#### Multi-Rule Patterns
- **Description**: Combine different rules across attributes (e.g., Number progression + Type progression)
- **Complexity**: More complex patterns requiring multi-attribute reasoning
- **Example**: Number increases by 1 AND shapes progress through triangle→square→pentagon
- **Reasoning**: Multi-dimensional pattern analysis and rule coordination

### Configuration Complexity

Different configurations provide varying levels of spatial reasoning complexity:

- **Center**: Simplest - single position, focus on attribute changes
- **2×2Grid**: Medium - 4 positions, spatial relationships between elements
- **3×3Grid**: Most Complex - 9 positions, complex spatial patterns

**Note**: *The current implementation does not automatically classify tasks by difficulty levels. Pattern complexity emerges naturally from the combination of rules and configurations used.*

## Evaluation Criteria

### Core Reasoning Assessment
1. **Pattern Recognition**: Can the model identify the underlying rule?
2. **Logical Completion**: Is the 9th panel correctly determined?
3. **Process Demonstration**: Does the video show logical reasoning steps?
4. **Rule Consistency**: Are the identified rules applied correctly?

### Video Quality Metrics
1. **Clarity**: Is the reasoning process visually clear?
2. **Completeness**: Does the video show the full reasoning sequence?
3. **Accuracy**: Is the final panel completion correct?
4. **Coherence**: Is the reasoning logically consistent throughout?

## Usage Examples

### Basic Generation
```python
from vmevalkit.tasks.raven_task import RavenGenerator, create_dataset

# Generate 50 RAVEN tasks
dataset = create_dataset(num_samples=50)
print(f"Generated {len(dataset['pairs'])} Progressive Matrix tasks")

# Access individual tasks
task = dataset['pairs'][0]
print(f"Task: {task['prompt']}")
print(f"Category: {task['task_category']}")
```

### Custom Configuration
```python  
generator = RavenGenerator()

# Generate specific configuration
task_data = generator.generate_single_task("center_single")
print(f"Generated {task_data['config_display']} task")
```

## Video Reasoning Evaluation

### What Makes This a Video Task?

Unlike traditional Progressive Matrix tests that only require the **final answer**, VMEvalKit's RAVEN task evaluates the **reasoning process** through video generation:

#### Input Requirements
- **Image**: Incomplete 3×3 matrix (8 panels filled, 1 missing)
- **Text**: Configuration-specific prompt ("Complete this center-focused pattern matrix...")

#### Expected Video Output
The model should generate a video demonstrating:
1. **Analysis Phase**: Visual examination of existing panels  
2. **Pattern Recognition**: Identification of transformation rules
3. **Rule Application**: Step-by-step application of discovered patterns
4. **Completion**: Final panel generation showing the solution

#### Video Reasoning Assessment

**Process Evaluation Criteria:**
- **Systematic Analysis**: Does the model examine patterns systematically?  
- **Rule Identification**: Can it identify the underlying transformation rules?
- **Logical Progression**: Are reasoning steps logically connected?
- **Visual Clarity**: Is the reasoning process clearly demonstrated?
- **Correct Completion**: Is the final panel logically correct?

### Comparison with Other VMEvalKit Tasks

| Task | Reasoning Type | Video Requirement |
|------|----------------|-------------------|  
| **RAVEN** | **Abstract Pattern Recognition** | Show rule discovery and application process |
| **Maze** | **Spatial Navigation** | Show path planning and execution |
| **Chess** | **Strategic Planning** | Show move calculation and decision process |

**RAVEN uniquely tests**: Abstract reasoning, analogical thinking, pattern completion

## Integration with VMEvalKit

### Dataset Creation
```python
# Generate RAVEN dataset
from vmevalkit.runner.create_dataset import main
main(task_type="raven_task", num_samples=100)
```

### Model Evaluation
```python
# Run inference on RAVEN tasks  
from vmevalkit.runner.inference import main
main(
    model_name="your_model",
    dataset_path="data/questions/raven_task/raven_tasks.json"
)
```

### Expected Model Capabilities
For successful RAVEN task completion, video models must demonstrate:
- **Visual Pattern Analysis**: Examine multiple panels systematically
- **Abstract Rule Inference**: Discover transformation patterns from examples  
- **Analogical Reasoning**: Apply discovered rules to new situations
- **Process Visualization**: Show reasoning steps through coherent video sequences

 

## Technical Implementation

### Local RAVEN Core Architecture

The task uses a **self-contained local implementation** (`local_raven/`) with the following components:

#### Abstract Object Tree (AOT) Structure
```python
# Entity: Individual visual elements
class Entity:
    bbox: (y, x, h, w)          # Relative positioning coordinates  
    type: Attribute             # Shape (triangle/square/pentagon/hexagon/circle)
    size: Attribute             # Size level (5 discrete levels)
    color: Attribute            # Color level (10 discrete levels)

# Layout: Manages entity positioning and count
class Layout:
    positions: List[bbox]       # Available position slots
    entities: List[Entity]      # Entities placed in positions
    
# Component → Structure → Root: Hierarchical containers
```

#### Matrix Generation Process

1. **Configuration Setup**: Choose layout (center_single, distribute_four, distribute_nine)
2. **Rule Sampling**: Generate 2 rules per component (main attribute + secondary attribute)
3. **Row Generation**: Apply rules systematically to create 3 matrix rows
4. **Panel Rendering**: Convert AOT nodes to 160×160 pixel images

```python
# Core generation algorithm (simplified)
def generate_panels(root, rule_groups):
    start_node = root.sample()
    
    def build_row(base_node):
        # Apply rule transformations: c2 = rule(c1), c3 = rule(c2)
        for rule_group in rule_groups:
            c2 = rule_group[0].apply_rule(base_node)
            c3 = rule_group[0].apply_rule(c2)
            # Apply additional rules in group
    
    row1 = build_row(start_node)
    row2 = build_row(start_node.resample())  # New base for row 2
    row3 = build_row(start_node.resample())  # New base for row 3
    
    return [render_panel(node) for node in all_nodes]
```

### Visual Rendering

#### Geometric Shapes
- **Shapes**: Triangle, Square, Pentagon, Hexagon, Circle
- **Rendering**: OpenCV-based line drawing with 2px thickness
- **Colors**: Grayscale (black shapes on white background)
- **Size Levels**: `[0.45, 0.55, 0.65, 0.75, 0.85]` relative to panel size

#### Image Processing  
- **Panel Size**: 160×160 pixels per panel
- **Matrix Size**: 480×480 pixels (3×3 grid)
- **Format**: PNG images
- **Grid Lines**: 2px black borders between panels
- **Incomplete Matrix**: 9th panel filled with white pixels
- **Complete Matrix**: All 9 panels rendered with geometric patterns

### Configuration-Specific Layouts

```python
# Center: Single centered position
build_center_single() → [(0.5, 0.5, 1.0, 1.0)]

# 2x2Grid: Four corner positions  
build_distribute_four() → [(0.25, 0.25, 0.45, 0.45), (0.25, 0.75, 0.45, 0.45), ...]

# 3x3Grid: Nine grid positions
build_distribute_nine() → [(0.17, 0.17, 0.3, 0.3), (0.17, 0.5, 0.3, 0.3), ...]
```

### Rule Processing

Rules are applied in **sequence within each row**, then **rows are built independently**:

- **Within Row**: Panel₂ = Rule(Panel₁), Panel₃ = Rule(Panel₂)  
- **Across Rows**: Each row starts from a **resampled base** to create row-to-row variation
- **Rule Consistency**: Same rules applied across all rows for pattern consistency

## Research Applications

### Abstract Reasoning Research
- **Analogical Thinking**: A:B :: C:? relationships in visual domain
- **Rule Learning**: Can models learn abstract transformation rules?
- **Transfer Learning**: Do learned patterns transfer across configurations?
- **Systematic Generalization**: Performance across unseen rule combinations

### Video Reasoning Evaluation
- **Process Visualization**: How do models show their reasoning?
- **Step-by-Step Logic**: Can models break down complex reasoning?
- **Visual Communication**: How clearly can models demonstrate abstract thinking?
- **Temporal Consistency**: Do reasoning steps follow logically in sequence?

## Implementation Design Decisions

### Reliability Improvements

The local RAVEN implementation prioritizes **reliability and consistency** over maximum complexity:

#### Simplified Rule Set
- **Decision**: Implement only 2 core rule types (Constant, Progression) 
- **Rationale**: Ensures consistent generation success across all configurations
- **Trade-off**: Reduced pattern complexity vs. guaranteed task generation
- **Benefit**: 100% generation success rate vs. frequent failures in full RAVEN

#### Python 3 Compatibility  
- **Decision**: Rebuild core logic for Python 3 compatibility
- **Rationale**: Original RAVEN was designed for Python 2.7
- **Implementation**: Clean, modern Python with type hints and error handling
- **Benefit**: Integration with modern ML/AI ecosystems

#### Fixed Configuration Set
- **Decision**: Support 3 robust configurations vs. 7 original configurations  
- **Rationale**: Focus on configurations with highest success rates
- **Configurations**: `center_single`, `distribute_four`, `distribute_nine`
- **Benefit**: Predictable, reliable pattern generation

### Current Limitations

#### Pattern Complexity
- **Rule Types**: Limited to Constant and Progression (no Arithmetic or Distribute_Three)
- **Combinations**: Simpler rule combinations vs. full RAVEN complexity
- **Impact**: Patterns are more predictable but still cognitively challenging

#### Visual Attributes  
- **Colors**: Grayscale only (black shapes on white background)
- **Shapes**: 5 basic geometric primitives
- **Positions**: Grid-based positioning only
- **Impact**: Clear, high-contrast patterns optimized for video reasoning

#### Matrix Structure
- **Size**: Fixed 3×3 structure  
- **Panels**: Always 160×160 pixels
- **Layout**: Grid-based arrangement only
- **Impact**: Consistent format but limited spatial reasoning variations

### Evaluation Considerations

#### Pattern Recognition vs. True Reasoning
- **Challenge**: Distinguishing memorization from genuine pattern understanding
- **Approach**: Focus on **process demonstration** through video generation
- **Metric**: Evaluate reasoning steps, not just final panel correctness

#### Video Quality Assessment
- **Challenge**: How to evaluate quality of reasoning demonstration in video
- **Approach**: Multi-criteria evaluation (clarity, completeness, accuracy, coherence)
- **Future**: Automated video reasoning assessment tools needed

## Future Extensions

### Enhanced Complexity
- **Temporal Matrices**: Progressive matrices that change over time
- **3D Reasoning**: Spatial reasoning with 3D transformations
- **Multi-Modal**: Combining visual patterns with textual rules
- **Interactive**: Progressive matrices that respond to user input

### Advanced Evaluation
- **Process Scoring**: Evaluate reasoning steps, not just final answers  
- **Creativity Metrics**: Assess novel approaches to pattern completion
- **Robustness Testing**: Performance under visual noise or partial occlusion
- **Explanability**: Natural language explanation of reasoning process

## Conclusion

The RAVEN Progressive Matrix task brings **abstract visual reasoning** evaluation to video models, complementing VMEvalKit's spatial (maze) and strategic (chess) reasoning tasks. It tests fundamental cognitive capabilities:

- **Pattern Recognition**: Identifying systematic transformations
- **Analogical Reasoning**: Understanding A:B :: C:? relationships  
- **Rule Application**: Applying abstract logical rules to visual data
- **Process Demonstration**: Showing reasoning steps through video

This creates a comprehensive evaluation framework for **higher-order reasoning capabilities** in video generation models, moving beyond basic visual quality to test genuine **abstract intelligence**.

The task is particularly valuable for:
- 🧠 **Cognitive AI Research**: Testing abstract reasoning capabilities
- 📊 **Model Benchmarking**: Standardized progressive matrix evaluation  
- 🔬 **Reasoning Analysis**: Understanding how models approach pattern completion
- 🎯 **Intelligence Assessment**: Measuring genuine vs. superficial pattern matching

**RAVEN tasks push video models beyond visual generation to demonstrate true reasoning intelligence.**
