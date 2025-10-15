#!/usr/bin/env python3
"""
Generate Center RAVEN Matrices

Creates several Center-type RAVEN progressive matrices showing:
- Single centered entity
- Focus on attribute changes (type, size, color)
- Simple reasoning patterns

Outputs both incomplete (first frame) and complete (final frame) matrices
to the output/ folder for easy visualization.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up virtual environment if needed
def setup_environment():
    """Activate venv if exists"""
    venv_activate = project_root / "venv" / "bin" / "activate"
    if venv_activate.exists():
        print("✅ Using virtual environment")
    else:
        print("⚠️  Virtual environment not found at venv/bin/activate")

def generate_center_matrices(num_matrices=10):
    """Generate several Center RAVEN matrices"""
    
    print("🧩 Generating Center RAVEN Matrices")
    print("="*50)
    
    setup_environment()
    
    # Import RAVEN components
    try:
        from vmevalkit.tasks.raven_task.raven_reasoning import RavenGenerator
        print("✅ RAVEN modules loaded successfully")
    except Exception as e:
        print(f"❌ Error loading RAVEN modules: {e}")
        print("Make sure you're in the VMEvalKit directory and virtual environment is active")
        return
    
    # Create output directory
    output_dir = project_root / "output" / "center_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = RavenGenerator()
    
    print(f"🎯 Generating {num_matrices} Center matrices...")
    print()
    
    successful_generations = 0
    
    for i in range(num_matrices):
        try:
            # Generate Center task specifically
            task_data = generator.generate_single_task("center_single")
            
            # Generate images
            task_id = f"center_{i:03d}"
            
            # Import image generation functions
            from vmevalkit.tasks.raven_task.raven_reasoning import generate_rpm_image
            
            # Create image paths
            first_path = output_dir / f"{task_id}_incomplete.png"
            final_path = output_dir / f"{task_id}_complete.png"
            
            # Generate incomplete matrix (first 8 panels + empty)
            generate_rpm_image(task_data["matrix"], str(first_path), incomplete=True)
            
            # Generate complete matrix (all 9 panels)
            generate_rpm_image(task_data["matrix"], str(final_path), incomplete=False)
            
            # Generate prompt
            from vmevalkit.tasks.raven_task.raven_reasoning import generate_prompt
            prompt = generate_prompt(task_data)
            
            # Save prompt as text file
            prompt_path = output_dir / f"{task_id}_prompt.txt"
            prompt_path.write_text(prompt)
            
            successful_generations += 1
            print(f"✅ Generated {task_id}: {task_data['config_display']}")
            print(f"   🖼️  Incomplete: {first_path.name}")
            print(f"   🖼️  Complete: {final_path.name}")
            print(f"   📝 Prompt: {prompt}")
            print()
            
        except Exception as e:
            print(f"❌ Failed to generate matrix {i}: {e}")
            continue
    
    print("="*50)
    print(f"🎉 Generated {successful_generations}/{num_matrices} Center matrices")
    print(f"📁 Saved to: {output_dir}")
    print()
    print("Matrix files:")
    print("  - *_incomplete.png: Shows 8 panels + empty (what the AI model sees)")
    print("  - *_complete.png: Shows all 9 panels (the correct solution)")
    print("  - *_prompt.txt: Task instruction text")
    print()
    print("Center matrices focus on:")
    print("  🎯 Single centered entity")
    print("  🔄 Attribute progression (type, size, color)")
    print("  📐 Simplest reasoning pattern")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Center RAVEN matrices")
    parser.add_argument("--num", type=int, default=10, help="Number of matrices to generate (default: 10)")
    
    args = parser.parse_args()
    
    generate_center_matrices(args.num)

if __name__ == "__main__":
    main()
