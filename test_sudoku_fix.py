#!/usr/bin/env python3
"""
Test script to verify sudoku task no longer generates sudoku_dataset.json
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vmevalkit.tasks.sudoku_task.sudoku_reasoning import create_dataset

def test_sudoku_no_json():
    """Test that sudoku task returns dict but doesn't save JSON file."""
    
    # Generate a small test dataset
    print("Testing sudoku dataset generation...")
    dataset = create_dataset(num_samples=3)
    
    # Check it returns a dictionary
    assert isinstance(dataset, dict), "create_dataset should return a dict"
    assert "pairs" in dataset, "Dataset should have 'pairs' field"
    assert len(dataset["pairs"]) == 3, f"Expected 3 pairs, got {len(dataset['pairs'])}"
    
    print(f"✅ Generated {len(dataset['pairs'])} sudoku tasks")
    print(f"✅ Dataset name: {dataset['name']}")
    print(f"✅ Dataset structure matches other tasks")
    
    # Check that no JSON file was created in the default location
    json_path = "data/questions/sudoku_task/sudoku_dataset.json"
    if os.path.exists(json_path):
        print(f"❌ ERROR: Found unexpected {json_path} - this file should not be created!")
        return False
    else:
        print(f"✅ Correctly did NOT create {json_path}")
    
    # Show the first task to confirm it's working
    first_task = dataset["pairs"][0]
    print(f"\nFirst task example:")
    print(f"  ID: {first_task['id']}")
    print(f"  Difficulty: {first_task['difficulty']}")
    print(f"  Given numbers: {first_task['num_given']}/9")
    
    print("\n✅ SUCCESS: Sudoku task now follows the same pattern as other tasks!")
    return True

if __name__ == "__main__":
    success = test_sudoku_no_json()
    if not success:
        sys.exit(1)
