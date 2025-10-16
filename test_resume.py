#!/usr/bin/env python3
"""
Test script to verify the resume mechanism works correctly.
"""

import sys
import json
import importlib.util
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import module with hyphen in name
spec = importlib.util.spec_from_file_location(
    "experiment", 
    Path(__file__).parent / "examples" / "experiment_2025-10-14.py"
)
experiment_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_module)
ProgressTracker = experiment_module.ProgressTracker

def test_progress_tracker():
    """Test the ProgressTracker class."""
    print("Testing ProgressTracker...")
    
    # Create a test output directory
    test_dir = Path("./test_resume_output")
    test_dir.mkdir(exist_ok=True)
    
    # Test 1: Create new tracker
    print("\n1. Creating new tracker...")
    tracker = ProgressTracker(test_dir, "test_exp_001")
    assert tracker.experiment_id == "test_exp_001"
    assert not tracker.is_resume
    print("   ✅ New tracker created")
    
    # Test 2: Job lifecycle
    print("\n2. Testing job lifecycle...")
    
    # Start a job
    job_id = "model1_task1"
    should_proceed = tracker.job_started(job_id)
    assert should_proceed == True
    assert job_id in tracker.progress["jobs_in_progress"]
    print("   ✅ Job started")
    
    # Complete the job
    tracker.job_completed(job_id, {"status": "success"})
    assert job_id in tracker.progress["jobs_completed"]
    assert job_id not in tracker.progress["jobs_in_progress"]
    print("   ✅ Job completed")
    
    # Try to start same job again
    should_proceed = tracker.job_started(job_id)
    assert should_proceed == False
    print("   ✅ Duplicate job prevented")
    
    # Test 3: Failed job
    print("\n3. Testing failed job...")
    failed_job_id = "model1_task2"
    tracker.job_started(failed_job_id)
    tracker.job_failed(failed_job_id, "Test error")
    assert failed_job_id not in tracker.progress["jobs_in_progress"]
    assert any(j.get("job_id") == failed_job_id for j in tracker.progress["jobs_failed"] if isinstance(j, dict))
    print("   ✅ Failed job tracked")
    
    # Test 4: Save progress
    print("\n4. Testing save/load...")
    tracker.save_progress(force=True)
    assert tracker.checkpoint_file.exists()
    print("   ✅ Progress saved")
    
    # Test 5: Resume from checkpoint
    print("\n5. Testing resume...")
    tracker2 = ProgressTracker(test_dir, "test_exp_001")
    assert tracker2.is_resume
    assert job_id in tracker2.progress["jobs_completed"]
    assert tracker2.should_skip_job(job_id)
    print("   ✅ Resume successful")
    
    # Test 6: Statistics update
    print("\n6. Testing statistics...")
    stats = {"completed": 1, "failed": 1, "total": 2}
    tracker2.update_statistics(stats)
    assert tracker2.progress["statistics"] == stats
    print("   ✅ Statistics updated")
    
    # Cleanup
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
    print("\n✅ All tests passed!")
    
    print("\n" + "="*50)
    print("Resume mechanism is working correctly!")
    print("="*50)

if __name__ == "__main__":
    test_progress_tracker()
