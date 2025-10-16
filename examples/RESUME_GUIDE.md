# Resume Mechanism Guide

This experiment script now includes a robust resume mechanism that allows you to:
- Pause/interrupt experiments and resume them later
- Automatically save progress checkpoints
- Track completed, failed, and in-progress jobs
- Resume from specific experiments or the latest one

## Features

### 1. Automatic Checkpointing
- Progress is automatically saved every 5 completed jobs
- Checkpoint files are stored in `./data/outputs/pilot_experiment/logs/`
- Each experiment gets a unique ID for tracking

### 2. Graceful Interruption
- Press `Ctrl+C` to safely interrupt an experiment
- Progress is automatically saved before exit
- You can resume from where you left off

### 3. Resume Options

#### Resume Latest Experiment
```bash
python examples/experiment_2025-10-14.py --resume latest
```

#### Resume Specific Experiment
```bash
python examples/experiment_2025-10-14.py --resume experiment_20241016_143022
```

#### List Available Checkpoints
```bash
python examples/experiment_2025-10-14.py --list-checkpoints
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--resume <ID or 'latest'>` | Resume a previous experiment |
| `--no-resume` | Disable resume mechanism entirely |
| `--experiment-id <ID>` | Set custom experiment ID |
| `--workers <N>` | Number of parallel workers (default: 6) |
| `--all-tasks` | Run all tasks instead of 1 per domain |
| `--list-checkpoints` | List available checkpoints and exit |

## How It Works

### Progress Tracking
The system tracks three types of jobs:
- **Completed**: Successfully finished jobs (won't be re-run)
- **Failed**: Jobs that failed (can be retried on resume)
- **In-Progress**: Jobs interrupted mid-execution (will be retried)

### Files Created
For each experiment, the following files are created:
- `checkpoint_<experiment_id>.json` - Main checkpoint file for resume
- `progress_<experiment_id>.json` - Detailed progress log
- `logs_final.json` - Final results
- `statistics.json` - Experiment statistics

### Safety Features
- **Atomic Writes**: Checkpoints use temp files to prevent corruption
- **Backup Creation**: Corrupted checkpoints are automatically backed up
- **Thread-Safe**: All progress updates are thread-safe for parallel execution
- **Signal Handling**: Properly handles SIGINT and SIGTERM signals

## Examples

### Start New Experiment with Custom ID
```bash
python examples/experiment_2025-10-14.py --experiment-id my_test_001
```

### Resume After Interruption
```bash
# Start experiment
python examples/experiment_2025-10-14.py

# Press Ctrl+C to interrupt...
# Later, resume from where you left off:
python examples/experiment_2025-10-14.py --resume latest
```

### Run Full Dataset with More Workers
```bash
python examples/experiment_2025-10-14.py --all-tasks --workers 10 --experiment-id full_run_001
```

### Resume Specific Experiment with Different Worker Count
```bash
python examples/experiment_2025-10-14.py --resume full_run_001 --workers 4
```

## Tips

1. **Experiment IDs**: Use descriptive IDs for easier tracking
2. **Worker Count**: Adjust based on your system resources and API rate limits
3. **Monitoring**: Check progress files in `logs/` directory while running
4. **Failed Jobs**: Failed jobs are tracked but not automatically retried - they'll be attempted again on resume

## Troubleshooting

### Checkpoint Not Found
If you get "No checkpoints found to resume from", check:
- The logs directory exists at `./data/outputs/pilot_experiment/logs/`
- Checkpoint files exist with pattern `checkpoint_*.json`
- Use `--list-checkpoints` to see available checkpoints

### Corrupted Checkpoint
If a checkpoint is corrupted:
- The system automatically creates a backup (`.backup.json`)
- You can manually restore from the backup if needed
- Start a fresh experiment if recovery isn't possible

### Resume Not Working
If resume isn't picking up where it left off:
- Check that you're using the correct experiment ID
- Verify the checkpoint file contains your completed jobs
- Ensure you haven't disabled resume with `--no-resume`
