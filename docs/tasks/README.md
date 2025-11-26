


## Examples

Solving Chess

![Chess Example](paper/video-models-start-to-solve/assets/chess_example.jpg)

Solving Maze

![Maze Example](paper/video-models-start-to-solve/assets/maze_example.jpg)

Mental Rotation

![Rotation Example](paper/video-models-start-to-solve/assets/rotation_example.jpg)

Raven's Matrices

![Raven Example](paper/video-models-start-to-solve/assets/raven_example.jpg)

Sudoku Solving

![Sudoku Example](paper/video-models-start-to-solve/assets/sudoku_example.jpg)




### External Benchmarks (HuggingFace)

| Dataset | Tasks | Domains | Key Features |
|---------|-------|---------|--------------|
| **VideoThinkBench** | ~4,000 | 4 subsets | Vision-centric (ARC-AGI, Eyeballing, Visual Puzzles) + Text-centric reasoning |
| **MME-CoF** | 59 | 16 domains | Video Chain-of-Frame reasoning across cognitive domains |



**VideoThinkBench Subsets:**
- `arc_agi_2` - Abstract reasoning (1,000 tasks)
- `eyeballing_puzzles` - Visual estimation (1,050 tasks)  
- `visual_puzzles` - Pattern recognition (496 tasks)
- `text_centric_tasks` - Math & multimodal reasoning (1,453 tasks)



### Sync with Cloud
```bash
# AWS S3 (enterprise backup)
python data/s3_sync.py --log
```

**Tips:**
- Use `--task-id chess_0001` to run specific questions  