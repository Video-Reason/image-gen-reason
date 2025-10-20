"""GPT-4O automatic evaluation for VMEvalKit."""

import json
import os
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import cv2
import numpy as np
from PIL import Image
import io
import asyncio
import httpx
from datetime import datetime

from .base import BaseEvaluator

logger = logging.getLogger(__name__)


class GPT4OEvaluator(BaseEvaluator):
    """Automatic evaluation using GPT-4O vision model.
    
    This evaluator uses OpenAI's GPT-4O model to automatically assess
    video generation results for reasoning capabilities.
    """
    
    def __init__(self, 
                 output_dir: str = "data/evaluations",
                 experiment_name: str = "pilot_experiment",
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o",
                 max_frames: int = 8,
                 temperature: float = 0.1):
        """Initialize the GPT-4O evaluator.
        
        Args:
            output_dir: Base directory for evaluation outputs
            experiment_name: Name of the experiment to evaluate
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: GPT-4O model to use
            max_frames: Maximum number of frames to extract from video
            temperature: Temperature for GPT-4O responses
        """
        super().__init__(output_dir, experiment_name)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")
        
        self.model = model
        self.max_frames = max_frames
        self.temperature = temperature
        self.base_url = "https://api.openai.com/v1"
        
        # HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
    
    def extract_frames(self, video_path: str, num_frames: int = 8) -> List[np.ndarray]:
        """Extract frames from video at regular intervals.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            List of frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"Video has no frames: {video_path}")
        
        # Calculate frame indices to extract
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            # Extract frames at regular intervals
            step = total_frames / num_frames
            frame_indices = [int(i * step) for i in range(num_frames)]
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def encode_image(self, image: Union[np.ndarray, str]) -> str:
        """Encode image to base64 string.
        
        Args:
            image: Image as numpy array or file path
            
        Returns:
            Base64 encoded image string
        """
        if isinstance(image, str):
            # Load image from file
            pil_image = Image.open(image)
        else:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
        
        # Convert to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        return base64.b64encode(image_bytes).decode('utf-8')
    
    async def call_gpt4o(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call GPT-4O API with messages.
        
        Args:
            messages: List of messages with text and images
            
        Returns:
            GPT-4O response
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 1000
        }
        
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"API call failed: {response.status_code} - {response.text}")
        
        return response.json()
    
    def create_evaluation_prompt(self, task_type: str) -> str:
        """Create task-specific evaluation prompt.
        
        Args:
            task_type: Type of task (e.g., 'chess_task', 'maze_task')
            
        Returns:
            Evaluation prompt string
        """
        base_prompt = """You are an expert evaluator for video generation models. 
Your task is to evaluate whether the solution shown in the generated video is correct.

Analyze the provided frames from the generated video and determine:

Is the solution shown in the video correct? Choose one:
- "Correct": The solution is fully correct
- "Incorrect": The solution is wrong
- "Partially Correct": The solution has some correct elements but is not fully correct

Provide your evaluation in JSON format:
{
    "solution_correctness": "<Correct/Incorrect/Partially Correct>",
    "explanation": "<brief explanation of why the solution is correct/incorrect>"
}
"""
        
        # Add task-specific guidance
        task_guidance = {
            "chess_task": "\nFor chess tasks, pay attention to whether the moves shown are legal and if they follow standard chess notation.",
            "maze_task": "\nFor maze tasks, verify that the path shown actually connects the start and end points without crossing walls.",
            "rotation_task": "\nFor rotation tasks, check if the rotation angle and direction match the instructions.",
            "raven_task": "\nFor Raven's matrices tasks, verify that the pattern completion follows logical rules.",
            "sudoku_task": "\nFor Sudoku tasks, check if the numbers placed follow Sudoku rules (no duplicates in rows, columns, or boxes)."
        }
        
        return base_prompt + task_guidance.get(task_type, "")
    
    async def evaluate_single_async(self,
                                   model_name: str,
                                   task_type: str,
                                   task_id: str,
                                   video_path: str,
                                   question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously evaluate a single video."""
        try:
            # Extract frames from video
            frames = self.extract_frames(video_path, self.max_frames)
            logger.info(f"Extracted {len(frames)} frames from video")
            
            # Load task images
            task_dir = Path(video_path).parent.parent
            first_frame_path = task_dir / "question" / "first_frame.png"
            prompt_path = task_dir / "question" / "prompt.txt"
            
            # Read prompt
            prompt_text = ""
            if prompt_path.exists():
                with open(prompt_path, 'r') as f:
                    prompt_text = f.read()
            
            # Create messages for GPT-4O
            messages = [
                {
                    "role": "system",
                    "content": self.create_evaluation_prompt(task_type)
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Task: {task_type}\nPrompt: {prompt_text}\n\nFirst, here is the input image:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self.encode_image(str(first_frame_path))}"
                            }
                        },
                        {
                            "type": "text",
                            "text": f"\nNow, here are {len(frames)} frames extracted from the generated video in chronological order:"
                        }
                    ]
                }
            ]
            
            # Add video frames
            for i, frame in enumerate(frames):
                messages[0]["content"].extend([
                    {
                        "type": "text",
                        "text": f"\nFrame {i+1}/{len(frames)}:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.encode_image(frame)}"
                        }
                    }
                ])
            
            messages[0]["content"].append({
                "type": "text",
                "text": "\nBased on these frames, provide your evaluation in the specified JSON format."
            })
            
            # Call GPT-4O
            response = await self.call_gpt4o(messages)
            
            # Parse response
            content = response["choices"][0]["message"]["content"]
            
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                evaluation_data = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse JSON from GPT-4O response")
            
            # Add metadata
            result = {
                "solution_correctness": evaluation_data.get("solution_correctness", "Unknown"),
                "explanation": evaluation_data.get("explanation", ""),
                "frames_analyzed": len(frames),
                "status": "completed"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}/{task_type}/{task_id}: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def evaluate_single(self,
                       model_name: str,
                       task_type: str,
                       task_id: str,
                       video_path: str,
                       question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single video generation result using GPT-4O.
        
        Args:
            model_name: Name of the model that generated the video
            task_type: Type of task (e.g., 'chess_task', 'maze_task')
            task_id: ID of the specific task (e.g., 'chess_0000')
            video_path: Path to the generated video
            question_data: Question metadata including prompt and images
            
        Returns:
            Dict containing evaluation results
        """
        # Run async evaluation in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.evaluate_single_async(model_name, task_type, task_id, video_path, question_data)
            )
            return result
        finally:
            loop.close()
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, 'client'):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.client.aclose())
                loop.close()
            except:
                pass
    
    def _calculate_summary(self, evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics from GPT-4O evaluation results.
        
        Args:
            evaluations: Dict of evaluation results by task type and task id
            
        Returns:
            Dict containing summary statistics
        """
        summary = super()._calculate_summary(evaluations)
        
        # Add GPT-4O specific statistics
        correctness_counts = {
            "Correct": 0,
            "Incorrect": 0,
            "Partially Correct": 0
        }
        task_type_counts = {}
        
        for task_type, tasks in evaluations.items():
            task_type_counts[task_type] = {
                "Correct": 0,
                "Incorrect": 0,
                "Partially Correct": 0
            }
            
            for task_id, result in tasks.items():
                if "error" not in result and "solution_correctness" in result:
                    correctness = result["solution_correctness"]
                    if correctness in correctness_counts:
                        correctness_counts[correctness] += 1
                        task_type_counts[task_type][correctness] += 1
        
        # Calculate percentages
        total_evaluated = sum(correctness_counts.values())
        if total_evaluated > 0:
            summary["correctness_percentages"] = {
                key: (count / total_evaluated) * 100
                for key, count in correctness_counts.items()
            }
            summary["accuracy"] = (correctness_counts["Correct"] / total_evaluated) * 100
        
        # Task type breakdowns
        summary["task_type_correctness"] = {}
        for task_type, counts in task_type_counts.items():
            total = sum(counts.values())
            if total > 0:
                summary["task_type_correctness"][task_type] = {
                    key: (count / total) * 100
                    for key, count in counts.items()
                }
        
        return summary
