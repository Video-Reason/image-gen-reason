"""
Runway ML Image-to-Video Generation Service.

Supports text + image → video generation using Runway's Gen-3 and Gen-4 models.
"""

import os
import time
import asyncio
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RunwayService:
    """Service for image-to-video generation using Runway ML models."""
    
    def __init__(self, model: str = "gen4_turbo"):
        """
        Initialize Runway service.
        
        Args:
            model: Runway model to use (gen4_turbo, gen4_aleph, gen3a_turbo)
        """
        self.api_secret = os.getenv("RUNWAYML_API_SECRET")
        if not self.api_secret:
            raise ValueError("RUNWAYML_API_SECRET environment variable is required")
        
        self.model = model
        
        # Validate model and set constraints
        self.model_constraints = self._get_model_constraints(model)
    
    def _get_model_constraints(self, model: str) -> Dict[str, Any]:
        """Get model-specific constraints."""
        constraints = {
            "gen4_turbo": {
                "durations": [5, 10],
                "ratios": ["1280:720", "720:1280", "1024:1024"],  # 720p landscape/portrait, square
                "description": "Runway Gen-4 Turbo - Fast high-quality generation"
            },
            "gen4_aleph": {
                "durations": [5],
                "ratios": ["1280:720", "720:1280", "1024:1024"],  # 720p landscape/portrait, square
                "description": "Runway Gen-4 Aleph - Premium quality"
            },
            "gen3a_turbo": {
                "durations": [5, 10],
                "ratios": ["1280:768", "768:1280"],  # Gen-3 specific ratios
                "description": "Runway Gen-3A Turbo - Proven performance"
            }
        }
        
        if model not in constraints:
            raise ValueError(f"Unknown Runway model: {model}. Available: {list(constraints.keys())}")
        
        return constraints[model]
    
    async def generate_video(
        self,
        prompt: str,
        image_path: Union[str, Path],
        duration: int = 5,
        ratio: Optional[str] = None,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate video from text prompt and image.
        
        Args:
            prompt: Text description for video generation
            image_path: Path to input image file (will be uploaded to get URL)
            duration: Video duration in seconds
            ratio: Video aspect ratio (if not provided, uses first available)
            output_path: Optional path to save video
            
        Returns:
            Dictionary with generation results
        """
        # Validate duration
        if duration not in self.model_constraints["durations"]:
            valid_durations = self.model_constraints["durations"]
            logger.warning(f"Duration {duration}s not supported for {self.model}. Using {valid_durations[0]}s")
            duration = valid_durations[0]
        
        # Set default ratio if not provided
        if not ratio:
            ratio = self.model_constraints["ratios"][0]
        elif ratio not in self.model_constraints["ratios"]:
            valid_ratios = self.model_constraints["ratios"]
            logger.warning(f"Ratio {ratio} not supported for {self.model}. Using {valid_ratios[0]}")
            ratio = valid_ratios[0]
        
        # Upload image to get URL (Runway requires image URLs)
        image_url = await self._upload_image(image_path)
        
        # Generate video using Runway SDK
        result = await self._generate_with_runway(prompt, image_url, duration, ratio)
        
        # Download video if output path provided
        if output_path and result.get("video_url"):
            saved_path = await self._download_video(result["video_url"], output_path)
            result["video_path"] = str(saved_path)
            logger.info(f"Video saved to: {saved_path}")
        
        result.update({
            "model": self.model,
            "prompt": prompt,
            "image_path": str(image_path),
            "duration": duration,
            "ratio": ratio
        })
        
        return result
    
    async def _upload_image(self, image_path: Union[str, Path]) -> str:
        """
        Upload image and return URL. 
        For now, this is a placeholder - in practice you'd need to upload to a CDN/S3.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # TODO: Implement actual image upload to CDN/S3
        # For now, we'll assume the user provides a publicly accessible image URL
        # or we could integrate with the existing S3 uploader from Luma
        from ..utils.s3_uploader import S3ImageUploader
        
        try:
            s3_uploader = S3ImageUploader()
            image_url = s3_uploader.upload(str(image_path))
            if not image_url:
                raise Exception("Failed to upload image to S3")
            logger.info(f"Uploaded image to: {image_url}")
            return image_url
        except Exception as e:
            logger.error(f"Failed to upload image: {e}")
            raise Exception(f"Image upload failed: {e}")
    
    async def _generate_with_runway(
        self, 
        prompt: str, 
        image_url: str, 
        duration: int, 
        ratio: str
    ) -> Dict[str, Any]:
        """Generate video using Runway SDK."""
        try:
            from runwayml import RunwayML, TaskFailedError
        except ImportError:
            raise ImportError("runwayml package not installed. Run: pip install runwayml")
        
        # Run in thread pool since Runway SDK is synchronous
        def _sync_generate():
            client = RunwayML()
            
            try:
                task = client.image_to_video.create(
                    model=self.model,
                    prompt_image=image_url,
                    prompt_text=prompt,
                    ratio=ratio,
                    duration=duration
                ).wait_for_task_output()
                
                return {
                    "task_id": task.id if hasattr(task, 'id') else 'unknown',
                    "video_url": task.output if hasattr(task, 'output') else None,
                    "status": "success"
                }
                
            except TaskFailedError as e:
                logger.error(f"Runway task failed: {e.task_details}")
                raise Exception(f"Runway generation failed: {e.task_details}")
            except Exception as e:
                logger.error(f"Runway SDK error: {e}")
                raise Exception(f"Runway generation error: {e}")
        
        # Run synchronous Runway call in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_generate)
    
    async def _download_video(self, video_url: str, output_path: Path) -> Path:
        """Download video from URL to local file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import httpx
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(video_url)
            if response.status_code != 200:
                raise Exception(f"Failed to download video: {response.status_code}")
            
            with open(output_path, "wb") as f:
                f.write(response.content)
        
        return output_path
