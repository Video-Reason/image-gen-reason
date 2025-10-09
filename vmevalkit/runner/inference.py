"""
Simple inference runner for video generation.

Handles running different models with a clean interface.
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import json

from ..models import LumaInference, VeoService, WaveSpeedService, RunwayService


class VeoWrapper:
    """
    Wrapper for VeoService to match the LumaInference interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./outputs",
        api_key: Optional[str] = None,  # Not used for Veo (uses GCP auth)
        **kwargs
    ):
        """Initialize Veo wrapper."""
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Create VeoService instance
        self.veo_service = VeoService(model_id=model, **kwargs)
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using Veo (synchronous wrapper).
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds (4, 6, or 8 for Veo)
            output_filename: Optional output filename
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generation results
        """
        import time
        start_time = time.time()
        
        # Convert duration to int (Veo requires int)
        duration_seconds = int(duration)
        
        # Run async generation in sync context
        video_bytes, metadata = asyncio.run(
            self.veo_service.generate_video(
                prompt=text_prompt,
                image_path=str(image_path),
                duration_seconds=duration_seconds,
                **kwargs
            )
        )
        
        # Save video if we got bytes
        video_path = None
        if video_bytes:
            if not output_filename:
                # Generate filename from operation name if available
                op_name = metadata.get('operation_name', 'veo_output')
                if isinstance(op_name, str) and '/' in op_name:
                    op_id = op_name.split('/')[-1]
                    output_filename = f"veo_{op_id}.mp4"
                else:
                    output_filename = f"veo_{int(time.time())}.mp4"
            
            video_path = self.output_dir / output_filename
            asyncio.run(self.veo_service.save_video(video_bytes, video_path))
        
        duration_taken = time.time() - start_time
        
        return {
            "video_path": str(video_path) if video_path else None,
            "generation_id": metadata.get('operation_name', 'unknown'),
            "status": "success" if video_bytes else "completed_no_download",
            "duration_seconds": duration_taken,
            "model": self.model,
            "prompt": text_prompt,
            "image_path": str(image_path),
            "metadata": metadata
        }


class WaveSpeedWrapper:
    """
    Wrapper for WaveSpeedService to match the LumaInference interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./outputs",
        api_key: Optional[str] = None,  # Not used - WaveSpeed uses WAVESPEED_API_KEY env var
        **kwargs
    ):
        """Initialize WaveSpeed wrapper."""
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Create WaveSpeedService instance
        self.wavespeed_service = WaveSpeedService(model=model)
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,  # Not used by WaveSpeed but kept for interface compatibility
        output_filename: Optional[str] = None,
        seed: int = -1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using WaveSpeed (synchronous wrapper).
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            duration: Not used (kept for compatibility)
            output_filename: Optional output filename
            seed: Random seed for reproducibility
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generation results
        """
        import time
        start_time = time.time()
        
        # Generate output path
        if not output_filename:
            # Create filename from model and timestamp
            safe_model = self.model.replace('/', '_').replace('-', '_')
            timestamp = int(time.time())
            output_filename = f"wavespeed_{safe_model}_{timestamp}.mp4"
        
        output_path = self.output_dir / output_filename
        
        # Run async generation in sync context
        result = asyncio.run(
            self.wavespeed_service.generate_video(
                prompt=text_prompt,
                image_path=str(image_path),
                seed=seed,
                output_path=output_path,
                **kwargs
            )
        )
        
        duration_taken = time.time() - start_time
        
        return {
            "video_path": result.get("video_path"),
            "generation_id": result.get("request_id", 'unknown'),
            "status": "success" if result.get("video_path") else "completed_no_download",
            "duration_seconds": duration_taken,
            "model": self.model,
            "prompt": text_prompt,
            "image_path": str(image_path),
            "video_url": result.get("video_url"),
            "seed": seed
        }


class RunwayWrapper:
    """
    Wrapper for RunwayService to match the LumaInference interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./outputs",
        api_key: Optional[str] = None,  # Not used - Runway uses RUNWAYML_API_SECRET env var
        **kwargs
    ):
        """Initialize Runway wrapper."""
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Create RunwayService instance
        self.runway_service = RunwayService(model=model)
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 5.0,
        output_filename: Optional[str] = None,
        ratio: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using Runway (synchronous wrapper).
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds (5 or 10 depending on model)
            output_filename: Optional output filename
            ratio: Video aspect ratio (model-specific)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generation results
        """
        import time
        start_time = time.time()
        
        # Convert duration to int (Runway expects int)
        duration_int = int(duration)
        
        # Generate output path
        if not output_filename:
            # Create filename from model and timestamp
            safe_model = self.model.replace('_', '-')
            timestamp = int(time.time())
            output_filename = f"runway_{safe_model}_{timestamp}.mp4"
        
        output_path = self.output_dir / output_filename
        
        # Run async generation in sync context
        result = asyncio.run(
            self.runway_service.generate_video(
                prompt=text_prompt,
                image_path=str(image_path),
                duration=duration_int,
                ratio=ratio,
                output_path=output_path,
                **kwargs
            )
        )
        
        duration_taken = time.time() - start_time
        
        return {
            "video_path": result.get("video_path"),
            "generation_id": result.get("task_id", 'unknown'),
            "status": "success" if result.get("video_path") else "completed_no_download",
            "duration_seconds": duration_taken,
            "model": self.model,
            "prompt": text_prompt,
            "image_path": str(image_path),
            "video_url": result.get("video_url"),
            "duration": duration_int,
            "ratio": result.get("ratio")
        }


# Available models and their configurations
AVAILABLE_MODELS = {
    "luma-ray-2": {
        "class": LumaInference,
        "model": "ray-2",
        "description": "Luma Ray 2 - Latest model with best quality"
    },
    "luma-ray-flash-2": {
        "class": LumaInference,
        "model": "ray-flash-2", 
        "description": "Luma Ray Flash 2 - Faster generation"
    },
    "veo-2.0-generate": {
        "class": VeoWrapper,
        "model": "veo-2.0-generate-001",
        "description": "Google Veo 2.0 - GA model for text+image→video"
    },
    "veo-3.0-generate": {
        "class": VeoWrapper,
        "model": "veo-3.0-generate-preview",
        "description": "Google Veo 3.0 - Preview model with advanced capabilities"
    },
    "veo-3.0-fast-generate": {
        "class": VeoWrapper,
        "model": "veo-3.0-fast-generate-preview",
        "description": "Google Veo 3.0 Fast - Preview model for faster generation"
    },
    
    # WaveSpeedAI WAN 2.2 Models
    "wavespeed-wan-2.2-i2v-480p": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-480p",
        "description": "WaveSpeed WAN 2.2 I2V 480p - Standard quality"
    },
    "wavespeed-wan-2.2-i2v-480p-ultra-fast": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-480p-ultra-fast",
        "description": "WaveSpeed WAN 2.2 I2V 480p Ultra Fast - Speed optimized"
    },
    "wavespeed-wan-2.2-i2v-480p-lora": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-480p-lora",
        "description": "WaveSpeed WAN 2.2 I2V 480p LoRA - Enhanced with LoRA"
    },
    "wavespeed-wan-2.2-i2v-480p-lora-ultra-fast": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-480p-lora-ultra-fast",
        "description": "WaveSpeed WAN 2.2 I2V 480p LoRA Ultra Fast - Best speed with LoRA"
    },
    "wavespeed-wan-2.2-i2v-5b-720p": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-5b-720p",
        "description": "WaveSpeed WAN 2.2 I2V 5B 720p - High resolution 5B model"
    },
    "wavespeed-wan-2.2-i2v-5b-720p-lora": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-5b-720p-lora",
        "description": "WaveSpeed WAN 2.2 I2V 5B 720p LoRA - High-res with LoRA"
    },
    "wavespeed-wan-2.2-i2v-720p": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-720p",
        "description": "WaveSpeed WAN 2.2 I2V 720p - High resolution"
    },
    "wavespeed-wan-2.2-i2v-720p-ultra-fast": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-720p-ultra-fast",
        "description": "WaveSpeed WAN 2.2 I2V 720p Ultra Fast - High-res speed optimized"
    },
    "wavespeed-wan-2.2-i2v-720p-lora": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-720p-lora",
        "description": "WaveSpeed WAN 2.2 I2V 720p LoRA - High-res with LoRA"
    },
    "wavespeed-wan-2.2-i2v-720p-lora-ultra-fast": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-720p-lora-ultra-fast",
        "description": "WaveSpeed WAN 2.2 I2V 720p LoRA Ultra Fast - Fastest high-res LoRA"
    },
    
    # WaveSpeedAI WAN 2.1 Models
    "wavespeed-wan-2.1-i2v-480p": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.1/i2v-480p",
        "description": "WaveSpeed WAN 2.1 I2V 480p - Standard quality"
    },
    "wavespeed-wan-2.1-i2v-480p-ultra-fast": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.1/i2v-480p-ultra-fast",
        "description": "WaveSpeed WAN 2.1 I2V 480p Ultra Fast - Speed optimized"
    },
    "wavespeed-wan-2.1-i2v-480p-lora": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.1/i2v-480p-lora",
        "description": "WaveSpeed WAN 2.1 I2V 480p LoRA - Enhanced with LoRA"
    },
    "wavespeed-wan-2.1-i2v-480p-lora-ultra-fast": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.1/i2v-480p-lora-ultra-fast",
        "description": "WaveSpeed WAN 2.1 I2V 480p LoRA Ultra Fast - Best speed with LoRA"
    },
    "wavespeed-wan-2.1-i2v-720p": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.1/i2v-720p",
        "description": "WaveSpeed WAN 2.1 I2V 720p - High resolution"
    },
    "wavespeed-wan-2.1-i2v-720p-ultra-fast": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.1/i2v-720p-ultra-fast",
        "description": "WaveSpeed WAN 2.1 I2V 720p Ultra Fast - High-res speed optimized"
    },
    "wavespeed-wan-2.1-i2v-720p-lora": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.1/i2v-720p-lora",
        "description": "WaveSpeed WAN 2.1 I2V 720p LoRA - High-res with LoRA"
    },
    "wavespeed-wan-2.1-i2v-720p-lora-ultra-fast": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.1/i2v-720p-lora-ultra-fast",
        "description": "WaveSpeed WAN 2.1 I2V 720p LoRA Ultra Fast - Fastest high-res LoRA"
    },
    
    # Runway ML Models
    "runway-gen4-turbo": {
        "class": RunwayWrapper,
        "model": "gen4_turbo",
        "description": "Runway Gen-4 Turbo - Fast high-quality generation (5s or 10s)"
    },
    "runway-gen4-aleph": {
        "class": RunwayWrapper,
        "model": "gen4_aleph",
        "description": "Runway Gen-4 Aleph - Premium quality (5s)"
    },
    "runway-gen3a-turbo": {
        "class": RunwayWrapper,
        "model": "gen3a_turbo",
        "description": "Runway Gen-3A Turbo - Proven performance (5s or 10s)"
    }
}


def run_inference(
    model_name: str,
    image_path: Union[str, Path],
    text_prompt: str,
    output_dir: str = "./outputs",
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run inference with specified model.
    
    Args:
        model_name: Name of model to use (e.g., "luma-ray-2", "luma-ray-flash-2")
        image_path: Path to input image
        text_prompt: Text instructions for video generation
        output_dir: Directory to save outputs
        api_key: Optional API key (uses env var if not provided)
        **kwargs: Additional model-specific parameters
        
    Returns:
        Dictionary with inference results
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(AVAILABLE_MODELS.keys())}"
        )
    
    model_config = AVAILABLE_MODELS[model_name]
    model_class = model_config["class"]
    
    # Create model instance with specific configuration
    model = model_class(
        api_key=api_key,
        model=model_config["model"],
        output_dir=output_dir,
        **kwargs
    )
    
    # Run inference
    return model.generate(image_path, text_prompt)


class InferenceRunner:
    """
    Simple inference runner for managing video generation.
    """
    
    def __init__(self, output_dir: str = "./outputs"):
        """
        Initialize runner.
        
        Args:
            output_dir: Directory to save generated videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Simple logging to track runs
        self.log_file = self.output_dir / "inference_log.json"
        self.runs = self._load_log()
    
    def run(
        self,
        model_name: str,
        image_path: Union[str, Path],
        text_prompt: str,
        run_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference and log results.
        
        Args:
            model_name: Model to use
            image_path: Input image
            text_prompt: Text instructions
            run_id: Optional run identifier
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with results
        """
        start_time = datetime.now()
        
        # Generate run ID if not provided
        if not run_id:
            run_id = f"{model_name}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Run inference
            result = run_inference(
                model_name=model_name,
                image_path=image_path,
                text_prompt=text_prompt,
                output_dir=self.output_dir,
                **kwargs
            )
            
            # Add metadata
            result["run_id"] = run_id
            result["timestamp"] = start_time.isoformat()
            
            # Log the run
            self._log_run(run_id, result)
            
            return result
            
        except Exception as e:
            # Log failure
            error_result = {
                "run_id": run_id,
                "status": "failed",
                "error": str(e),
                "model": model_name,
                "timestamp": start_time.isoformat()
            }
            self._log_run(run_id, error_result)
            return error_result
    
    def list_models(self) -> Dict[str, str]:
        """List available models and their descriptions."""
        return {
            name: config["description"]
            for name, config in AVAILABLE_MODELS.items()
        }
    
    def _load_log(self) -> list:
        """Load existing run log."""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return []
    
    def _log_run(self, run_id: str, result: Dict[str, Any]):
        """Log a run to the log file."""
        self.runs.append({
            "run_id": run_id,
            **result
        })
        
        with open(self.log_file, 'w') as f:
            json.dump(self.runs, f, indent=2)
