"""
WaveSpeedAI WAN 2.x Image-to-Video Generation Service.

Supports text + image → video generation using WaveSpeedAI's WAN 2.1 and WAN 2.2 models.
"""

import os
import httpx
import json
import asyncio
import base64
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class WANModel(str, Enum):
    """WaveSpeedAI WAN model endpoints for I2V generation."""
    
    # WAN 2.2 Models
    WAN_2_2_I2V_480P = "wan-2.2/i2v-480p"
    WAN_2_2_I2V_480P_ULTRA_FAST = "wan-2.2/i2v-480p-ultra-fast"
    WAN_2_2_I2V_480P_LORA = "wan-2.2/i2v-480p-lora"
    WAN_2_2_I2V_480P_LORA_ULTRA_FAST = "wan-2.2/i2v-480p-lora-ultra-fast"
    WAN_2_2_I2V_5B_720P = "wan-2.2/i2v-5b-720p"
    WAN_2_2_I2V_5B_720P_LORA = "wan-2.2/i2v-5b-720p-lora"
    WAN_2_2_I2V_720P = "wan-2.2/i2v-720p"
    WAN_2_2_I2V_720P_ULTRA_FAST = "wan-2.2/i2v-720p-ultra-fast"
    WAN_2_2_I2V_720P_LORA = "wan-2.2/i2v-720p-lora"
    WAN_2_2_I2V_720P_LORA_ULTRA_FAST = "wan-2.2/i2v-720p-lora-ultra-fast"
    
    # WAN 2.1 Models
    WAN_2_1_I2V_480P = "wan-2.1/i2v-480p"
    WAN_2_1_I2V_480P_ULTRA_FAST = "wan-2.1/i2v-480p-ultra-fast"
    WAN_2_1_I2V_480P_LORA = "wan-2.1/i2v-480p-lora"
    WAN_2_1_I2V_480P_LORA_ULTRA_FAST = "wan-2.1/i2v-480p-lora-ultra-fast"
    WAN_2_1_I2V_720P = "wan-2.1/i2v-720p"
    WAN_2_1_I2V_720P_ULTRA_FAST = "wan-2.1/i2v-720p-ultra-fast"
    WAN_2_1_I2V_720P_LORA = "wan-2.1/i2v-720p-lora"
    WAN_2_1_I2V_720P_LORA_ULTRA_FAST = "wan-2.1/i2v-720p-lora-ultra-fast"


class WaveSpeedService:
    """Service for image-to-video generation using WaveSpeedAI WAN models."""
    
    def __init__(self, model: str = WANModel.WAN_2_2_I2V_720P):
        """
        Initialize WaveSpeed service.
        
        Args:
            model: WAN model variant to use
        """
        self.api_key = os.getenv("WAVESPEED_API_KEY")
        if not self.api_key:
            raise ValueError("WAVESPEED_API_KEY environment variable is required")
        
        self.model = model
        self.base_url = "https://api.wavespeed.ai"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_video(
        self,
        prompt: str,
        image_path: Union[str, Path],
        seed: int = -1,
        output_path: Optional[Path] = None,
        poll_timeout_s: float = 300.0,
        poll_interval_s: float = 2.0
    ) -> Dict[str, Any]:
        """
        Generate video from text prompt and image.
        
        Args:
            prompt: Text description for video generation
            image_path: Path to input image file
            seed: Random seed for reproducibility (-1 for random)
            output_path: Optional path to save video (if not provided, returns URL)
            poll_timeout_s: Maximum time to wait for completion
            poll_interval_s: Time between polling attempts
            
        Returns:
            Dictionary with generation results including video URL/path
        """
        # Encode image to base64
        image_b64 = self._encode_image(image_path)
        
        # Submit generation request
        request_id = await self._submit_generation(prompt, image_b64, seed)
        logger.info(f"WaveSpeed generation started. Request ID: {request_id}")
        
        # Poll for completion
        result_url = await self._poll_generation(
            request_id, 
            poll_timeout_s, 
            poll_interval_s
        )
        
        result = {
            "video_url": result_url,
            "request_id": request_id,
            "model": self.model,
            "prompt": prompt,
            "image_path": str(image_path),
            "seed": seed
        }
        
        # Download video if output path provided
        if output_path and result_url:
            saved_path = await self._download_video(result_url, output_path)
            result["video_path"] = str(saved_path)
            logger.info(f"Video saved to: {saved_path}")
        
        return result
    
    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image file to base64."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(path, "rb") as f:
            image_data = f.read()
        
        return base64.b64encode(image_data).decode("utf-8")
    
    async def _submit_generation(self, prompt: str, image_b64: str, seed: int) -> str:
        """Submit I2V generation request."""
        # Build endpoint URL
        submit_url = f"{self.base_url}/api/v3/wavespeed-ai/{self.model}"
        
        payload = {
            "prompt": prompt,
            "image": image_b64,
            "seed": seed
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                submit_url,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                error_msg = f"Failed to submit generation: {response.status_code} {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            result_data = response.json()
            
            # Extract request ID from response
            if "data" in result_data:
                data = result_data["data"]
                if "id" in data:
                    return data["id"]
            
            # Fallback: try direct access
            if "id" in result_data:
                return result_data["id"]
                
            raise Exception(f"Invalid response format - no request ID found: {result_data}")
    
    async def _poll_generation(
        self, 
        request_id: str, 
        timeout_s: float, 
        interval_s: float
    ) -> str:
        """Poll for generation completion."""
        poll_url = f"{self.base_url}/api/v3/predictions/{request_id}/result"
        poll_headers = {"Authorization": f"Bearer {self.api_key}"}
        
        start_time = asyncio.get_event_loop().time()
        max_attempts = int(timeout_s / interval_s)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(max_attempts):
                if asyncio.get_event_loop().time() - start_time > timeout_s:
                    raise TimeoutError(f"Generation timed out after {timeout_s}s")
                
                try:
                    response = await client.get(poll_url, headers=poll_headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Handle data wrapper
                        if "data" in data:
                            data = data["data"]
                        
                        status = data.get("status", "unknown")
                        
                        if status == "completed":
                            outputs = data.get("outputs")
                            if outputs:
                                # Return first output URL
                                output_url = outputs[0] if isinstance(outputs, list) else outputs
                                logger.info(f"Generation completed successfully")
                                return output_url
                            else:
                                raise Exception("Generation completed but no output found")
                        
                        elif status == "failed":
                            error_msg = data.get("error", "Unknown error")
                            raise Exception(f"Generation failed: {error_msg}")
                        
                        elif status in ["starting", "processing", "pending", "queued"]:
                            logger.debug(f"Generation in progress... Status: {status}")
                        else:
                            logger.debug(f"Unknown status: {status}")
                    
                    else:
                        logger.warning(f"Poll request failed: {response.status_code}")
                
                except httpx.TimeoutException:
                    logger.warning(f"Poll request timed out on attempt {attempt + 1}")
                
                await asyncio.sleep(interval_s)
        
        raise TimeoutError(f"Generation timed out after {max_attempts} attempts")
    
    async def _download_video(self, video_url: str, output_path: Path) -> Path:
        """Download video from URL to local file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(video_url)
            if response.status_code != 200:
                raise Exception(f"Failed to download video: {response.status_code}")
            
            with open(output_path, "wb") as f:
                f.write(response.content)
        
        return output_path