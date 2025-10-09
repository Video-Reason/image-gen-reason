"""
Temporary image hosting utility using free services like imgbb or similar.
"""

import requests
import base64
from pathlib import Path
from typing import Optional
import time

class TempImageHost:
    """Upload images temporarily to public hosting services."""
    
    def __init__(self):
        # We'll use multiple services as fallbacks
        self.services = [
            self._upload_to_imgur,
            self._upload_to_tmpfiles,
            self._upload_to_fileio,
        ]
    
    def upload(self, image_path: str) -> Optional[str]:
        """
        Upload an image and return its public URL.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Public URL of the uploaded image, or None if all services fail
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Try each service until one works
        for service in self.services:
            try:
                url = service(path)
                if url:
                    print(f"[TempImageHost] Uploaded {path.name} to: {url}")
                    return url
            except Exception as e:
                print(f"[TempImageHost] Service failed: {e}")
                continue
        
        return None
    
    def _upload_to_imgur(self, path: Path) -> Optional[str]:
        """Upload to Imgur (anonymous upload, stored for months)."""
        # Imgur allows anonymous uploads with their public client ID
        client_id = "2c36c8a9b3f7dd7"  # Public client ID for anonymous uploads
        
        url = "https://api.imgur.com/3/image"
        headers = {"Authorization": f"Client-ID {client_id}"}
        
        with open(path, 'rb') as f:
            # Convert to base64 for imgur
            image_data = base64.b64encode(f.read()).decode()
        
        data = {
            'image': image_data,
            'type': 'base64',
            'name': path.name,
            'title': f'Maze test: {path.stem}'
        }
        
        response = requests.post(url, headers=headers, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result.get('data'):
                return result['data']['link']
        
        return None
    
    def _upload_to_tmpfiles(self, path: Path) -> Optional[str]:
        """Upload to tmpfiles.org (stores for 1 hour)."""
        # tmpfiles.org accepts direct file uploads
        url = "https://tmpfiles.org/api/v1/upload"
        
        with open(path, 'rb') as f:
            files = {'file': (path.name, f, 'image/png')}
            response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                # Convert the URL to direct link format
                file_url = data['data']['url']
                # tmpfiles.org format: https://tmpfiles.org/XXXXX/filename
                # Direct link format: https://tmpfiles.org/dl/XXXXX/filename
                direct_url = file_url.replace('/tmpfiles.org/', '/tmpfiles.org/dl/')
                return direct_url
        
        return None
    
    def _upload_to_fileio(self, path: Path) -> Optional[str]:
        """Upload to file.io (stores for 14 days by default)."""
        url = "https://file.io"
        
        with open(path, 'rb') as f:
            files = {'file': (path.name, f, 'image/png')}
            response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return data['link']
        
        return None


def upload_maze_images(maze_paths: list) -> dict:
    """
    Upload multiple maze images and return their URLs.
    
    Args:
        maze_paths: List of paths to maze images
        
    Returns:
        Dictionary mapping local paths to public URLs
    """
    uploader = TempImageHost()
    url_map = {}
    
    for path in maze_paths:
        print(f"\nUploading {path}...")
        url = uploader.upload(path)
        if url:
            url_map[path] = url
        else:
            print(f"Failed to upload {path}")
        
        # Small delay between uploads
        time.sleep(1)
    
    return url_map
