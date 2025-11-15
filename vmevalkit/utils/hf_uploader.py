"""
HuggingFace uploader for VMEvalKit - handles question dataset uploads and downloads.
"""

import os
from pathlib import Path
from typing import Optional
from huggingface_hub import HfApi, snapshot_download


class HFUploader:
    """Upload and download VMEvalKit question datasets to/from HuggingFace Hub."""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize HuggingFace uploader.
        
        Args:
            token: HuggingFace API token (defaults to HF_TOKEN env var)
        """
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            print("[HF] Warning: No HF_TOKEN provided. Private repos will not be accessible.")
        
        self.api = HfApi(token=self.token)
    
    def upload_questions(
        self, 
        repo_id: str, 
        questions_dir: str = "data/questions",
        private: bool = False
    ) -> Optional[str]:
        """
        Upload questions dataset to HuggingFace.
        
        Args:
            repo_id: HuggingFace repo ID (e.g., 'username/vmeval-questions')
            questions_dir: Path to questions directory
            private: Make repository private
            
        Returns:
            URL to the dataset on HuggingFace, or None on failure
        """
        questions_path = Path(questions_dir)
        if not questions_path.exists():
            print(f"[HF] ❌ Questions directory not found: {questions_dir}")
            return None
        
        print(f"[HF] Creating repository: {repo_id}")
        self.api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        
        print(f"[HF] Uploading questions dataset...")
        self.api.upload_folder(
            folder_path=str(questions_path),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload VMEval questions dataset",
            ignore_patterns=[".DS_Store", "__pycache__", "*.pyc", ".git"]
        )
        
        url = f"https://huggingface.co/datasets/{repo_id}"
        print(f"[HF] ✅ Successfully uploaded to: {url}")
        return url
    
    def download_questions(
        self,
        repo_id: str,
        target_dir: str = "data/questions"
    ) -> Optional[Path]:
        """
        Download questions dataset from HuggingFace.
        
        Args:
            repo_id: HuggingFace repo ID
            target_dir: Target directory for download
            
        Returns:
            Path to downloaded directory, or None on failure
        """
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[HF] Downloading questions dataset from {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(target_path),
            token=self.token,
        )
        
        print(f"[HF] ✅ Downloaded to: {target_path}")
        return target_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Upload/Download VMEvalKit questions dataset to/from HuggingFace Hub"
    )
    
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload questions dataset")
    upload_parser.add_argument(
        "--questions-dir",
        default="data/questions",
        help="Path to questions directory (default: data/questions)"
    )
    upload_parser.add_argument(
        "--dataset-name",
        required=True,
        help="HuggingFace dataset name (e.g., Video-Model-Start-To-Solve)"
    )
    upload_parser.add_argument(
        "--username",
        default=os.getenv("HF_USERNAME"),
        help="HuggingFace username (default: HF_USERNAME env var)"
    )
    upload_parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download questions dataset")
    download_parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo ID (e.g., username/dataset-name)"
    )
    download_parser.add_argument(
        "--target-dir",
        default="data/questions",
        help="Target directory for download (default: data/questions)"
    )
    
    args = parser.parse_args()
    
    if not args.action:
        parser.print_help()
        exit(1)
    
    # Check HF_TOKEN
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HF_TOKEN environment variable not set")
        print("\nPlease set your HuggingFace token:")
        print("  export HF_TOKEN=hf_your_token_here")
        print("\nGet your token from: https://huggingface.co/settings/tokens")
        exit(1)
    
    uploader = HFUploader(token=hf_token)
    
    if args.action == "upload":
        if not args.username:
            print("❌ Error: HuggingFace username not provided")
            print("\nPlease provide username via:")
            print("  --username your-username")
            print("  OR export HF_USERNAME=your-username")
            exit(1)
        
        repo_id = f"{args.username}/{args.dataset_name}"
        
        print("=" * 60)
        print("📦 Uploading Questions Dataset to HuggingFace")
        print("=" * 60)
        print(f"Source: {args.questions_dir}")
        print(f"Target: {repo_id}")
        print(f"Visibility: {'Private' if args.private else 'Public'}")
        print("=" * 60)
        
        url = uploader.upload_questions(
            repo_id=repo_id,
            questions_dir=args.questions_dir,
            private=args.private
        )
        
        if url:
            print("\n" + "=" * 60)
            print("✅ Upload complete!")
            print(f"Dataset URL: {url}")
            print("=" * 60)
        else:
            print("\n❌ Upload failed")
            exit(1)
    
    elif args.action == "download":
        print("=" * 60)
        print("📥 Downloading Questions Dataset from HuggingFace")
        print("=" * 60)
        print(f"Source: {args.repo_id}")
        print(f"Target: {args.target_dir}")
        print("=" * 60)
        
        path = uploader.download_questions(
            repo_id=args.repo_id,
            target_dir=args.target_dir
        )
        
        if path:
            print("\n" + "=" * 60)
            print("✅ Download complete!")
            print(f"Location: {path}")
            print("=" * 60)
        else:
            print("\n❌ Download failed")
            exit(1)

