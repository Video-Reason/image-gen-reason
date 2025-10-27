#!/usr/bin/env python3
"""
Video Decomposition Utility for Research Paper Figures

This module provides utilities to decompose videos into temporal frame sequences,
specifically designed for creating publication-quality figures in research papers.
It extracts key frames at regular intervals and arranges them in various layouts
suitable for academic publications.

Key Features:
    - Extract N frames from videos at regular temporal intervals
    - Multiple layout options: horizontal timeline, vertical progression, grid arrangement
    - Publication-quality output in PNG and EPS (vector) formats
    - Model comparison figures for AI research
    - Customizable timestamps, frame numbers, and styling
    - Support for common video formats (MP4, WebM, AVI, MOV)

Typical Use Cases:
    1. Temporal Analysis: Show how AI-generated content evolves over time
    2. Model Comparison: Side-by-side comparison of different AI models
    3. Task Progression: Demonstrate progression through complex tasks
    4. Quality Analysis: Analyze video quality at different time points

Usage Examples:
    
    Basic single video decomposition:
    >>> from vmevalkit.utils import decompose_video
    >>> frames, figure_path = decompose_video(
    ...     video_path="path/to/video.mp4",
    ...     n_frames=4,
    ...     layout="horizontal",
    ...     title="Temporal Progression Analysis"
    ... )
    
    Model comparison figure:
    >>> from vmevalkit.utils import create_video_comparison_figure
    >>> figure_path = create_video_comparison_figure(
    ...     video_paths=["model1.mp4", "model2.mp4", "model3.mp4"],
    ...     model_names=["Sora", "Veo", "Luma"],
    ...     n_frames=4,
    ...     title="AI Model Comparison"
    ... )

Dependencies:
    - OpenCV (cv2): Video processing
    - NumPy: Array operations
    - Matplotlib: Figure generation and styling
    - Pathlib: File system operations

Notes:
    - All output figures are saved in both PNG (raster) and EPS (vector) formats
    - DPI defaults to 300 for publication quality
    - Frame extraction uses evenly spaced temporal sampling
    - Professional typography and styling applied automatically
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Tuple, Optional, Union, Literal
import logging

logger = logging.getLogger(__name__)

def decompose_video(
    video_path: Union[str, Path],
    n_frames: int = 4,
    output_dir: Optional[Union[str, Path]] = None,
    create_figure: bool = True,
    layout: Literal["horizontal", "vertical", "grid"] = "horizontal",
    figure_size: Tuple[int, int] = (16, 4),
    save_individual_frames: bool = False,
    frame_format: str = "png",
    dpi: int = 300,
    add_timestamps: bool = True,
    add_frame_numbers: bool = True,
    title: Optional[str] = None,
) -> Tuple[List[np.ndarray], Optional[str]]:
    """
    Decompose a video into a series of temporal frames for paper figures.
    
    This function extracts frames from a video at evenly spaced temporal intervals
    and optionally creates publication-quality figures. The frames are sampled
    across the entire video duration to show temporal progression.
    
    Args:
        video_path (Union[str, Path]): Path to the input video file. Supports
            common formats: MP4, WebM, AVI, MOV.
        n_frames (int, optional): Number of frames to extract. Default is 4.
            If n_frames >= total video frames, all frames are extracted.
        output_dir (Union[str, Path], optional): Directory to save outputs.
            If None, uses the same directory as the input video.
        create_figure (bool, optional): Whether to create a combined figure
            showing all frames. Default is True.
        layout (str, optional): How to arrange frames in the figure:
            - "horizontal": Side-by-side timeline (good for temporal progression)
            - "vertical": Stacked vertically (good for detailed comparison)  
            - "grid": Arranged in a roughly square grid (good for many frames)
            Default is "horizontal".
        figure_size (Tuple[int, int], optional): Size of the combined figure
            in inches as (width, height). Default is (16, 4).
        save_individual_frames (bool, optional): Whether to save each frame
            as a separate image file. Default is False.
        frame_format (str, optional): Format for individual frame files.
            Options: "png", "jpg", "eps". Default is "png".
        dpi (int, optional): DPI (dots per inch) for saved figures.
            Default is 300 for publication quality. Use 150+ for papers.
        add_timestamps (bool, optional): Whether to add timestamp labels
            (e.g., "t=2.5s") to frames. Default is True.
        add_frame_numbers (bool, optional): Whether to add frame number labels
            (e.g., "Frame 1") to frames. Default is True.
        title (str, optional): Custom title for the combined figure.
            If None and layout is "horizontal", auto-generates a title.
    
    Returns:
        Tuple[List[np.ndarray], Optional[str]]: A tuple containing:
            - List of frame arrays (RGB format, shape: [H, W, 3])
            - Path to the combined figure (if create_figure=True), else None
    
    Raises:
        FileNotFoundError: If the video file doesn't exist
        ValueError: If the video file cannot be opened or no frames can be extracted
        
    Examples:
        Extract 4 frames in a horizontal timeline:
        >>> frames, fig_path = decompose_video("video.mp4", n_frames=4)
        
        Create a vertical arrangement with custom title:
        >>> frames, fig_path = decompose_video(
        ...     "video.mp4", 
        ...     layout="vertical",
        ...     title="Chess Game Progression"
        ... )
        
        Extract many frames in a grid without figure creation:
        >>> frames, _ = decompose_video(
        ...     "video.mp4", 
        ...     n_frames=16, 
        ...     create_figure=False
        ... )
    """
    
    # Validate input and set up paths
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Set up output directory - use video's directory if not specified
    if output_dir is None:
        output_dir = video_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize video capture object
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    try:
        # Extract video metadata for temporal calculations
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        
        # Calculate which frames to extract - evenly distributed across video duration
        if n_frames >= total_frames:
            # If requested frames >= total frames, extract all frames
            frame_indices = list(range(total_frames))
        else:
            # Use linear interpolation to get evenly spaced frame indices
            frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        
        # Process each selected frame
        frames = []
        timestamps = []
        
        for i, frame_idx in enumerate(frame_indices):
            # Seek to the specific frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Could not read frame at index {frame_idx}")
                continue
            
            # Convert from OpenCV's BGR color format to RGB for matplotlib compatibility
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            # Calculate the timestamp for this frame
            timestamp = frame_idx / fps if fps > 0 else 0
            timestamps.append(timestamp)
            
            # Optionally save individual frame files
            if save_individual_frames:
                frame_filename = f"{video_path.stem}_frame_{i:03d}.{frame_format}"
                frame_path = output_dir / frame_filename
                
                if frame_format.lower() == 'eps':
                    # EPS (vector format) requires matplotlib for proper rendering
                    plt.figure(figsize=(8, 6))
                    plt.imshow(frame_rgb)
                    plt.axis('off')
                    plt.savefig(frame_path, format='eps', dpi=dpi, bbox_inches='tight',
                               pad_inches=0, facecolor='white')
                    plt.close()
                else:
                    # PNG/JPG can be saved directly with OpenCV (faster)
                    cv2.imwrite(str(frame_path), frame)
                
                logger.info(f"Saved frame {i+1}/{len(frame_indices)}: {frame_path}")
        
        # Validate that we successfully extracted frames
        if not frames:
            raise ValueError("No frames could be extracted from the video")
        
        logger.info(f"Successfully extracted {len(frames)} frames")
        
        # Create combined figure if requested
        figure_path = None
        if create_figure:
            figure_path = _create_combined_figure(
                frames=frames,
                timestamps=timestamps,
                video_name=video_path.stem,
                output_dir=output_dir,
                layout=layout,
                figure_size=figure_size,
                dpi=dpi,
                add_timestamps=add_timestamps,
                add_frame_numbers=add_frame_numbers,
                title=title,
                duration=duration
            )
        
        return frames, figure_path
        
    finally:
        cap.release()


def _create_combined_figure(
    frames: List[np.ndarray],
    timestamps: List[float],
    video_name: str,
    output_dir: Path,
    layout: str,
    figure_size: Tuple[int, int],
    dpi: int,
    add_timestamps: bool,
    add_frame_numbers: bool,
    title: Optional[str],
    duration: float
) -> str:
    """
    Create a publication-quality combined figure showing all frames in the specified layout.
    
    This internal function handles the matplotlib figure creation, styling, and export.
    It applies professional typography, arranges frames according to the layout,
    adds labels and annotations, and saves in both PNG and EPS formats.
    
    Args:
        frames: List of frame arrays in RGB format
        timestamps: List of timestamp values for each frame
        video_name: Name of the video (used for filename generation)
        output_dir: Directory to save the figure
        layout: Layout arrangement ("horizontal", "vertical", "grid")
        figure_size: Figure size in inches (width, height)
        dpi: Resolution for output files
        add_timestamps: Whether to add timestamp labels
        add_frame_numbers: Whether to add frame number labels
        title: Custom title for the figure
        duration: Total video duration for display
        
    Returns:
        str: Path to the created PNG figure
        
    Note:
        This function saves both PNG and EPS versions of the figure
        for maximum compatibility with different publication workflows.
    """
    
    n_frames = len(frames)
    
    # Configure matplotlib with professional styling for publications
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    
    # Calculate subplot arrangement based on layout preference
    if layout == "horizontal":
        # Single row, multiple columns - good for temporal progression
        nrows, ncols = 1, n_frames
        figsize = figure_size
    elif layout == "vertical":
        # Multiple rows, single column - good for detailed comparison
        nrows, ncols = n_frames, 1
        figsize = (figure_size[1], figure_size[0])  # Swap width/height for vertical
    elif layout == "grid":
        # Arrange in roughly square grid - good for many frames
        ncols = int(np.ceil(np.sqrt(n_frames)))  # Number of columns
        nrows = int(np.ceil(n_frames / ncols))   # Number of rows needed
        # Adjust figure size to maintain aspect ratio
        figsize = (figure_size[0], figure_size[0] * nrows / ncols)
    else:
        raise ValueError(f"Unknown layout: {layout}")
    
    # Create matplotlib figure and subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='white')
    
    # Handle matplotlib's inconsistent return types for different subplot configurations
    if n_frames == 1:
        # Single subplot returns bare Axes object, wrap in list
        axes = [axes]
    elif layout in ["horizontal", "vertical"]:
        # Linear arrangements return 1D array, ensure it's iterable
        axes = axes if hasattr(axes, '__iter__') else [axes]
    else:  # grid layout
        # 2D grid needs flattening to 1D for easy iteration
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Set main title if provided
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    elif layout == "horizontal":
        fig.suptitle(f'Temporal Decomposition: {video_name}', 
                    fontsize=14, fontweight='bold', y=0.95)
    
    # Plot each frame
    for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        if i >= len(axes):
            break
            
        ax = axes[i]
        ax.imshow(frame)
        ax.axis('off')
        
        # Add frame number and/or timestamp
        label_text = []
        if add_frame_numbers:
            label_text.append(f"Frame {i+1}")
        if add_timestamps:
            label_text.append(f"t={timestamp:.2f}s")
        
        if label_text:
            ax.text(0.02, 0.98, " | ".join(label_text), 
                   transform=ax.transAxes, fontsize=9, fontweight='bold',
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add subtle border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color('gray')
    
    # Hide unused subplots in grid layout
    if layout == "grid":
        for i in range(n_frames, len(axes)):
            axes[i].axis('off')
            axes[i].set_visible(False)
    
    # Add temporal arrow for horizontal layout
    if layout == "horizontal" and n_frames > 1:
        # Add arrow below the frames indicating time progression
        fig.text(0.1, 0.02, 'Time Progression', fontsize=11, fontweight='bold')
        fig.text(0.25, 0.02, '→', fontsize=16, fontweight='bold')
        fig.text(0.85, 0.02, f'Duration: {duration:.2f}s', fontsize=10, 
                horizontalalignment='right')
    
    plt.tight_layout()
    if title or (layout == "horizontal"):
        plt.subplots_adjust(top=0.9, bottom=0.1)
    
    # Save figure in multiple formats for publication
    base_filename = f"{video_name}_decomposed_{n_frames}frames_{layout}"
    
    # Save PNG (high quality)
    png_path = output_dir / f"{base_filename}.png"
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    
    # Save EPS (vector format for publications)
    eps_path = output_dir / f"{base_filename}.eps"
    plt.savefig(eps_path, format='eps', bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    plt.close()
    
    logger.info(f"Created combined figure: {png_path}")
    logger.info(f"Created vector figure: {eps_path}")
    
    return str(png_path)


def create_video_comparison_figure(
    video_paths: List[Union[str, Path]],
    n_frames: int = 4,
    output_dir: Optional[Union[str, Path]] = None,
    model_names: Optional[List[str]] = None,
    figure_size: Tuple[int, int] = (16, 10),
    dpi: int = 300,
    title: Optional[str] = None
) -> str:
    """
    Create a comparison figure showing temporal decomposition of multiple videos.
    
    This function is particularly useful for AI research papers where you need to
    compare the temporal evolution of videos generated by different models. It creates
    a grid layout where each row represents a different model/video and each column
    represents a temporal frame, allowing for easy visual comparison.
    
    Args:
        video_paths (List[Union[str, Path]]): List of paths to video files to compare.
            All videos should ideally be of similar duration for meaningful comparison.
        n_frames (int, optional): Number of frames to extract from each video.
            Default is 4. Frames are extracted at the same relative time points
            across all videos for fair comparison.
        output_dir (Union[str, Path], optional): Directory to save the output figure.
            If None, uses the directory of the first video.
        model_names (List[str], optional): List of model names for row labels.
            Must have the same length as video_paths. If None, uses video filenames.
        figure_size (Tuple[int, int], optional): Size of the figure in inches
            as (width, height). Default is (16, 10) which works well for most papers.
        dpi (int, optional): DPI for the saved figure. Default is 300 for
            publication quality. Use 150+ for academic papers.
        title (str, optional): Custom title for the entire comparison figure.
            If None, uses a default comparison title.
            
    Returns:
        str: Path to the created comparison figure (PNG format)
        
    Raises:
        ValueError: If video_paths is empty or if model_names length doesn't match video_paths
        FileNotFoundError: If any video file doesn't exist
        
    Examples:
        Compare three AI models on the same task:
        >>> figure_path = create_video_comparison_figure(
        ...     video_paths=["sora_output.mp4", "veo_output.mp4", "luma_output.mp4"],
        ...     model_names=["OpenAI Sora", "Google Veo 3.0", "Luma Dream Machine"],
        ...     n_frames=4,
        ...     title="Chess Game Generation - Model Comparison"
        ... )
        
        Quick comparison using filenames as labels:
        >>> figure_path = create_video_comparison_figure(
        ...     ["model1.mp4", "model2.mp4"],
        ...     n_frames=6
        ... )
    
    Note:
        The function saves both PNG and EPS versions of the figure. The figure
        layout is optimized for academic publications with professional typography
        and clear model/temporal labeling.
    """
    
    # Input validation
    if not video_paths:
        raise ValueError("At least one video path must be provided")
    
    video_paths = [Path(p) for p in video_paths]
    n_videos = len(video_paths)
    
    # Set up output directory - use first video's directory if not specified
    if output_dir is None:
        output_dir = video_paths[0].parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle model naming - use filenames if not provided
    if model_names is None:
        model_names = [p.stem for p in video_paths]
    elif len(model_names) != n_videos:
        raise ValueError("Number of model names must match number of videos")
    
    # Process each video to extract frames and calculate timestamps
    all_frames = []
    all_timestamps = []
    
    for video_path in video_paths:
        # Extract frames using the main decompose_video function
        frames, _ = decompose_video(
            video_path=video_path,
            n_frames=n_frames,
            create_figure=False,        # Don't create individual figures
            save_individual_frames=False  # Don't save individual frames
        )
        
        # Calculate exact timestamps for this specific video
        # (needed because different videos may have different FPS/duration)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Use same temporal sampling as decompose_video for consistency
        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        timestamps = [idx / fps if fps > 0 else 0 for idx in frame_indices]
        
        all_frames.append(frames)
        all_timestamps.append(timestamps)
    
    # Set up matplotlib
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 9
    
    # Create subplot grid: rows = videos, cols = frames
    fig, axes = plt.subplots(n_videos, n_frames, figsize=figure_size, facecolor='white')
    
    # Handle single video or single frame cases
    if n_videos == 1:
        axes = [axes] if n_frames > 1 else [[axes]]
    elif n_frames == 1:
        axes = [[ax] for ax in axes]
    
    # Set main title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    else:
        fig.suptitle('Video Generation Model Comparison - Temporal Decomposition',
                    fontsize=14, fontweight='bold', y=0.95)
    
    # Plot frames for each video
    for video_idx, (frames, timestamps, model_name) in enumerate(zip(all_frames, all_timestamps, model_names)):
        for frame_idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            ax = axes[video_idx][frame_idx]
            ax.imshow(frame)
            ax.axis('off')
            
            # Add model name to first frame of each row
            if frame_idx == 0:
                ax.text(-0.02, 0.5, model_name, transform=ax.transAxes,
                       fontsize=11, fontweight='bold', rotation=90,
                       verticalalignment='center', horizontalalignment='right')
            
            # Add timestamp to frames in first row
            if video_idx == 0:
                ax.text(0.5, -0.02, f't={timestamp:.2f}s', transform=ax.transAxes,
                       fontsize=9, horizontalalignment='center', verticalalignment='top')
            
            # Add subtle border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color('gray')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, left=0.1, bottom=0.1)
    
    # Save figure
    timestamp_str = "_".join([f"{ts:.1f}s" for ts in all_timestamps[0]])
    filename = f"video_comparison_{n_videos}models_{n_frames}frames.png"
    output_path = output_dir / filename
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    # Also save EPS version
    eps_path = output_dir / filename.replace('.png', '.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    plt.close()
    
    logger.info(f"Created video comparison figure: {output_path}")
    
    return str(output_path)


# Command Line Interface and Example Usage
# This section provides a CLI for the video decomposer utility, allowing users to
# process videos directly from the command line without writing Python code.
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Decompose video into temporal frames for paper figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 4 frames horizontally
  python -m vmevalkit.utils.video_decomposer video.mp4 --frames 4 --layout horizontal
  
  # Create comparison figure for multiple models
  python -m vmevalkit.utils.video_decomposer model1.mp4 model2.mp4 model3.mp4 --comparison
  
  # Extract frames vertically with timestamps
  python -m vmevalkit.utils.video_decomposer video.mp4 --frames 6 --layout vertical --timestamps
        """
    )
    
    parser.add_argument("video_paths", nargs="+", help="Path(s) to video file(s)")
    parser.add_argument("--frames", "-f", type=int, default=4,
                       help="Number of frames to extract (default: 4)")
    parser.add_argument("--output", "-o", type=str,
                       help="Output directory (default: same as video)")
    parser.add_argument("--layout", "-l", choices=["horizontal", "vertical", "grid"],
                       default="horizontal", help="Frame layout (default: horizontal)")
    parser.add_argument("--comparison", "-c", action="store_true",
                       help="Create comparison figure for multiple videos")
    parser.add_argument("--individual", "-i", action="store_true",
                       help="Also save individual frame images")
    parser.add_argument("--no-timestamps", action="store_true",
                       help="Don't add timestamp labels")
    parser.add_argument("--no-frame-numbers", action="store_true",
                       help="Don't add frame number labels")
    parser.add_argument("--title", "-t", type=str,
                       help="Title for the figure")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for saved figures (default: 300)")
    
    args = parser.parse_args()
    
    if args.comparison and len(args.video_paths) > 1:
        # Create comparison figure
        create_video_comparison_figure(
            video_paths=args.video_paths,
            n_frames=args.frames,
            output_dir=args.output,
            title=args.title,
            dpi=args.dpi
        )
    else:
        # Process individual videos
        for video_path in args.video_paths:
            frames, figure_path = decompose_video(
                video_path=video_path,
                n_frames=args.frames,
                output_dir=args.output,
                layout=args.layout,
                save_individual_frames=args.individual,
                add_timestamps=not args.no_timestamps,
                add_frame_numbers=not args.no_frame_numbers,
                title=args.title,
                dpi=args.dpi
            )
            
            print(f"✅ Processed {video_path}")
            if figure_path:
                print(f"   📊 Created figure: {figure_path}")
