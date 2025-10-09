#!/usr/bin/env python3
"""
Create visual icons for maze rendering (green dot, flag, trophy, etc.).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def create_green_dot_icon(size=64):
    """Create a green circle icon for start position."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Create green circle with white border
    circle = patches.Circle((0.5, 0.5), 0.35, 
                          facecolor='#4caf50', edgecolor='white', linewidth=3)
    ax.add_patch(circle)
    
    # Add inner highlight for 3D effect
    highlight = patches.Circle((0.45, 0.55), 0.15, 
                             facecolor='#66bb6a', alpha=0.7)
    ax.add_patch(highlight)
    
    return fig


def create_flag_icon(size=64):
    """Create a red flag icon for end position - improved visibility."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_xlim(0, 1) 
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Thicker, darker flag pole for better contrast
    pole = patches.Rectangle((0.15, 0.05), 0.12, 0.9, 
                           facecolor='#424242', edgecolor='#212121', linewidth=2)
    ax.add_patch(pole)
    
    # Larger rectangular flag with better proportions  
    flag = patches.Rectangle((0.27, 0.6), 0.6, 0.25, 
                           facecolor='#d32f2f', edgecolor='white', linewidth=3)
    ax.add_patch(flag)
    
    # Add a more prominent highlight for 3D effect
    highlight = patches.Rectangle((0.27, 0.72), 0.45, 0.08, 
                                facecolor='#ffcdd2', alpha=0.8)
    ax.add_patch(highlight)
    
    # Add wavy edge for realistic flag movement
    wave_x = np.linspace(0.87, 0.87, 6)
    wave_y = np.array([0.85, 0.82, 0.79, 0.76, 0.73, 0.70, 0.67, 0.64, 0.61])
    wave_amplitude = 0.03
    
    for i, y in enumerate(wave_y[:-1]):
        wave_offset = wave_amplitude * np.sin(i * 1.5)
        ax.plot([0.87, 0.87 + wave_offset], [y, wave_y[i+1]], 
               color='#d32f2f', linewidth=4)
    
    return fig


def create_trophy_icon(size=64):
    """Create a golden trophy icon for success - enhanced design."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1) 
    ax.axis('off')
    
    # Trophy base - larger and more detailed
    base = patches.FancyBboxPatch((0.25, 0.05), 0.5, 0.15, 
                                boxstyle="round,pad=0.02",
                                facecolor='#d97706', edgecolor='#92400e', linewidth=2)
    ax.add_patch(base)
    
    # Trophy stem with gradient effect
    stem = patches.Rectangle((0.44, 0.2), 0.12, 0.08, 
                           facecolor='#d97706', edgecolor='#92400e', linewidth=1)
    ax.add_patch(stem)
    
    # Main trophy bowl with better proportions
    bowl = patches.FancyBboxPatch((0.28, 0.28), 0.44, 0.5, 
                                boxstyle="round,pad=0.08",
                                facecolor='#fbbf24', edgecolor='#d97706', linewidth=3)
    ax.add_patch(bowl)
    
    # More prominent handles
    left_handle = patches.Arc((0.18, 0.5), 0.3, 0.4, angle=0, theta1=270, theta2=90, 
                            color='#d97706', linewidth=5)
    right_handle = patches.Arc((0.82, 0.5), 0.3, 0.4, angle=0, theta1=90, theta2=270,
                             color='#d97706', linewidth=5)
    ax.add_patch(left_handle)
    ax.add_patch(right_handle)
    
    # Multiple highlights for better 3D effect
    highlight1 = patches.Ellipse((0.42, 0.6), 0.18, 0.25, 
                               facecolor='#fef3c7', alpha=0.8)
    ax.add_patch(highlight1)
    
    highlight2 = patches.Ellipse((0.38, 0.65), 0.08, 0.12, 
                               facecolor='#fffbeb', alpha=0.9)
    ax.add_patch(highlight2)
    
    # Add decorative elements on the base
    decoration1 = patches.Rectangle((0.3, 0.08), 0.4, 0.03, 
                                  facecolor='#92400e', alpha=0.6)
    ax.add_patch(decoration1)
    
    decoration2 = patches.Rectangle((0.32, 0.14), 0.36, 0.02, 
                                  facecolor='#92400e', alpha=0.4)
    ax.add_patch(decoration2)
    
    return fig


def create_star_icon(size=64):
    """Create a blue star icon for moving element."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Create 5-pointed star
    angles = np.linspace(0, 2*np.pi, 11)  # 10 points + back to start
    outer_radius = 0.4
    inner_radius = 0.2
    
    star_points = []
    for i, angle in enumerate(angles[:-1]):  # Skip last point (same as first)
        if i % 2 == 0:  # Outer points
            x = 0.5 + outer_radius * np.cos(angle - np.pi/2)
            y = 0.5 + outer_radius * np.sin(angle - np.pi/2)
        else:  # Inner points
            x = 0.5 + inner_radius * np.cos(angle - np.pi/2)
            y = 0.5 + inner_radius * np.sin(angle - np.pi/2)
        star_points.append([x, y])
    
    star = patches.Polygon(star_points, 
                         facecolor='#2196f3', edgecolor='white', linewidth=2)
    ax.add_patch(star)
    
    # Star highlight
    highlight = patches.Circle((0.45, 0.55), 0.1, 
                             facecolor='#64b5f6', alpha=0.7)
    ax.add_patch(highlight)
    
    return fig


def save_icons(output_dir):
    """Generate and save all icon assets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    icons = {
        'green_dot.png': create_green_dot_icon(),
        'flag.png': create_flag_icon(), 
        'trophy.png': create_trophy_icon(),
        'star.png': create_star_icon()
    }
    
    for filename, fig in icons.items():
        filepath = output_path / filename
        fig.savefig(filepath, transparent=True, bbox_inches='tight', 
                   pad_inches=0, dpi=64, facecolor='none')
        plt.close(fig)
        print(f"✓ Created {filepath}")
    
    print(f"\n🎨 All icons created in {output_path}")


if __name__ == "__main__":
    save_icons("vmevalkit/assets/icons")
