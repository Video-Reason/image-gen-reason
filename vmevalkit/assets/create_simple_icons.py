#!/usr/bin/env python3
"""
Create SIMPLE visual icons with clean and minimal styling.
No gradients or fancy effects.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def create_simple_green_dot(size=64):
    """Create a simple solid green circle."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_alpha(0)  # Transparent background
    
    # Simple solid green circle - no gradients, no effects
    circle = patches.Circle((0.5, 0.5), 0.4, 
                          facecolor='#22c55e', edgecolor='none')  # Simple green
    ax.add_patch(circle)
    
    return fig


def create_simple_flag(size=64):
    """Create a simple red flag - improved for better visibility."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_xlim(0, 1) 
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_alpha(0)  # Transparent background
    
    # Thicker, darker pole for better visibility
    pole = patches.Rectangle((0.12, 0.05), 0.1, 0.9, 
                           facecolor='#4a5568', edgecolor='none')  # Dark gray pole
    ax.add_patch(pole)
    
    # Larger rectangular flag with better proportions
    flag = patches.Rectangle((0.22, 0.55), 0.55, 0.3, 
                           facecolor='#dc2626', edgecolor='none')  # Bright red
    ax.add_patch(flag)
    
    # Add a white border to make it pop
    flag_border = patches.Rectangle((0.22, 0.55), 0.55, 0.3, 
                                  facecolor='none', edgecolor='white', linewidth=2)
    ax.add_patch(flag_border)
    
    # Add a small wavy effect to make it more flag-like
    wave_points = np.array([[0.77, 0.85], [0.82, 0.82], [0.77, 0.79], 
                           [0.82, 0.76], [0.77, 0.73], [0.82, 0.70], 
                           [0.77, 0.67], [0.82, 0.64], [0.77, 0.61], 
                           [0.82, 0.58], [0.77, 0.55]])
    for i in range(len(wave_points)-1):
        ax.plot([wave_points[i][0], wave_points[i+1][0]], 
               [wave_points[i][1], wave_points[i+1][1]], 
               color='#dc2626', linewidth=3)
    
    return fig


def create_simple_trophy(size=64):
    """Create a simple golden trophy - improved design."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1) 
    ax.axis('off')
    fig.patch.set_alpha(0)  # Transparent background
    
    # Trophy base - wider and more stable looking
    base = patches.Rectangle((0.25, 0.05), 0.5, 0.12, 
                           facecolor='#d97706', edgecolor='#92400e', linewidth=2)
    ax.add_patch(base)
    
    # Trophy stem
    stem = patches.Rectangle((0.45, 0.17), 0.1, 0.08, 
                           facecolor='#d97706', edgecolor='none')
    ax.add_patch(stem)
    
    # Trophy bowl - more bowl-like shape using FancyBboxPatch
    bowl = patches.FancyBboxPatch((0.3, 0.25), 0.4, 0.45, 
                                boxstyle="round,pad=0.05",
                                facecolor='#fbbf24', edgecolor='#d97706', linewidth=2)
    ax.add_patch(bowl)
    
    # Add simple handles
    left_handle = patches.Arc((0.2, 0.45), 0.25, 0.35, angle=0, theta1=270, theta2=90, 
                            color='#d97706', linewidth=4)
    right_handle = patches.Arc((0.8, 0.45), 0.25, 0.35, angle=0, theta1=90, theta2=270,
                             color='#d97706', linewidth=4)
    ax.add_patch(left_handle)
    ax.add_patch(right_handle)
    
    # Add a shine/highlight on the bowl
    highlight = patches.Ellipse((0.42, 0.55), 0.15, 0.25, 
                              facecolor='#fef3c7', alpha=0.7)
    ax.add_patch(highlight)
    
    return fig


def create_simple_star(size=64):
    """Create a simple blue star."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_alpha(0)  # Transparent background
    
    # Simple 5-pointed star
    angles = np.linspace(0, 2*np.pi, 11)  # 10 points + back to start
    outer_radius = 0.35
    inner_radius = 0.15
    
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
                         facecolor='#3b82f6', edgecolor='none')  # Simple blue
    ax.add_patch(star)
    
    return fig


def save_simple_icons(output_dir):
    """Generate and save simple icons."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    icons = {
        'green_dot.png': create_simple_green_dot(),
        'flag.png': create_simple_flag(), 
        'trophy.png': create_simple_trophy(),
        'star.png': create_simple_star()
    }
    
    for filename, fig in icons.items():
        filepath = output_path / filename
        fig.savefig(filepath, transparent=True, bbox_inches='tight', 
                   pad_inches=0, dpi=64, facecolor='none')
        plt.close(fig)
        print(f"✓ Created SIMPLE {filepath}")
    
    print(f"\n🎨 All SIMPLE icons created in {output_path}")
    print("Clean, minimal style icons generated.")


if __name__ == "__main__":
    save_simple_icons("vmevalkit/assets/icons")
