from irl_scg.HypoDataset import HypoDataset, FILE_NUMBERS_BY_PIG
from irl_scg import config
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cmap_color = 'gray'

BVDS_LEVELS_DIR = config.PROCESSED_DATA_DIR / 'bvds_levels'

file_paths = sorted([BVDS_LEVELS_DIR / f for f in os.listdir(BVDS_LEVELS_DIR) if f.endswith('.mat')])
BVDS_FILE_PATHS = file_paths


def create_scg_heatmap(scg_beats_array, title="SCG Beats Heatmap", cmap="gray", figsize=(12, 8)):
    """
    Creates a heatmap visualization of SCG beats.
    
    Parameters:
    -----------
    scg_beats_array : ndarray
        2D array where each row represents a single SCG beat time signal
    title : str, optional
        Title for the plot
    cmap : str, optional
        Colormap to use for the heatmap (default is 'gray' for black and white)
    figsize : tuple, optional
        Figure size (width, height) in inches
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Check if input is a numpy array, if not, convert it
    if not isinstance(scg_beats_array, np.ndarray):
        scg_beats_array = np.array(scg_beats_array)
    
    # Ensure 2D array
    if scg_beats_array.ndim == 1:
        scg_beats_array = scg_beats_array.reshape(1, -1)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap using seaborn with gray colormap for black and white
    heatmap = sns.heatmap(scg_beats_array, cmap=cmap, ax=ax)
    
    # Set labels and title
    ax.set_xlabel('Time Samples')
    ax.set_ylabel('Beat Number')
    ax.set_title(title)
    
    # Add colorbar label
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Amplitude')
    
    plt.tight_layout()
    return fig, ax

def normalize_scg_by_percent_difference(scg_beats, baseline=None):
    """
    Normalizes SCG beats by calculating percent difference from a baseline.
    
    Parameters:
    -----------
    scg_beats : ndarray
        2D array where each row represents a single SCG beat time signal
    baseline : ndarray or str, optional
        Baseline to use for normalization. Can be:
        - None: Use mean of each beat as its own baseline (default)
        - 'global_mean': Use global mean of all beats
        - ndarray: Custom baseline values (must match signal length)
    
    Returns:
    --------
    normalized_beats : ndarray
        Array of same shape as input, with values representing 
        percent difference from baseline
    """
    # Ensure input is numpy array
    if not isinstance(scg_beats, np.ndarray):
        scg_beats = np.array(scg_beats)
    
    # Ensure 2D array
    if scg_beats.ndim == 1:
        scg_beats = scg_beats.reshape(1, -1)
    
    # Calculate baseline based on input parameter
    if baseline is None:
        # Use mean of each beat as its own baseline
        baseline = np.mean(scg_beats, axis=1, keepdims=True)
    elif baseline == 'global_mean':
        # Use global mean of all beats
        baseline = np.mean(scg_beats) * np.ones_like(scg_beats)
    else:
        # Ensure custom baseline is properly shaped
        if not isinstance(baseline, np.ndarray):
            baseline = np.array(baseline)
        if baseline.ndim == 1 and baseline.shape[0] == scg_beats.shape[1]:
            # Broadcast single baseline to all beats
            baseline = np.tile(baseline, (scg_beats.shape[0], 1))
    
    # Calculate percent difference: (value - baseline) / baseline * 100
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    normalized_beats = (scg_beats - baseline) / (np.abs(baseline) + epsilon) * 100
    
    return normalized_beats

def remove_outlier_beats(scg_beats, threshold=3.0, method='zscore'):
    """
    Removes outlier beats based on the percent difference values.
    
    Parameters:
    -----------
    scg_beats : ndarray
        2D array where each row represents a single SCG beat time signal
    threshold : float, optional
        Threshold for determining outliers:
        - If method='zscore': Number of standard deviations from mean to consider as outlier
        - If method='iqr': Multiplier for IQR (values outside Q1-threshold*IQR and Q3+threshold*IQR are outliers)
    method : str, optional
        Method to use for outlier detection ('zscore' or 'iqr')
    
    Returns:
    --------
    filtered_beats : ndarray
        Array with outlier beats removed
    outlier_indices : ndarray
        Indices of the beats that were identified as outliers
    """
    # Ensure input is numpy array and 2D
    if not isinstance(scg_beats, np.ndarray):
        scg_beats = np.array(scg_beats)
    
    if scg_beats.ndim == 1:
        scg_beats = scg_beats.reshape(1, -1)
    
    # Calculate normalized beats to detect outliers
    normalized_beats = normalize_scg_by_percent_difference(scg_beats)
    
    # Calculate mean or median of each beat to simplify outlier detection
    beat_stats = np.mean(np.abs(normalized_beats), axis=1)
    
    # Identify outliers based on chosen method
    if method.lower() == 'zscore':
        # Z-score method
        mean_val = np.mean(beat_stats)
        std_val = np.std(beat_stats)
        z_scores = np.abs((beat_stats - mean_val) / std_val)
        outlier_indices = np.where(z_scores > threshold)[0]
        
    elif method.lower() == 'iqr':
        # IQR method
        q1 = np.percentile(beat_stats, 25)
        q3 = np.percentile(beat_stats, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outlier_indices = np.where((beat_stats < lower_bound) | (beat_stats > upper_bound))[0]
    
    else:
        raise ValueError("Method must be either 'zscore' or 'iqr'")
    
    # Create mask for non-outlier beats
    mask = np.ones(scg_beats.shape[0], dtype=bool)
    mask[outlier_indices] = False
    
    # Return filtered beats and outlier indices
    filtered_beats = scg_beats[mask]
    
    return filtered_beats, outlier_indices

import scipy.io

def get_bvds_level_ranges(bvds_file_path=None, bvds_array=None):
    """
    Find the index ranges for each BVDS level in the array.
    
    Parameters:
    -----------
    bvds_file_path : str or Path, optional
        Path to the BVDS levels .mat file. If provided, bvds_array is ignored.
    bvds_array : ndarray, optional
        Array containing BVDS levels. Used only if bvds_file_path is None.
        
    Returns:
    --------
    list of tuples
        Each tuple contains (level_value, [start_index, end_index])
    """
    # Load BVDS data from file if path is provided
    if bvds_file_path is not None:
        bvds_data_dict = scipy.io.loadmat(bvds_file_path)
        bvds_array = bvds_data_dict['bvds_beats'][::5]
    
    # Ensure we have data to process
    if bvds_array is None:
        raise ValueError("Either bvds_file_path or bvds_array must be provided")
    
    # Flatten the array if it's 2D
    if bvds_array.ndim > 1:
        bvds_data = bvds_array.flatten()
    else:
        bvds_data = bvds_array
    
    # Get unique levels
    unique_levels = np.unique(bvds_data)
    
    # Store the ranges for each level
    level_ranges = []
    
    for level in unique_levels:
        # Find indices where this level occurs
        level_indices = np.where(bvds_data == level)[0]
        
        # Find continuous ranges
        if len(level_indices) > 0:
            ranges = []
            range_start = level_indices[0]
            prev_idx = level_indices[0]
            
            for i in range(1, len(level_indices)):
                # If not continuous, save previous range and start new one
                if level_indices[i] > prev_idx + 1:
                    ranges.append([range_start, prev_idx])
                    range_start = level_indices[i]
                prev_idx = level_indices[i]
            
            # Add the last range
            ranges.append([range_start, level_indices[-1]])
            
            # Add each range to the result
            for range_pair in ranges:
                level_ranges.append((level, range_pair))
    
    # Sort by start index
    level_ranges.sort(key=lambda x: x[1][0])
    
    return level_ranges

from matplotlib.patches import Patch
import numpy as np

def plot_scg_heatmap_with_bvds_levels(file_numbers, bvds_file_path, color_map=cmap_color, downsample_factor=1, 
                                      bvds_labels=['0%', '7%', '14%', '21%', '28%']):
    """
    Plots SCG beats heatmap with colored rectangles showing BVDS level ranges.
    
    Parameters:
    -----------
    file_numbers : list
        List of file numbers to process for the SCG data
    bvds_file_path : str or Path
        Path to the BVDS levels .mat file
    color_map : str, optional
        Colormap for the SCG heatmap
    downsample_factor : int, optional
        Factor to downsample BVDS data (default is 5)
    bvds_labels : list, optional
        Labels for the BVDS levels
    
    Returns:
    --------
    fig : matplotlib Figure
        The figure containing the plot
    """
    # 1. Process and combine SCG beats
    combined_beats = []
    
    print(f"Processing files: {file_numbers}")
    for file_num in file_numbers:
        try:
            # Load data for this file number
            pig_data = HypoDataset(file_num)
            
            # Get SCG beats for this file
            file_scg_beats = pig_data.data[0]['scgBeats']
            
            # Normalize and remove outliers
            norm_beats = normalize_scg_by_percent_difference(file_scg_beats)
            clean_beats, outliers = remove_outlier_beats(norm_beats)
            
            print(f"  File {file_num}: {len(file_scg_beats)} beats, {len(outliers)} outliers removed")
            
            # Add to combined array
            combined_beats.append(clean_beats)
            
        except Exception as e:
            print(f"  Error processing file {file_num}: {str(e)}")
    
    # Stack all beats
    if not combined_beats:
        raise ValueError("No valid SCG data found for the specified files.")
    
    all_pig_beats = np.vstack(combined_beats)
    print(f"Total combined beats: {all_pig_beats.shape[0]}")
    
    # 2. Load BVDS data
    pig_levels = scipy.io.loadmat(bvds_file_path)
    pig_bvds_beats = pig_levels['bvds_beats'][::downsample_factor]
    pig_bvds_rel = pig_levels['rel_bvds_beats'][::downsample_factor]
    
    # 3. Get BVDS level ranges
    bvds_level_ranges = get_bvds_level_ranges(bvds_array=pig_bvds_beats)
    
    # Convert absolute BVDS values to relative for better visualization
    bvds_levels = [0.0, 0.25, 0.5, 0.75, 1.0]  # Map 0, 7, 14, 21, 28 to 0-1 range
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    
    # Create mapping from raw BVDS values to normalized levels
    raw_to_norm = {
        0.0: 0.0,    # Normal
        7.0: 0.25,   # Mild
        14.0: 0.5,   # Moderate
        21.0: 0.75,  # Severe
        28.0: 1.0    # Critical
    }
    
    # 4. Create the figure
    fig = plt.figure(figsize=(14, 10))
    
    # Main heatmap plot
    ax_scg = plt.subplot2grid((5, 1), (0, 0), rowspan=4)
    
    # Create heatmap
    sns.heatmap(all_pig_beats, cmap=color_map, ax=ax_scg)
    
    # Determine pig number from file numbers
    pig_num = int(file_numbers[0])
    ax_scg.set_title(f"Pig {pig_num} SCG Beats with BVDS Levels")
    ax_scg.set_xlabel("Time Samples")
    ax_scg.set_ylabel("Beat Number")
    
    # BVDS levels plot below
    # ax_bvds = plt.subplot2grid((5, 1), (4, 0), rowspan=1, sharex=ax_scg)
    
    # Plot raw BVDS data
    bvds_data = pig_bvds_beats.flatten()
    # ax_bvds.plot(bvds_data, color='black', linewidth=1.5)
    # ax_bvds.set_ylabel("BVDS Level")
    
    # Highlight different level ranges with different colors
    legend_elements = []
    
    for i, (level, (start, end)) in enumerate(bvds_level_ranges):
        # Get the normalized level
        if level in raw_to_norm:
            norm_level = raw_to_norm[level]
            level_idx = bvds_levels.index(norm_level)
            color = colors[level_idx]
            label = bvds_labels[level_idx]
            
            # Highlight the range in both plots
            # ax_bvds.axvspan(start, end, alpha=0.3, color=color)
            
            # Scale the indices to match the heatmap size
            scale_factor = all_pig_beats.shape[0] / len(bvds_data)
            scaled_start = int(start * scale_factor)
            scaled_end = min(int(end * scale_factor), all_pig_beats.shape[0])
            
            if scaled_start < all_pig_beats.shape[0] and scaled_end > 0:
                ax_scg.axhspan(scaled_start, scaled_end, alpha=0.15, color=color)
            
            # Add to legend if not already there
            if label not in [le.get_label() for le in legend_elements]:
                legend_elements.append(Patch(facecolor=color, alpha=0.3, label=label))
    
    # Add legend
    if legend_elements:
        ax_scg.legend(handles=legend_elements, loc='upper right', 
                      bbox_to_anchor=(1.0, -0.08), ncol=len(legend_elements))
    
    plt.tight_layout()
    return fig

def plot_scg_heatmap_range(file_numbers, start_idx, end_idx, cmap=cmap_color, title=None, bvds_label=None):
    """
    Combines SCG beats from multiple files and plots a specified range as a heatmap.
    
    Parameters:
    -----------
    file_numbers : list
        List of file numbers to process for the SCG data
    start_idx : int
        Starting index of the range to plot
    end_idx : int
        Ending index of the range to plot
    cmap : str, optional
        Colormap to use for the heatmap
    title : str, optional
        Custom title for the plot. If None, a default title is generated.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Process and combine SCG beats
    combined_beats = []
    
    print(f"Processing files: {file_numbers}")
    for file_num in file_numbers:
        try:
            # Load data for this file number
            pig_data = HypoDataset(file_num)
            
            # Get SCG beats for this file
            file_scg_beats = pig_data.data[0]['scgBeats']
            
            # Normalize and remove outliers
            norm_beats = normalize_scg_by_percent_difference(file_scg_beats)
            clean_beats, outliers = remove_outlier_beats(norm_beats)
            
            print(f"  File {file_num}: {len(file_scg_beats)} beats, {len(outliers)} outliers removed")
            
            # Add to combined array
            combined_beats.append(clean_beats)
            
        except Exception as e:
            print(f"  Error processing file {file_num}: {str(e)}")
    
    # Stack all beats
    if not combined_beats:
        raise ValueError("No valid SCG data found for the specified files.")
    
    all_beats = np.vstack(combined_beats)
    total_beats = all_beats.shape[0]
    
    # Validate indices
    if start_idx < 0:
        start_idx = 0
    if end_idx > total_beats:
        end_idx = total_beats
    if start_idx >= end_idx:
        raise ValueError(f"Start index ({start_idx}) must be less than end index ({end_idx}).")
    
    # Extract the range of beats to plot
    beats_to_plot = all_beats[start_idx:end_idx]
    
    # Generate default title if none provided
    if title is None:
        pig_num = int(file_numbers[0])
        title = f"Pig {pig_num} SCG Beats (Range: {start_idx} to {end_idx}) (BVDS: {bvds_label})"
    
    # Create the heatmap
    fig, ax = create_scg_heatmap(
        beats_to_plot,
        title=title,
        cmap=cmap
    )
    
    print(f"Plotted beats {start_idx} to {end_idx} (total: {end_idx - start_idx} beats)")
    
    return fig, ax