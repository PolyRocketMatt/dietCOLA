import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_points_vs_x_parallel(df: pd.DataFrame, bins: int = None, output_path: str = None, cell_ids: list = None) -> None:
    """
    Plots the number of unique points by their x_parallel position.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'x_parallel', 'cell_id', 'frame', and 'point_id' columns.
    bins (int): Number of bins for x_parallel.
    output_path (str): Path to save the plot. If None, the plot is shown instead.
    cell_ids (list): List of cell IDs to include in the plot. If None, all cells are included.

    Returns:
    None
    """

    df = df.copy()

    # Filter by cell_ids if provided
    if cell_ids is not None:
        df = df[df['cell_id'].isin(cell_ids)]

    # Drop NaNs in required columns
    df = df.dropna(subset=['x_parallel'])

    # Only look at the first timepoint in each cell (right before cut)
    first_timepoints = df.groupby('cell_id')['frame'].min().reset_index()
    df = df.merge(first_timepoints, on=['cell_id', 'frame'])

    plt.figure(figsize=(6, 5))
    plt.hist(df['x_parallel'], bins=bins, edgecolor='black')
    plt.xlabel('x_parallel (fraction of cut length)')
    plt.ylabel('Number of points')
    plt.title('Number of points by x_parallel', fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')

    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

    return


def plot_points_vs_x_perpendicular(df: pd.DataFrame, abs: bool = True, bins: int = None, output_path: str = None, cell_ids: list = None) -> None:
    """
    Plots the number of unique points by their x_perpendicular position.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'x_perpendicular', 'cell_id', 'frame', and 'point_id' columns.
    bins (int): Number of bins for x_perpendicular.
    abs (bool): If True, plot using absolute values of x_perpendicular.
    output_path (str): Path to save the plot. If None, the plot is shown instead.
    cell_ids (list): List of cell IDs to include in the plot. If None, all cells are included.

    Returns:
    None
    """

    df = df.copy()

    # Filter by cell_ids if provided
    if cell_ids is not None:
        df = df[df['cell_id'].isin(cell_ids)]

    # Drop NaNs in required columns
    df = df.dropna(subset=['x_perpendicular'])

    # Only look at the first timepoint in each cell (right before cut)
    first_timepoints = df.groupby('cell_id')['frame'].min().reset_index()
    df = df.merge(first_timepoints, on=['cell_id', 'frame'])

    plt.figure(figsize=(9.5, 5))

    if abs:
        df['x_perpendicular'] = df['x_perpendicular'].abs()

    plt.title('Number of points by |x_perpendicular|' if abs else 'Number of points by x_perpendicular', fontweight='bold')
    plt.xlabel('|x_perpendicular|' if abs else 'x_perpendicular')

    plt.hist(df['x_perpendicular'], bins=bins, edgecolor='black')
    plt.ylabel('Number of points')
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')

    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

    return


def animate_points_vs_x_perpendicular(df: pd.DataFrame, abs: bool = True, bin_length: float = 5.0, 
                                       start_offset: int = -1, end_offset: int = None,
                                       output_path: str = None, cell_ids: list = None, fps: int = 5):
    """
    Creates an animation showing the distribution of points by x_perpendicular over time offsets.

    Uses the existing 'frame_rel' column (relative frame to cut) instead of computing offsets from 'cut'.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'x_perpendicular', 'cell_id', 'frame_rel', 'cut', and 'point_id' columns.
    abs (bool): If True, plot using absolute values of x_perpendicular. Default is True.
    bin_length (float): Width of histogram bins in pixels. Default is 5.0.
    start_offset (int): Starting time offset relative to cut. Default is -1.
    end_offset (int): Ending time offset relative to cut. If None, uses max offset in data.
    output_path (str): Path to save the animation as GIF. If None, returns HTML object for display.
    cell_ids (list): List of cell IDs to include. If None, all cells are included.
    fps (int): Frames per second for the animation. Default is 5.

    Returns:
    HTML or None: IPython HTML object containing the animation (or None if saved to file)
    """
    from IPython.display import HTML
    from matplotlib.animation import PillowWriter
    import matplotlib.animation as animation

    df = df.copy()

    # Filter by cell_ids if provided
    if cell_ids is not None:
        df = df[df['cell_id'].isin(cell_ids)]

    # Drop NaNs in required columns (use frame_rel already present)
    df = df.dropna(subset=['x_perpendicular', 'frame_rel'])

    # Exclude actual cut frames
    if 'cut' in df.columns:
        df = df[df['cut'] == False]

    # Determine offset range using frame_rel directly
    if end_offset is None:
        end_offset = int(df['frame_rel'].max())

    offsets = np.arange(start_offset, end_offset + 1)

    # Set up bins based on abs parameter
    if abs:
        max_val = np.ceil(df['x_perpendicular'].abs().max()) if not df['x_perpendicular'].abs().empty else 0.0
        bins = np.arange(0.0, max_val + bin_length, bin_length)
    else:
        min_val = np.floor(df['x_perpendicular'].min()) if not df['x_perpendicular'].empty else 0.0
        max_val = np.ceil(df['x_perpendicular'].max()) if not df['x_perpendicular'].empty else 0.0
        bins = np.arange(min_val, max_val + bin_length, bin_length)

    # Compute max y for fixed axis
    max_y = 0
    for offset in offsets:
        tdf = df[df['frame_rel'] == offset]
        data = tdf['x_perpendicular'].abs() if abs else tdf['x_perpendicular']
        counts, _ = np.histogram(data, bins=bins)
        if len(counts) > 0 and counts.max() > max_y:
            max_y = counts.max()

    max_y = max_y * 1.05  # add 5% margin
    max_y = max(1, max_y)  # avoid zero

    # Create figure and animation
    fig, ax = plt.subplots(figsize=(9.5, 5))

    def update(i):
        offset = offsets[i]
        tdf = df[df['frame_rel'] == offset]
        ax.cla()

        data = tdf['x_perpendicular'].abs() if abs else tdf['x_perpendicular']
        ax.hist(data, bins=bins, edgecolor='black')

        xlabel = '|x_perpendicular| (pixels)' if abs else 'x_perpendicular (pixels)'
        title = f'Number of points by |x_perpendicular| at frame_rel {offset}' if abs else f'Number of points by x_perpendicular at frame_rel {offset}'

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Number of points')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.text(0.99, 0.95, f'frame_rel={offset} total points={tdf["point_id"].nunique()}',
                transform=ax.transAxes, ha='right', va='top')

        if abs:
            ax.set_xlim(-bin_length, bins.max())
        else:
            ax.set_xlim(bins.min() - bin_length/2, bins.max() + bin_length/2)

        ax.set_ylim(0, max_y + 1)

        # Add vertical red dashed lines for means
        if not abs:
            less = tdf[tdf['x_perpendicular'] < 0]['x_perpendicular']
            more = tdf[tdf['x_perpendicular'] > 0]['x_perpendicular']
            if not less.empty:
                mean_less = np.mean(less)
                ax.axvline(mean_less, color='red', linestyle='--', linewidth=2, label='Mean')
            if not more.empty:
                mean_more = np.mean(more)
                ax.axvline(mean_more, color='red', linestyle='--', linewidth=2,)
        else:
            # For abs, just one mean
            if not data.empty:
                mean_abs = np.mean(data)
                ax.axvline(mean_abs, color='red', linestyle='--', linewidth=2, label='Mean')

        # Only show legend on first frame if any mean lines were drawn
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

        return ax.patches

    anim = animation.FuncAnimation(fig, update, frames=len(offsets), repeat=False)

    # Save or return
    if output_path:
        # Determine file extension
        ext = output_path.lower().split('.')[-1]
        if ext == 'gif':
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer, dpi=300)
        else:
            # Default to mp4 if not gif
            if not output_path.lower().endswith('.mp4'):
                output_path = output_path + '.mp4'
            writer = animation.FFMpegWriter(fps=fps)
            anim.save(output_path, writer=writer, dpi=300)
        plt.close(fig)
        return None
    else:
        html = HTML(anim.to_jshtml(fps=fps))
        plt.close(fig)
        return html


def plot_points_vs_x_perpendicular_overlapping(df: pd.DataFrame, abs: bool = True, bin_length: float = 5.0, 
                                                start_offset: int = -1, end_offset: int = None,
                                                output_path: str = None, cell_ids: list = None,
                                                colormap: str = 'viridis', alpha: float = 0.7):
    """
    Creates a static plot showing overlapping distribution lines of x_perpendicular over time offsets.
    Adds vertical dashed mean lines:
      - If abs=True → one mean per offset (of |x_perpendicular|)
      - If abs=False → two means per offset (for x_perpendicular < 0 and > 0)
    Includes a legend entry for the mean indicator.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    df = df.copy()

    # Filter by cell_ids if provided
    if cell_ids is not None:
        df = df[df['cell_id'].isin(cell_ids)]

    # Drop NaNs in required columns
    df = df.dropna(subset=['x_perpendicular', 'frame_rel'])

    # Exclude actual cut frames
    if 'cut' in df.columns:
        df = df[df['cut'] == False]

    # Determine offset range
    if end_offset is None:
        end_offset = int(df['frame_rel'].max())

    offsets = np.arange(start_offset, end_offset + 1)

    # Set up bins based on abs parameter, centered around zero
    if abs:
        max_val = np.ceil(df['x_perpendicular'].abs().max()) if not df['x_perpendicular'].abs().empty else 0.0
        bins = np.arange(0.0, max_val + bin_length, bin_length)
    else:
        max_abs = np.ceil(np.max(np.abs(df['x_perpendicular']))) if not df['x_perpendicular'].empty else 0.0
        # Center bins so that the central bin is symmetric around zero
        half_bin = bin_length / 2.0
        bins = np.arange(-max_abs - half_bin, max_abs + half_bin + bin_length, bin_length)

    # Set up colormap
    cmap = cm.get_cmap(colormap)
    norm = Normalize(vmin=start_offset, vmax=end_offset)

    # Create figure
    plt.figure(figsize=(12, 5))
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot each offset as a line + means
    for offset in offsets:
        tdf = df[df['frame_rel'] == offset]
        if len(tdf) == 0:
            continue

        color = cmap(norm(offset))

        if abs:
            data = tdf['x_perpendicular'].abs()
        else:
            data = tdf['x_perpendicular']

        counts, _ = np.histogram(data, bins=bins)

        # Plot histogram line
        plt.plot(bin_centers, counts, color=color, alpha=alpha, linewidth=1.5)

        # Compute and plot mean(s)
        if abs:
            mean_val = np.mean(data)
            plt.axvline(mean_val, color=color, linestyle='--', alpha=0.8, linewidth=1.0)
        else:
            neg_vals = data[data < 0]
            pos_vals = data[data > 0]
            if len(neg_vals) > 0:
                neg_mean = np.mean(neg_vals)
                plt.axvline(neg_mean, color=color, linestyle='--', alpha=0.8, linewidth=1.0)
            if len(pos_vals) > 0:
                pos_mean = np.mean(pos_vals)
                plt.axvline(pos_mean, color=color, linestyle='--', alpha=0.8, linewidth=1.0)

    # Labels and title
    xlabel = '|x_perpendicular| (pixels)' if abs else 'x_perpendicular (pixels)'
    title = 'Distribution of |x_perpendicular| over time' if abs else 'Distribution of x_perpendicular over time'
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Number of points', fontsize=12)
    plt.title(title, fontweight='bold', fontsize=14)
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Set x-axis limits
    if abs:
        plt.xlim(-bin_length, bins.max())
    else:
        plt.xlim(bins.min() - bin_length/2, bins.max() + bin_length/2)

    # Add colorbar for time progression
    dummy_scatter = plt.scatter([], [], c=[], cmap=cmap, vmin=start_offset, vmax=end_offset)
    cbar = plt.colorbar(dummy_scatter, label='frame_rel')

    plotted_offsets = [offset for offset in offsets if len(df[df['frame_rel'] == offset]) > 0]
    cbar.set_ticks(plotted_offsets)
    cbar.set_ticklabels([f'{int(t)}' for t in plotted_offsets])

    # Add legend showing what the dashed lines mean
    mean_line = Line2D([0], [0], color='black', linestyle='--', linewidth=1.0, label='Mean')
    plt.legend(handles=[mean_line], loc='upper right')

    plt.tight_layout()

    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

    return


def plot_points_vs_frame(df: pd.DataFrame, cell_ids: list = None, output_path: str = None) -> None:
    """
    Plots the number of unique points over frames.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'frame_rel', 'cell_id', and 'point_id' columns.
    cell_ids (list): List of cell IDs to include in the plot. If None, all cells are included.
    output_path (str): Path to save the plot. If None, the plot is shown instead.

    Returns:
    None
    """

    df = df.copy()

    # Filter by cell_ids if provided
    if cell_ids is not None:
        df = df[df['cell_id'].isin(cell_ids)]

    # Drop NaNs in required columns
    df = df.dropna(subset=['frame_rel'])

    plt.figure(figsize=(6, 5))
    bins = np.arange(df['frame_rel'].min() - 0.5, df['frame_rel'].max() + 1.5, 1)
    plt.hist(df['frame_rel'], bins=bins, edgecolor='black')
    plt.xlabel('Frame (relative to cut)')
    plt.ylabel('Number of unique points')
    plt.title('Number of unique points over time', fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.xticks(np.arange(df['frame_rel'].min(), df['frame_rel'].max() + 1, 1))

    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

    return


def plot_velocities_vs_x_perpendicular(df: pd.DataFrame, abs: bool = True, frame_rel: int = None, cell_ids: list = None, output_path: str = None) -> None:
    """
    Plots the velocities of points by their x_perpendicular position at a specific frame_rel.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'x_perpendicular', 'velocity', and 'frame_rel' columns.
    abs (bool): If True, plot using absolute values of x_perpendicular.
    frame_rel (int): The frame_rel to filter the data. If None, uses all frames.
    cell_ids (list): List of cell IDs to include in the plot. If None, all cells are included.
    output_path (str): Path to save the plot. If None, the plot is shown instead.

    Returns:
    None
    """
    
    df = df.copy()

    # Filter by frame_rel
    if frame_rel is not None:
        df = df[df['frame_rel'] == frame_rel]

    # Filter by cell_ids if provided
    if cell_ids is not None:
        df = df[df['cell_id'].isin(cell_ids)]

    # Drop NaNs in required columns
    df = df.dropna(subset=['x_perpendicular', 'velocity_cut'])

    plt.figure(figsize=(9.5, 5))

    if abs:
        df['x_perpendicular'] = df['x_perpendicular'].abs()

    plt.scatter(df['x_perpendicular'], df['velocity_cut'], alpha=0.6)
    plt.xlabel('|x_perpendicular|' if abs else 'x_perpendicular')
    plt.ylabel('Velocity (pixels/frame)')
    plt.title(f'Point Velocities by {"|x_perpendicular|" if abs else "x_perpendicular"} at frame_rel {frame_rel}', fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')

    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

    return


def plot_velocity_cut_vs_time(
        df: pd.DataFrame,
        cell_ids: list = None,
        end_offset: int = None,
        output_path: str = None,
    ) -> None:
    """
    Plots the velocities of points over time (frame_rel) as lines per point, and overlays the average velocity.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'frame_rel', 'velocity_cut', and 'cell_id' columns.
    cell_ids (list): List of cell IDs to include in the plot. If None, all cells are included.
    end_offset (int): Ending time offset relative to cut. If None, uses max offset in data.
    output_path (str): Path to save the plot. If None, the plot is shown instead.
    
    Returns:
    None
    """

    df = df.copy()

    # Filter by cell_ids if provided
    if cell_ids is not None:
        df = df[df['cell_id'].isin(cell_ids)]

    if end_offset is not None:
        df = df[df['frame_rel'] <= end_offset]

    # Drop NaNs in required columns
    df = df.dropna(subset=['frame_rel', 'velocity_cut', 'point_id'])

    plt.figure(figsize=(9.5, 5))

    # Plot each point's velocity as a line
    for pid, group in df.groupby('point_id'):
        plt.plot(group['frame_rel'], group['velocity_cut'], color='gray', alpha=0.05, linewidth=0.5)

    # Plot the average velocity as a bold line
    avg = df.groupby('frame_rel')['velocity_cut'].mean()
    plt.plot(avg.index, avg.values, color='red', linewidth=2.5, label='Average velocity')

    plt.xlabel('Frame (relative to cut)')
    plt.ylabel('Velocity (pixels/frame)')
    plt.xticks(np.arange(int(df['frame_rel'].min()), int(df['frame_rel'].max()) + 1, 1))
    plt.title('Point Velocities (relative to cut) over Time', fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()

    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

    return


def plot_v_perpendicular_vs_time(
        df: pd.DataFrame,
        cell_ids: list = None,
        output_path: str = None,
        abs: bool = True,
        end_offset: int = None,
        cell_type: str = None,
    ) -> None:
    """
    Plots the perpendicular velocities of points over time (frame_rel) as lines per point,
    and overlays the average perpendicular velocity.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'frame_rel', 'v_perpendicular', and 'cell_id' columns.
    cell_ids (list): List of cell IDs to include in the plot. If None, all cells are included.
    output_path (str): Path to save the plot. If None, the plot is shown instead.
    abs (bool): If True, compute the average of the absolute perpendicular velocities.
    end_offset (int): Ending time offset relative to cut. If None, uses max offset in data.
    cell_type (str): OPTIONAL Cell type to include in the title.

    Returns:
    None
    """
    df = df.copy()

    # Filter by cell_ids if provided
    if cell_ids is not None:
        df = df[df['cell_id'].isin(cell_ids)]

    # Drop NaNs in required columns
    df = df.dropna(subset=['frame_rel', 'v_perpendicular', 'point_id'])

    if end_offset is not None:
        df = df[df['frame_rel'] <= end_offset]

    if abs:
        df['v_perpendicular'] = df['v_perpendicular'].abs()

    plt.figure(figsize=(9.5, 5))

    # Plot each point's perpendicular velocity as a faint line
    for pid, group in df.groupby('point_id'):
        plt.plot(group['frame_rel'], group['v_perpendicular'], color='gray', alpha=0.05, linewidth=0.5)

    # Compute average across frames
    avg = df.groupby('frame_rel')['v_perpendicular'].mean()

    # Label depends on abs setting
    label = 'Average |v_perpendicular|' if abs else 'Average v_perpendicular'

    # Plot the average velocity
    plt.plot(avg.index, avg.values, color='blue', linewidth=2.5, label=label)

    plt.xlabel('Frame (relative to cut)')
    plt.ylabel('Perpendicular Velocity (pixels/frame)')
    plt.xticks(np.arange(int(df['frame_rel'].min()), int(df['frame_rel'].max()) + 1, 1))
    plt.title('Point Perpendicular Velocities over Time' + ('' if not cell_type else f'Cell Type: {cell_type}'), fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()

    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

    return


def compare_v_perpendicular_vs_time(
        *datasets,
        cell_ids: list = None,
        output_path: str = None,
        abs: bool = True,
        end_offset: int = None,
    ) -> None:
    """
    Compares the perpendicular velocities of points over time (frame_rel) between multiple datasets.
    Plots the average perpendicular velocity from all datasets on the same graph.

    Parameters:
    *datasets : tuple of (df, label, color)
        Variable number of datasets, each as a tuple of (dataframe, label, color)
    cell_ids (list): List of cell IDs to include in the plot. If None, all cells are included.
    output_path (str): Path to save the plot. If None, the plot is shown instead.
    abs (bool): If True, compute the average of the absolute perpendicular velocities.
    end_offset (int): Ending time offset relative to cut. If None, uses max offset in data.
    
    Returns:
    None
    """
    if len(datasets) == 0:
        raise ValueError("At least one dataset must be provided")
    
    plt.figure(figsize=(9.5, 5))
    
    all_min_frames = []
    all_max_frames = []
    
    for df, label, color in datasets:
        df = df.copy()
        
        # Filter by cell_ids if provided
        if cell_ids is not None:
            df = df[df['cell_id'].isin(cell_ids)]
        
        # Drop NaNs in required columns
        df = df.dropna(subset=['frame_rel', 'v_perpendicular'])
        
        if end_offset is not None:
            df = df[df['frame_rel'] <= end_offset]
        
        if abs:
            df['v_perpendicular'] = df['v_perpendicular_signed']
        
        # Compute average across frames
        avg = df.groupby('frame_rel')['v_perpendicular'].mean()
        
        if len(avg) > 0:
            all_min_frames.append(avg.index.min())
            all_max_frames.append(avg.index.max())
        
        # Label depends on abs setting
        plot_label = label + ' Average |v_perpendicular|' if abs else label + ' Average v_perpendicular'
        
        # Plot the average velocity
        plt.plot(avg.index, avg.values, color=color, linewidth=2.5, label=plot_label)
    
    plt.xlabel('Frame (relative to cut)')
    plt.ylabel('Perpendicular Velocity (pixels/frame)')
    
    if all_min_frames and all_max_frames:
        plt.xticks(np.arange(int(min(all_min_frames)), int(max(all_max_frames)) + 1, 1))
    
    plt.title('Comparison of Point Perpendicular Velocities over Time', fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()

    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

    return


def compare_mean_displacement_velocity(
        *datasets,
        cell_ids: list = None,
        output_path: str = None,
        abs: bool = True,
        end_offset: int = None,
    ) -> None:
    """
    Compares velocities calculated from the shift in mean x_perpendicular position over time.
    This provides a measure of bulk tissue movement rather than individual point velocities.

    Parameters:
    *datasets : tuple of (df, label, color)
        Variable number of datasets, each as a tuple of (dataframe, label, color)
    cell_ids (list): List of cell IDs to include in the plot. If None, all cells are included.
    output_path (str): Path to save the plot. If None, the plot is shown instead.
    abs (bool): If True, compute the average of the absolute x_perpendicular values before calculating velocity.
    end_offset (int): Ending time offset relative to cut. If None, uses max offset in data.
    
    Returns:
    None
    """
    if len(datasets) == 0:
        raise ValueError("At least one dataset must be provided")
    
    plt.figure(figsize=(9.5, 5))
    
    all_min_frames = []
    all_max_frames = []
    
    for df, label, color in datasets:
        df = df.copy()

        df = df[df['cut'] == False]
        
        # Filter by cell_ids if provided
        if cell_ids is not None:
            df = df[df['cell_id'].isin(cell_ids)]
        
        # Drop NaNs in required columns
        df = df.dropna(subset=['frame_rel', 'x_perpendicular'])
        
        if end_offset is not None:
            df = df[df['frame_rel'] <= end_offset]
        
        # Calculate mean x_perpendicular per frame_rel
        if abs:
            df['x_perpendicular'] = df['x_perpendicular'].abs()
            
        mean_x_perp = df.groupby('frame_rel')['x_perpendicular'].mean()
        
        # Calculate velocity as change in mean position
        # velocity[t] = mean_x_perp[t+1] - mean_x_perp[t]
        velocity = mean_x_perp.diff()
        
        # Remove NaN values (first frame has no velocity)
        velocity = velocity.dropna()
        
        if len(velocity) > 0:
            all_min_frames.append(velocity.index.min())
            all_max_frames.append(velocity.index.max())
        
        # Label depends on abs setting
        if abs:
            plot_label = f'{label} Mean Displacement Velocity (|x_perp|)'
        else:
            plot_label = f'{label} Mean Displacement Velocity'
        
        # Plot the velocity
        plt.plot(velocity.index, velocity.values, color=color, linewidth=2.5, label=plot_label, marker='o', markersize=4)
    
    plt.xlabel('Frame (relative to cut)')
    plt.ylabel('Mean Displacement Velocity (pixels/frame)')
    
    if all_min_frames and all_max_frames:
        plt.xticks(np.arange(int(min(all_min_frames)), int(max(all_max_frames)) + 1, 1))
    
    plt.title('Comparison of Mean Displacement Velocities over Time', fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)

    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

    return


def plot_point_position_mse(
        df: pd.DataFrame,
        cell_ids: list = None,
        output_path: str = None,
        end_offset: int = None,
    ) -> None:
    """
    Plots the Mean Squared Error (MSE) of point positions over time (frame_rel).

    Parameters:
    df (pd.DataFrame): DataFrame containing 'frame_rel', 'mse', and 'cell_id' columns.
    cell_ids (list): List of cell IDs to include in the plot. If None, all cells are included.
    output_path (str): Path to save the plot. If None, the plot is shown instead.

    Returns:
    None
    """

    df = df.copy()

    # Filter by cell_ids if provided
    if cell_ids is not None:
        df = df[df['cell_id'].isin(cell_ids)]

    # Drop NaNs in required columns
    df = df.dropna(subset=['frame_rel', 'se'])

    if end_offset is not None:
        df = df[(df['frame_rel'] >= -1) & (df['frame_rel'] <= end_offset)]

    mse_per_frame_rel = df.groupby('frame_rel')['se'].mean()

    plt.figure(figsize=(6,5))
    plt.plot(mse_per_frame_rel.index, mse_per_frame_rel.values, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Frame relative to cut')
    plt.ylabel('Mean Squared Error (pixels²)')
    plt.ylim(bottom=0)
    plt.title('PIV Point Position Error Over Time', fontweight='bold')
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

    return


