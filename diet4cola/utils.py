import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_curve(data: np.array,
               time: np.ndarray,
               title: str = 'Curve'):
    if time is None:
        time = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(time, data, color='blue', linewidth=2, label=f'{title}', zorder=1)
    ax.set_title(title)

    ax.legend(loc='upper right', fontsize=8)
    plt.show()

def plot_2d_array(data: np.ndarray, 
                  title: str = 'Noise',
                  cmap: str = 'gray',
                  min: float = 0,
                  max: float = 1) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(data, cmap=cmap, vmin=min, vmax=max)
    ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)  # pass the AxesImage, not the array
    cbar.set_label('Value', rotation=270, labelpad=12)

    plt.show()

def plot_2d_array_comparison(left: np.ndarray,
                             right: np.ndarray, 
                             left_title: str = 'Left', 
                             right_title: str = 'Right',
                             cmap: str = 'gray',
                             min: float = 0,
                             max: float = 1) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Left image
    im_a = axes[0].imshow(left, cmap=cmap, vmin=min, vmax=max)
    axes[0].set_title(left_title)
    divider = make_axes_locatable(axes[0])
    cax_a = divider.append_axes("right", size="5%", pad=0.05)
    cbar_a = fig.colorbar(im_a, cax=cax_a)
    cbar_a.set_label('Value', rotation=270, labelpad=12)

    # Right image
    im_b = axes[1].imshow(right, cmap=cmap, vmin=min, vmax=max)
    axes[1].set_title(right_title)
    divider = make_axes_locatable(axes[1])
    cax_b = divider.append_axes("right", size="5%", pad=0.05)
    cbar_b = fig.colorbar(im_b, cax=cax_b)
    cbar_b.set_label('Value', rotation=270, labelpad=12)

    plt.tight_layout()
    plt.show()

def plot_synthetic_cut(data: np.ndarray,
                       cut_center: tuple[int, int],
                       cut_origin: tuple[int, int],
                       cut_destination: tuple[int, int],
                       cmap: str = 'gray',
                       min: float = 0,
                       max: float = 1) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(data, cmap=cmap, vmin=min, vmax=max)
    ax.set_title('Synthetic Cut')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)  # pass the AxesImage, not the array
    cbar.set_label('Value', rotation=270, labelpad=12)

    ax.plot(
        [cut_origin[0], cut_destination[0]], 
        [cut_origin[1], cut_destination[1]], 
        color='green', linewidth=2, label='Cut Line', zorder=1
    )
    ax.scatter(cut_center[0], cut_center[1], color='red', s=20, label='Cut Center', zorder=2)
    ax.legend(loc='upper right', fontsize=8)
    
    plt.show()

def plot_sdf(data: np.ndarray,
             origin: tuple[int, int],
             destination: tuple[int, int],
             title: str = 'SDF',
             cmap: str = 'RdBu_r',
             legend: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    img = ax.imshow(data, cmap=cmap, origin='upper')
    ax.set_title(title)
    
    ax.plot(
        [origin[0], destination[0]], 
        [origin[1], destination[1]], 
        color='green', linewidth=2, label='CoLA Cut', zorder=1
    )
    ax.scatter(*origin, color='red', s=20, zorder=2)
    ax.scatter(*destination, color='red', s=20, zorder=2)
    ax.contour(data, levels=[0], colors='yellow', linewidths=1.5)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label('Distance (Euclidean)', rotation=270, labelpad=12)
    
    if legend:
        ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_velocity_field(velocity: np.ndarray,
                        sdf_base: np.ndarray,
                        origin: tuple[int, int],
                        destination: tuple[int, int],
                        title: str = 'Velocity Field',
                        cmap: str = 'viridis'):
    cols = 2

    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    axes = np.array(axes).reshape(1, cols)

    img_a = velocity
    img_a_mappable = axes[0, 0].imshow(img_a, cmap=cmap)
    axes[0, 0].set_title(f'{title}')
    axes[0, 0].plot(
        [origin[0], destination[0]], 
        [origin[1], destination[1]], 
        color='green', linewidth=2, label='CoLA Cut', zorder=1
    )
    axes[0, 0].scatter(*origin, color='red', s=20, zorder=2)
    axes[0, 0].scatter(*destination, color='red', s=20, zorder=2)

    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(img_a_mappable, cax=cax)
    cbar.set_label('Distance (Euclidean)', rotation=270, labelpad=12)
    axes[0, 0].legend(loc='upper right', fontsize=8)

    img_b = sdf_base
    img_b_mappable = axes[0, 1].imshow(img_b, cmap='RdBu_r')
    axes[0, 1].set_title(f'SDF Base')
    axes[0, 1].plot(
        [origin[0], destination[0]], 
        [origin[1], destination[1]], 
        color='green', linewidth=2, label='CoLA Cut', zorder=1
    )
    axes[0, 1].scatter(*origin, color='red', s=20, zorder=2)
    axes[0, 1].scatter(*destination, color='red', s=20, zorder=2)
    axes[0, 1].contour(sdf_base, levels=[0], colors='yellow', linewidths=1.5)

    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(img_b_mappable, cax=cax)
    cbar.set_label('Distance (Euclidean)', rotation=270, labelpad=12)
    axes[0, 1].legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()

def plot_gradient_field(dx: np.ndarray,
                        dy: np.ndarray, 
                        step: int = 1, 
                        normalize: bool = True,
                        cmap: str = 'viridis',
                        title: str = 'Gradient Map'):
    if dx.shape != dy.shape:
        raise ValueError(f'Gradient components dx and dy must have same shape (dx: {dx.shape}, dy: {dy.shape})')

    ny, nx = dx.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))

    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    magnitude[magnitude == 0] = 1  # avoid division by zero
    dx_vis = dx.copy()
    dy_vis = dy.copy()

    if normalize:
        dx_vis = dx_vis / magnitude
        dy_vis = dy_vis / magnitude

    # Slice data for visualization
    x_ = x[::step, ::step]
    y_ = y[::step, ::step]
    dx_ = dx_vis[::step, ::step]
    dy_ = dy_vis[::step, ::step]
    mag_ = magnitude[::step, ::step]

    # Create the quiver plot with color mapping
    plt.figure(figsize=(6, 6))
    Q = plt.quiver(
        x_, y_,
        dx_, dy_,
        mag_,                  # color each arrow by magnitude
        angles='xy',
        scale_units='xy',
        scale=0.1,
        cmap=cmap
    )
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.title(title)
    plt.colorbar(Q, label='Gradient Magnitude')
    plt.show()

def animate_curve(data: np.ndarray,
                  interval: int = 200,
                  title: str = 'Animation') -> FuncAnimation:
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a 2D NumPy array")
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got {data.ndim}D")
    if data.size == 0:
        raise ValueError("data cannot be empty")
    
    num_frames = data.shape[0]
    fig, ax = plt.subplots()
    img = ax.plot(data, color='blue', linewidth=2, label=f'{title}', zorder=1)
    ax.axis('off')  
    ax.set_title(f'{title}')

    def update(frame):
        img.set_data(data[frame])
        return [img]

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)
    plt.close(anim._fig)
    return anim

def animate_2d_data(data: np.ndarray, 
                    origin: tuple[int, int],
                    destination: tuple[int, int],
                    interval: int = 200, 
                    cmap: str = 'viridis',
                    title: str = 'Animation') -> FuncAnimation:
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a 3D NumPy array")
    if data.ndim != 3:
        raise ValueError(f"data must be 3D, got {data.ndim}D")
    if data.size == 0:
        raise ValueError("data cannot be empty")
    
    num_frames = data.shape[0]

    # Min/max for consistent color scaling
    vmin, vmax = data.min(), data.max()

    fig, ax = plt.subplots()
    img = ax.imshow(data[0], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')  
    ax.set_title(f'{title}')

    if origin is not None and destination is not None:
        ax.plot(
            [origin[0], destination[0]], 
            [origin[1], destination[1]], 
            color='green', linewidth=2, label='CoLA Cut', zorder=1
        )

        ax.scatter(*origin, color='red', s=20, zorder=2)
        ax.scatter(*destination, color='red', s=20, zorder=2)

    cbar = fig.colorbar(img, ax=ax)

    def update(frame):
        img.set_data(data[frame])
        return [img]

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)
    plt.close(anim._fig)
    return anim

def animate_gradient_field(dx: np.ndarray,
                           dy: np.ndarray, 
                           step: int = 1, 
                           normalize: bool = True,
                           interval: int = 200, 
                           cmap: str = 'viridis',
                           title: str = 'Gradient Map'):
    if not isinstance(dx, np.ndarray) or not isinstance(dy, np.ndarray):
        raise TypeError("dx and dy must be NumPy arrays")
    if dx.ndim != 3 or dy.ndim != 3:
        raise ValueError(f"dx and dy must be 3D arrays (frames, height, width). Got {dx.ndim}D and {dy.ndim}D.")
    if dx.shape != dy.shape:
        raise ValueError(f"dx and dy must have the same shape (dx: {dx.shape}, dy: {dy.shape})")

    num_frames, ny, nx = dx.shape

    # Helper method to obtain proper gradient directions
    def construct_frame(dx_i: np.ndarray,
                        dy_i: np.ndarray,
                        normalize: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
        magnitude = np.sqrt(dx_i**2 + dy_i**2)
        magnitude[magnitude == 0] = 1
        dx_vis, dy_vis = dx_i.copy(), dy_i.copy()
        if normalize:
            dx_vis /= magnitude
            dy_vis /= magnitude
        return dx_vis, dy_vis, np.sqrt(dx_i**2 + dy_i**2)
    
    # Set up visualization grid
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    dx_vis, dy_vis, mag = construct_frame(dx[0], dy[0], normalize)

    # Slice data for visualization
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    dx_vis, dy_vis, mag = construct_frame(dx[0], dy[0])

    x_ = x[::step, ::step]
    y_ = y[::step, ::step]
    dx_ = dx_vis[::step, ::step]
    dy_ = dy_vis[::step, ::step]
    mag_ = mag[::step, ::step]

    # Create the quiver plot with color mapping
    fig, ax = plt.subplots(figsize=(6, 6))
    Q = ax.quiver(x_, 
                  y_, 
                  dx_, 
                  dy_, 
                  mag_, 
                  angles='xy', 
                  scale_units='xy', 
                  scale=0.1, 
                  cmap=cmap
    )
    ax.set_title(title)
    ax.invert_yaxis()
    ax.axis('equal')
    ax.axis('off')
    cbar = fig.colorbar(Q, ax=ax, label='Gradient magnitude')

    # Update function
    def update(frame):
        dx_vis, dy_vis, mag = construct_frame(dx[frame], dy[frame])
        dx_ = dx_vis[::step, ::step]
        dy_ = dy_vis[::step, ::step]
        mag_ = mag[::step, ::step]
        Q.set_UVC(dx_, dy_, mag_)  # update quiver directions & colors
        ax.set_title(f'{title}')
        return Q,

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)
    plt.close(fig)
    return anim

def save_2d_array(data: np.ndarray, filename: str, cmap: str = 'gray') -> None:
    plt.imshow(data, cmap=cmap, aspect='equal', interpolation='nearest')
    plt.axis('off')
    plt.margins(0)
    plt.gca().set_position([0, 0, 1, 1])  # fill entire figure
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_array(data: np.ndarray, filename: str) -> None:
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a NumPy ndarray.")
    np.save(filename, data)

def load_array(filename: str) -> np.ndarray:
    data = np.load(filename, allow_pickle=False)
    return data