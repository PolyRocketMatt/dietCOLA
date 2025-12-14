import torch
import torch.nn as nn

import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

from diet4cola.model.models import UNet, ResidualUNet

DEVICE = 'cuda'

def load_unet_model(path: str) -> nn.Module:
    model = UNet(base_ch=32, in_ch=2, out_ch=2)
    model.to(DEVICE)
    model.load_state_dict(torch.load(f'{path}', map_location=torch.device(DEVICE)))
    model.eval()

    return model

def load_unet_residual_model(path: str) -> nn.Module:
    model = ResidualUNet(base_ch=32, in_ch=2, out_ch=2)
    model.to(DEVICE)
    model.load_state_dict(torch.load(f'{path}', map_location=torch.device(DEVICE)))
    model.eval()

    return model

def get_motion_fields(frames: np.ndarray, model: nn.Module) -> np.ndarray:
    if frames.ndim != 3:
        raise ValueError("Expected three dimensions <num_frames, 512, 512>")
    if frames.shape[1] != 512 or frames.shape[2] != 512:
        raise ValueError("Expected shape of frame to be 512x512")
    
    num_frames = frames.shape[0]
    model_inputs = []
    
    for i in range(num_frames - 1):
        frame_a = torch.tensor(frames[i], dtype=torch.float32)
        frame_b = torch.tensor(frames[i + 1], dtype=torch.float32)
        input = torch.stack((frame_a, frame_b)).unsqueeze(0)
        input.to(DEVICE)
        model_inputs.append(input)

    outputs = []
    with torch.no_grad():
        for input in tqdm(model_inputs, 'Processing...'):
            output = model(input.to(DEVICE)).cpu().detach().numpy().squeeze().squeeze()

            velocity_x = output[0]
            velocity_y = output[1]

            outputs.append((velocity_x, velocity_y))

    return np.stack(outputs)

def scale_frames(frames: np.ndarray) -> np.ndarray:
    frame_shape = frames.shape[1]
    factor_h = 512 / frame_shape
    factor_w = 512 / frame_shape

    # Do NOT scale the first dimension → use 1.0
    frames_upsampled = zoom(frames, (1, factor_h, factor_w), order = 1)

    return frames_upsampled

def get_trajectories(points: np.ndarray, frames: np.ndarray, model: nn.Module) -> dict:
    if points.ndim != 2:
        raise ValueError("Expected three dimensions <num_points, 2>")
    if points.shape[1] != 2:
        raise ValueError("Points are expected to have two elements")
    frames = scale_frames(frames)

    velocity_fields = get_motion_fields(frames, model)
    num_points = points.shape[0]

    trajectories = {}

    # Create trajectory for each point
    for i in range(num_points):
        point = points[i]
        trajectories[i] = [point]

    index = 0

    # Move over every velocity field
    for vf in tqdm(velocity_fields, desc='Tracking...'):
        # Obtain the velocity field components
        vf_x = vf[0]
        vf_y = vf[1]

        # Iterate over every point
        for i in range(num_points):
            # Obtain the point and its previous position
            point = points[i] 
            previous_position = (trajectories[i])[index]

            ipx = int(previous_position[0])
            ipy = int(previous_position[1])

            v_x = vf_x[ipy, ipx]
            v_y = vf_y[ipy, ipx]

            next_x = previous_position[0] + v_x 
            next_y = previous_position[1] + v_y

            trajectories[i].append(np.array([next_x, next_y]))
        
        index = index + 1
    
    return trajectories, velocity_fields