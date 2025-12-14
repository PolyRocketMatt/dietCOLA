import argparse
import csv
import numpy as np
import shortuuid

from pathlib import Path
from scipy.ndimage import rotate, zoom, shift
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from diet4cola.advection import advect_backward_sim, advect_forward_sim
from diet4cola.cut import CoLACut
from diet4cola.cortex import Cortex, CortexSpec, generate_cortex_example
from diet4cola.mask import mask_on_condition
from diet4cola.operations import invert, directional_gradients, blur, mul
from diet4cola.velocity import compute_velocity_fields, prior_exp
from diet4cola.sdf import sdf_box, sdf_capsule, compute_rounded_sdfs
from diet4cola.utils import save_array, load_array

# Grid parameters
WIDTH = 512
HEIGHT = 512

#MULTIPLIER = 1 / 0.05035490409067586
MULTIPLIER = 2

def initialize_velocity_exponential(alpha: float, t: float, k: float) -> float:
    return alpha * np.exp(-t * k)

def initialize_velocity_hyperbolic(alpha: float, t: float, k: float) -> float:
    return (alpha ** 2) * (1.0 / (k * t + alpha))

def augment_array_chw(arr,
                      rotate_deg=0,
                      scale_factor=1.0,
                      translate=(0, 0)):
    C, H, W = arr.shape
    out = np.zeros_like(arr)

    for c in range(C):
        out[c] = augment_single_2d(arr[c],
                                   rotate_deg=rotate_deg,
                                   scale_factor=scale_factor,
                                   translate=translate)

    return out

def augment_single_2d(arr,
                      rotate_deg=0,
                      scale_factor=1.0,
                      translate=(0, 0)):

        # 1. Rotate around center
        arr_rot = rotate(arr, rotate_deg, reshape=False,
                         order=1, mode='constant', cval=0.0)

        # 2. Scale around center
        if scale_factor != 1.0:
            scaled = zoom(arr_rot, scale_factor, order=1)
            sh, sw = scaled.shape
            H, W = arr.shape
            out = np.zeros_like(arr)

            # center placement
            start_y = (H - sh) // 2
            start_x = (W - sw) // 2

            y0_src = max(0, -start_y); y0_dst = max(0, start_y)
            x0_src = max(0, -start_x); x0_dst = max(0, start_x)

            y1_src = y0_src + min(H - y0_dst, sh - y0_src)
            x1_src = x0_src + min(W - x0_dst, sw - x0_src)

            out[
                y0_dst:y0_dst+(y1_src - y0_src),
                x0_dst:x0_dst+(x1_src - x0_src)
            ] = scaled[y0_src:y1_src, x0_src:x1_src]

            arr_scaled = out
        else:
            arr_scaled = arr_rot

        # 3. Shift (zero padding)
        arr_shifted = shift(arr_scaled,
                            shift=translate,
                            order=1,
                            mode='constant',
                            cval=0.0)

        return arr_shifted

def gen_advected_cortex_worker(idx: int, out: str, step: float, max_offset: int, extent_major: int,
                               extent_minor: int, t_max: float) -> dict:
    # 1. Simulation Parameters
    t_min = 0.0
    t_max = t_max  # Total simulation time (in seconds)
    time_step = step
    timepoints = np.arange(t_min, t_max + time_step, time_step)
    iterations = len(timepoints)
    normalized_timepoints = timepoints / t_max

    # 2. Generate Cortex (Initial State)
    spec = CortexSpec()
    seed = np.random.randint(0, 1000000000)
    np.random.seed(seed)
    spec.seed = seed
    spec.max_center_offset = max_offset
    spec.max_cell_extent = (extent_major, extent_minor)
    spec.cell_blur_radius = np.random.randint(45, 55)
    cortex = generate_cortex_example(spec)

    # Calculate cut center (randomized offset as in notebook Cell 3)
    off_x = WIDTH // 2 + (np.random.randint(64) - 32)
    off_y = HEIGHT // 2 + (np.random.randint(64) - 32)
    
    # 3. Create CoLA Cut    
    cola_cut = CoLACut((off_x, off_y), 64, 150, 200, seed)

    # 4. Compute Time-Dependent SDF/Velocity Parameters
    # 4a. Sample initial parameters
    sdf_radius = np.random.rand() * 5               # Max SDF radius parameter of 5
    falloff_distance = np.random.randint(0, 100)    # Max falloff distance of 100
    k = np.random.rand() * 0.5

    # 4b. Sample initial velocity (based on annotated data)
    v_0_mean = 3.146
    v_0_stddev = 1.002
    v_0 = max(0.0, np.random.normal(v_0_mean, v_0_stddev))    # Initial velocity based on mean and stddev of data

    v_0 = v_0 * MULTIPLIER # Multiplier to account for scale difference!

    # 4c. Create velocity profiles for the edge of the SDF
    initial_velocities = np.array([initialize_velocity_exponential(v_0, t, k) for t in timepoints])
    
    # 4d. Create radius and width of the cut over time
    radii = [0] * iterations
    widths = [0] * iterations

    for i in range(iterations):
        if i == 0:
            widths[i] = 0
            continue

        widths[i] = widths[i - 1] + time_step * initial_velocities[i]

    # 5. Compute SDF and Velocity Fields
    # 5a. Compute SDF fields
    sdf_fields = compute_rounded_sdfs(WIDTH, HEIGHT, cola_cut.cut_origin, cola_cut.cut_destination, widths, radii, False, sdf_capsule, iterations)
    
    # 5b. Create SDF Masks and Clipped SDF
    masked_sdf_fields = np.array([mask_on_condition(sdf < 0, 0, 1) for sdf in sdf_fields]) 
    clipped_rounded_sdf_fields = np.array([mask_on_condition(sdf < 0, 0, sdf) for sdf in sdf_fields])

    # 5c. Compute raw velocity fields
    velocity_fields = compute_velocity_fields(clipped_rounded_sdf_fields, prior_exp, timepoints, initial_velocities, falloff_distance, iterations)
    
    # 5d. Create masked velocity fields and zero out first timestep
    masked_velocity_fields = np.array([vf * msdf for (vf, msdf) in zip(velocity_fields, masked_sdf_fields)])
    masked_velocity_fields[0, :, :] = np.where(clipped_rounded_sdf_fields[0, :, :] < 1, masked_velocity_fields[0, :, :], 0)
    
    # 5e. Multiply the masked velocity fields, this time with the cell mask (i.e. ONLY get a velocity field along the cell)
    cell_boundary = cortex.cell_mask
    masked_velocity_fields = masked_velocity_fields * cell_boundary

    # 5f. Compute SDF gradients
    sdf_gradients = directional_gradients(clipped_rounded_sdf_fields)
    sdf_dys = sdf_gradients[:, 0, :, :]
    sdf_dxs = sdf_gradients[:, 1, :, :]

    # 5g. Normalize the SDF gradients, s.t. we can infer velocity in dx and dy directions
    grad_mag = np.sqrt(sdf_dxs ** 2 + sdf_dys ** 2 + 1e-8)
    norm_dxs = sdf_dxs / grad_mag
    norm_dys = sdf_dys / grad_mag

    # 5h. Compute velocity components vel_dxs and vel_dys
    vel_dxs = norm_dxs * velocity_fields
    vel_dys = norm_dys * velocity_fields

    # 5i. Compute blurred SDF fields 
    blurred_masked_sdf_fields = np.array([blur(sdf.astype(np.float64), (5, 5), 5) for sdf in masked_sdf_fields])

    # 6. Perform Backward Advection 
    # 6a. Backward advection for the actomyosin cortex
    advected_cortex = advect_backward_sim(
        cortex.data, 
        sdf_dxs, 
        sdf_dys, 
        velocity_fields, 
        iterations, 
        time_step
    )

    # 6b. Multiply the advected cortices with the blurred SDF fields to obtain a similar data representation as the actual input.
    non_multiplied = advected_cortex.copy()
    advected_cortex = mul(advected_cortex, blurred_masked_sdf_fields)
    
    # 6d. Better training if velocity field is multiplied by actomyosin layer? (Black pixels can't move)
    # TODO -> Make this a BINARY MASK based on advected cortex above some threshold!
    threshold = 0
    cortex_mask = (advected_cortex > threshold).astype(float)
    vel_dxs = vel_dxs * advected_cortex
    vel_dys = vel_dys * advected_cortex

    # 6c. Diversify
    aug_angle = 0 #np.random.randint(360)
    aug_scale = 1 #np.random.rand() * 2
    aug_trans = (0, 0) #(np.random.randint(128) - 64, np.random.randint(128) - 64)

    non_multiplied = augment_array_chw(non_multiplied, aug_angle, aug_scale, aug_trans)
    advected_cortex = augment_array_chw(advected_cortex, aug_angle, aug_scale, aug_trans)
    velocity_fields = augment_array_chw(velocity_fields, aug_angle, aug_scale, aug_trans)
    vel_dxs = augment_array_chw(vel_dxs, aug_angle, aug_scale, aug_trans)
    vel_dys = augment_array_chw(vel_dys, aug_angle, aug_scale, aug_trans)
    cell_boundary = augment_single_2d(cell_boundary, aug_angle, aug_scale, aug_trans)

    # 7. Save the advected fields and the velocity field
    # 7a. Generate a UUID for this cell, just have a unique identifier
    cell_uuid = shortuuid.uuid()

    # 7b. Save the advected cortex and its separate layers
    save_array(non_multiplied, Path(out) / f'cortex_{cell_uuid}_poster_example.npy')
    save_array(advected_cortex, Path(out) / f'cortex_{cell_uuid}_actomyosin.npy')

    # 7c. Save the velocity field that was used to advect the cortex
    save_array(clipped_rounded_sdf_fields, Path(out) / f'cortex_{cell_uuid}_sdfs.npy')
    save_array(velocity_fields, Path(out) / f'cortex_{cell_uuid}_velocity_field.npy')
    save_array(vel_dxs, Path(out) / f'cortex_{cell_uuid}_velocity_field_dx.npy')
    save_array(vel_dys, Path(out) / f'cortex_{cell_uuid}_velocity_field_dy.npy')

    save_array(cell_boundary, Path(out) / f'cortex_{cell_uuid}_boundary.npy')

    # Return useful metadata
    return {
        # General metadata for reproduction
        'idx': idx,
        'uuid': cell_uuid,
        'seed': seed,

        # Cell metadata
        'cell_extent': cortex.cell_extent,
        'cell_angle': cortex.cell_angle,
        
        # Cut metadata
        'cut_off_x': off_x,
        'cut_off_y': off_y,
        'cut_off_bound': 32,
        'cut_length_limit': 150,

        # Velocity metadata
        'sdf_radius': sdf_radius,
        'initial_velocity': v_0,
        'falloff_distance': falloff_distance,
        'prior': 'exponential',

        # Simulation metadata
        'width': WIDTH,
        'height': HEIGHT,
        't_min': t_min,
        't_max': t_max,
        'step': time_step
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel Cortex Generator with Advection")
    parser.add_argument("--n", type=int, required=True, help="Total number of cortices to generate")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers for parallel processing")
    parser.add_argument("--out", type=str, default="./data", help="Output directory for generated data")
    parser.add_argument("--step", type=float, default=0.5, help="Timestep to use in the advection simulation")
    parser.add_argument("--max_offset", type=int, default=100, help="Max offset from center for cell placement")
    parser.add_argument("--extent_long", type=int, default=300, help="Max extent for the major axis of the cell")
    parser.add_argument("--extent_short", type=int, default=300, help="Max extent for the minor axis of the cell")
    parser.add_argument("--t_max", type=float, default=5, help="Time of the simulation")
    args = parser.parse_args()

    Path(args.out).mkdir(exist_ok=True)

    # Prepare tasks
    tasks = [(i, args.out, args.step, args.max_offset, args.extent_long, args.extent_short, args.t_max) for i in range(args.n)]

    results = [0] * args.n
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = [executor.submit(gen_advected_cortex_worker, *t) for t in tasks]

        # Proper progress bar with as_completed
        for f in tqdm(as_completed(futures), total=args.n, desc="Generating advected cortices"):
            result = f.result()
            results[result['idx']] = result

    # Define a custom header for the new metadata
    csv_header = [
        'idx', 
        'uuid',
        'seed', 
        'cell_extent', 
        'cell_angle', 
        'cut_off_x',
        'cut_off_y',
        'cut_off_bound',
        'cut_length_limit',
        'sdf_radius',
        'initial_velocity',
        'falloff_distance',
        'prior',
        'width',
        'height',
        't_min',
        't_max',
        'step'
    ]

    # Transform the results to match the header
    csv_rows = []
    for cortex_metadata in tqdm(results, desc='Saving metadata'):
        csv_rows.append(cortex_metadata)

    csv_out_path = Path(args.out) / "cortices_metadata.csv"
    file_exists = csv_out_path.exists()

    with open(csv_out_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        if not file_exists:
            writer.writeheader()
        writer.writerows(csv_rows)

    print(f'Done.')


if __name__ == '__main__':
    main()