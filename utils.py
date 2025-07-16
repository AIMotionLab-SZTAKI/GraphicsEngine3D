import numpy as np
from PIL import Image
import itertools
import matplotlib as mpl

def getColorMap():
    cmap = mpl.colormaps['magma'].resampled(255)
    cmap = (1-cmap(np.linspace(0, 1, 256)))*255
    cmap = np.delete(cmap,-1,axis=1).tolist()
    return cmap

def convert_grid_static_to_instancelist(grid_static):
    # Convert grid_static to flattened data type
    indices_set = set()
    for idx in itertools.product(*map(range,grid_static.shape)):
        if grid_static[idx] > 0.01:
            if any(i == 0 or i == max_idx-1 for i, max_idx in zip(idx, grid_static.shape)):
                indices_set.add(idx)
            # Check each neighbor in all 6 directions if within bounds
            if idx[0]>0 and grid_static[idx] != grid_static[idx[0] - 1, idx[1], idx[2]]:
                indices_set.add(idx)
            if idx[0]<grid_static.shape[0]-1 and grid_static[idx] != grid_static[idx[0] + 1, idx[1], idx[2]]:
                indices_set.add(idx)
            if idx[1]>0 and grid_static[idx] != grid_static[idx[0], idx[1] - 1, idx[2]]:
                indices_set.add(idx)
            if idx[1]<grid_static.shape[1]-1 and grid_static[idx] != grid_static[idx[0], idx[1] + 1, idx[2]]:
                indices_set.add(idx)
            if idx[2]>0 and grid_static[idx] != grid_static[idx[0], idx[1], idx[2] - 1]:
                indices_set.add(idx)
            if idx[2]<grid_static.shape[2]-1 and grid_static[idx] != grid_static[idx[0], idx[1], idx[2] + 1]:
                indices_set.add(idx)

    indices = np.array(list(indices_set), dtype=int)

    grid_static[np.isinf(grid_static)] = 1
    values = np.array([grid_static[i, j, k] for i, j, k in indices])
    grid_static_instancelist = np.column_stack((indices, np.expand_dims(values,axis=1)))
    #indices = np.where(grid_static >= 0.01)
    #self.grid_static_instancelist = np.column_stack(( np.transpose(indices), grid_static[indices]))
    return grid_static_instancelist

def convert_grid_seq_to_instancelist(grid_seq):
    # Convert grid_seq to flattened data type (sequence of arrays containing all the instance indices and values)
    indices = np.where((grid_seq >= 0.01) & (grid_seq < np.inf))                 # determine the max number of dynamic instances PER FRAME
    max_instance_per_frame = np.max(np.bincount(indices[0]))

    swap_Y = False # Swaps the addition order of the indices for the Y axis, so the direction of transparency changes

    grid_seq_dynamic_instancelist = np.zeros((grid_seq.shape[0], max_instance_per_frame, 4))
    for i in range(grid_seq.shape[0]):
        grid_transpose = np.transpose(grid_seq[i], axes=(2,1,0))
        if swap_Y: grid_transpose = np.flip(grid_transpose,axis=1)
        indices = np.where((grid_transpose >= 0.01) & (grid_transpose < np.inf))
        if swap_Y: indices = (indices[0], grid_seq[i].shape[1]-1-indices[1], indices[2])

        indices = tuple(indices[::-1])
        grid_seq_dynamic_instancelist[i,0:len(indices[0])] = \
            np.column_stack((np.transpose(indices), grid_seq[i][indices])) # shape: time x instance x (index, value)

    return grid_seq_dynamic_instancelist

def convert_heightmap_to_terrain_obj(heightmap, obj_path, resolution=1.0):
    """
    Converts a 2D numpy heightmap to a smooth surface and exports as OBJ.
    :param heightmap: 2D numpy array of heights
    :param obj_path: Output OBJ file path
    :param resolution: scale factor for every direction (X, Y, Z)
    """
    h, w = heightmap.shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    Z = heightmap

    # Get ranges for each axis
    x_range = w - 1 if w > 1 else 1
    y_range = h - 1 if h > 1 else 1
    z_range = np.max(Z) - np.min(Z) if np.max(Z) > np.min(Z) else 1

    # Normalize to [0,1] and scale to rectangle
    Xs = (X - X.min()) / x_range
    Ys = (Y - Y.min()) / y_range
    Zs = (Z - np.min(Z)) / z_range

    # Center the mesh
    Xs = (Xs - 0.5) * x_range * resolution
    Ys = (Ys - 0.5) * y_range * resolution
    Zs = (Zs - 0.5) * z_range * resolution

    # Flatten for OBJ
    vertices = np.column_stack((Xs.ravel(), Ys.ravel(), Zs.ravel()))

    # Compute normals using central differences (on scaled Z)
    dzdx = np.gradient(Zs, axis=1)
    dzdy = np.gradient(Zs, axis=0)
    normals = np.dstack((-dzdx, -dzdy, np.ones_like(Zs)))
    n_flat = normals / np.linalg.norm(normals, axis=2, keepdims=True)
    n_flat = n_flat.reshape(-1, 3)

    # Texture coordinates (map X, Y to [0,1])
    nrows, ncols = Z.shape
    u = (X - X.min()) / (X.max() - X.min()) if X.max() > X.min() else X
    v = (Y - Y.min()) / (Y.max() - Y.min()) if Y.max() > Y.min() else Y
    texcoords = np.column_stack((u.ravel(), v.ravel()))

    # Faces (two triangles per quad, counter-clockwise)
    faces = []
    for i in range(nrows-1):
        for j in range(ncols-1):
            idx = i * ncols + j
            idx_right = idx + 1
            idx_down = idx + ncols
            idx_down_right = idx_down + 1
            # Triangle 1 (CCW)
            faces.append((idx+1, idx_right+1, idx_down+1))
            # Triangle 2 (CCW)
            faces.append((idx_right+1, idx_down_right+1, idx_down+1))

    # Write OBJ
    with open(obj_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in texcoords:
            f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        for n in n_flat:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for face in faces:
            # OBJ is 1-indexed, and we use the same index for v, vt, and vn
            f.write(f"f {face[0]}/{face[0]}/{face[0]} {face[1]}/{face[1]}/{face[1]} {face[2]}/{face[2]}/{face[2]}\n")

def convert_heightmap_to_terrain_texture(heightmap, png_path):
    """
    Generate a terrain-like RGB PNG texture from a heightmap using the provided colormap.
    :param heightmap: 2D numpy array of heights
    :param png_path: Output PNG file path
    """
    # Normalize heightmap to [0,1]
    hmap = (heightmap - np.min(heightmap)) / (np.ptp(heightmap) + 1e-8)

    # Vectorized colormap
    val = np.clip(hmap, 0.0, 1.0)
    R = 2.335*val**3 - 5.957*val**2 + 3.516*val + 0.316
    G = 2.774*val**3 - 5.416*val**2 + 2.151*val + 0.623
    B = 2.172*val**3 - 4.441*val**2 + 2.178*val + 0.129
    rgb = np.stack([R, G, B], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255).astype(np.uint8)

    # Flip texture to match heightmap orientation
    rgb = np.flip(rgb, axis=0)

    img = Image.fromarray(rgb, 'RGB')
    img.save(png_path)

def main():
    from pathlib import Path
    grid_static_path = Path(__file__).parents[1] / 'Data/Processing/demo_5drones/grid_static.npz'
    grid_static = np.load(grid_static_path)['grid_static']  # shape: (x, y, z)

    # For each (x, y), find the highest z index where grid_static[x, y, z] == np.inf
    x_dim, y_dim, z_dim = grid_static.shape
    heightmap = np.zeros((x_dim, y_dim), dtype=float)
    for i in range(x_dim):
        for j in range(y_dim):
            # Find the highest z index where grid_static[i, j, z] == np.inf
            obs_indices = np.where(grid_static[i, j, :] == np.inf)[0]
            if len(obs_indices) > 0:
                # Take the highest z index and scale to [0, 1]
                heightmap[i, j] = obs_indices[-1] / (z_dim - 1)
            else:
                heightmap[i, j] = 0.0


    heightmap = heightmap * z_dim  # Scale to the original z dimension
    heightmap_path = Path(__file__).parents[1] / 'Data/Processing/demo_5drones/heightmap.npy'
    np.save(heightmap_path, heightmap)

    print(f"grid_static shape: {grid_static.shape}")
    print(f"heightmap shape: {heightmap.shape}")
    print(f"Heightmap max value: {np.max(heightmap)}")

    '''
    # Example: create a test heightmap (e.g., a Gaussian hill)
    h, w = 128, 64
    y, x = np.mgrid[0:h, 0:w]
    heightmap = np.exp(-((x-w/2)**2 + (y-h/2)**2) / (2*(w/5)**2)) * 16
    np.save(Path(__file__).parents[1]/'Data/Processing/demo_5drones/heightmap.npy', heightmap)
    '''

    






if __name__ == "__main__":
    main()