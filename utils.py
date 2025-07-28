import numpy as np
from PIL import Image
import itertools
import matplotlib as mpl
import pygame as pg
import moderngl as mgl
from pathlib import Path
import pickle as pkl

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

def convert_meshgrid_to_terrain_obj(X, Y, Z, obj_path):
    """
    Converts meshgrid coordinates (X, Y, Z) to a smooth surface and exports as OBJ.
    The meshgrid is normalized to fit within a 2x2x2 cube while preserving aspect ratios.
    :param X: 2D numpy array of X coordinates
    :param Y: 2D numpy array of Y coordinates
    :param Z: 2D numpy array of Z coordinates (heights)
    :param obj_path: Output OBJ file path
    """
    # Ensure all arrays have the same shape
    assert X.shape == Y.shape == Z.shape, "X, Y, Z must have the same shape"
    
    h, w = Z.shape
    
    # Calculate the ranges for each axis
    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()
    z_range = Z.max() - Z.min()
    
    # Find the largest dimension to scale by
    max_range = max(x_range, y_range, z_range)
    
    # Avoid division by zero
    if max_range == 0:
        max_range = 1.0
    
    # Scale factor to fit within 2x2x2 cube
    scale_factor = 2.0 / max_range
    
    # Center and scale the coordinates
    X_centered = (X - (X.max() + X.min()) / 2) * scale_factor
    Y_centered = (Y - (Y.max() + Y.min()) / 2) * scale_factor
    Z_centered = (Z - (Z.max() + Z.min()) / 2) * scale_factor
    
    # Flatten for OBJ vertices
    vertices = np.column_stack((X_centered.ravel(), Y_centered.ravel(), Z_centered.ravel()))

    # Compute normals using central differences on the scaled coordinates
    dzdx = np.gradient(Z_centered, X_centered[0, :], axis=1)  # gradient along x-axis
    dzdy = np.gradient(Z_centered, Y_centered[:, 0], axis=0)  # gradient along y-axis
    normals = np.dstack((-dzdx, -dzdy, np.ones_like(Z_centered)))
    n_flat = normals / np.linalg.norm(normals, axis=2, keepdims=True)
    n_flat = n_flat.reshape(-1, 3)

    # Texture coordinates (map original X, Y to [0,1])
    u = (X - X.min()) / (X.max() - X.min()) if X.max() > X.min() else np.zeros_like(X)
    v = (Y - Y.min()) / (Y.max() - Y.min()) if Y.max() > Y.min() else np.zeros_like(Y)
    texcoords = np.column_stack((u.ravel(), v.ravel()))

    # Faces (two triangles per quad, counter-clockwise)
    faces = []
    for i in range(h-1):
        for j in range(w-1):
            idx = i * w + j
            idx_right = idx + 1
            idx_down = idx + w
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

def create_texture_from_rgba(ctx, rgba, size=(1, 1)):
    """
    Create a solid color texture from RGBA values.
    
    Args:
        rgba: Tuple of (r, g, b, a) values (0-255)
        size: Tuple of (width, height) for texture size, default (1, 1)
    
    Returns:
        ModernGL texture object
    """
    # Ensure RGBA values are integers in range 0-255
    r, g, b, a = [int(max(0, min(255, val))) for val in rgba]
    
    # Create pygame surface with the specified size and RGBA format
    surface = pg.Surface(size, pg.SRCALPHA, 32)
    surface.fill((r, g, b, a))
    
    # Convert to string data for ModernGL
    texture_data = pg.image.tostring(surface, 'RGBA')
    
    # Create ModernGL texture
    texture = ctx.texture(size=size, components=4, data=texture_data)
    
    # Apply same settings as get_texture
    texture.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
    texture.build_mipmaps()
    texture.anisotropy = 32.0
    
    return texture

def center_obj_file(path):
    ''' Center the vertices of an OBJ file around the origin and comment out mtl lines.'''
    vertices = []
    lines = []

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)
            lines.append(line)

    vertices = np.array(vertices)
    #bbox = vertices.max(axis=0) - vertices.min(axis=0)
    #centered_vertices = vertices - (bbox / 2)
    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid

    vert_idx = 0
    with open(path, 'w') as f:
        for line in lines:
            if line.startswith('v '):
                v = centered_vertices[vert_idx]
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                vert_idx += 1
            elif 'mtl' in line:
                # Comment out lines containing 'mtl'
                if not line.startswith('#'):
                    f.write(f"#{line}")
                else:
                    f.write(line)
            else:
                f.write(line)

def get_demo_heightmap_from_grid_static():
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

def get_demo_data_for_obj_plans():
    folder = Path(__file__).parent/'demo/demo_Mate'

    path1 = np.load(folder/'back_and_forth_trajectory_1.npy')
    path2 = np.load(folder/'ellipsoidal_trajectory_map_centered.npy')
    path3 = np.load(folder/'ellipsoidal_trajectory_map_origin.npy')
    path4 = np.load(folder/'drone.npy')
    world_dim = np.array([400e3, 400e3, 6e3], dtype=np.float32)

    # DEBUG
    path4[:,0] -= path4[0,0]
    path4[:,6] = np.linspace(0, 2*np.pi, path4.shape[0])
    #path4[:,5] = 0
    #path4[:,6] = 0

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for i in range(4, 7):
        plt.scatter(path4[:,0], path4[:, i], label=f'Column {i}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    

    obj_plans = [{'id':'radar_0', 'type':'radar', 'path':path1,'color':(255,0,0,0.5),'world_dimensions':world_dim, 'dimension':80e3},
                 {'id':'cone_0', 'type':'cone', 'path':path2,'color':(0,255,0,0.5),'world_dimensions':world_dim, 'dimension':60e3},
                 {'id':'torus_0', 'type':'torus', 'path':path3,'color':(0,0,255,0.5),'world_dimensions':world_dim, 'dimension':40e3},
                 {'id':'drone_0', 'type':'drone', 'path':path4,'color':(0,0,0,1.0),'world_dimensions':world_dim, 'dimension':20e3}]
    
    with open(folder/'obj_plans.pkl', 'wb') as f:
        pkl.dump(obj_plans, f)  


def main():
    #center_obj_file(Path(__file__).parent/'objects/obj/drone.obj') # DELETE CACHE BEFORE RUNNING
    #get_demo_heightmap_from_grid_static()
    get_demo_data_for_obj_plans()

if __name__ == "__main__":
    main()