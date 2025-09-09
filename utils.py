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

def convert_stl_to_obj(stl_path, obj_path, generate_texture_coords=True):
    """
    Convert STL file to OBJ file with vertices (v), texture coordinates (vt), and normals (vn).
    
    Args:
        stl_path: Path to input STL file
        obj_path: Path to output OBJ file
        generate_texture_coords: Whether to generate texture coordinates
    """
    try:
        from stl import mesh
    except ImportError:
        raise ImportError("numpy-stl is required. Install with: pip install numpy-stl")
    
    # Load STL file
    stl_mesh = mesh.Mesh.from_file(str(stl_path))
    
    # Extract vertices and normals
    vertices = stl_mesh.vectors.reshape(-1, 3)  # Flatten triangles to vertices
    normals = np.repeat(stl_mesh.normals, 3, axis=0)  # Each face normal repeated 3 times
    
    # Remove duplicate vertices and create index mapping
    unique_vertices, vertex_indices = np.unique(vertices, axis=0, return_inverse=True)
    
    # Average normals for shared vertices
    unique_normals = np.zeros_like(unique_vertices)
    for i in range(len(unique_vertices)):
        mask = vertex_indices == i
        unique_normals[i] = np.mean(normals[mask], axis=0)
        # Normalize the averaged normal
        norm = np.linalg.norm(unique_normals[i])
        if norm > 0:
            unique_normals[i] /= norm
    
    # Generate texture coordinates if requested
    if generate_texture_coords:
        # Simple planar projection based on X and Z coordinates
        min_x, max_x = np.min(unique_vertices[:, 0]), np.max(unique_vertices[:, 0])
        min_z, max_z = np.min(unique_vertices[:, 2]), np.max(unique_vertices[:, 2])
        
        # Avoid division by zero
        x_range = max_x - min_x if max_x != min_x else 1.0
        z_range = max_z - min_z if max_z != min_z else 1.0
        
        texture_coords = np.column_stack([
            (unique_vertices[:, 0] - min_x) / x_range,  # U coordinate
            (unique_vertices[:, 2] - min_z) / z_range   # V coordinate
        ])
    else:
        # Default texture coordinates (0, 0) for all vertices
        texture_coords = np.zeros((len(unique_vertices), 2))
    
    # Create faces using the vertex indices
    faces = vertex_indices.reshape(-1, 3)
    
    # Write OBJ file
    obj_path = Path(obj_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(obj_path, 'w') as f:
        f.write(f"# OBJ file converted from {stl_path}\n")
        f.write(f"# Vertices: {len(unique_vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")
        
        # Write vertices
        for vertex in unique_vertices:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        
        f.write("\n")
        
        # Write texture coordinates
        for texcoord in texture_coords:
            f.write(f"vt {texcoord[0]:.6f} {texcoord[1]:.6f}\n")
        
        f.write("\n")
        
        # Write normals
        for normal in unique_normals:
            f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
        
        f.write("\n")
        
        # Write faces (OBJ is 1-indexed)
        for face in faces:
            # Format: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
            f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                   f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                   f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")
    
    print(f"Successfully converted {stl_path} to {obj_path}")
    print(f"  Vertices: {len(unique_vertices)}")
    print(f"  Faces: {len(faces)}")
    print(f"  Normals: {len(unique_normals)}")
    print(f"  Texture coordinates: {len(texture_coords)}")


def convert_stl_to_obj_advanced(stl_path, obj_path, texture_mapping='planar'):
    """
    Advanced STL to OBJ converter with different texture mapping options.
    
    Args:
        stl_path: Path to input STL file
        obj_path: Path to output OBJ file
        texture_mapping: 'planar', 'cylindrical', 'spherical', or 'cubic'
        normalize: Whether to normalize the model
    """
    try:
        from stl import mesh
    except ImportError:
        raise ImportError("numpy-stl is required. Install with: pip install numpy-stl")
    
    # Load and process STL (same as above)
    stl_mesh = mesh.Mesh.from_file(str(stl_path))
    vertices = stl_mesh.vectors.reshape(-1, 3)
    normals = np.repeat(stl_mesh.normals, 3, axis=0)
    
    unique_vertices, vertex_indices = np.unique(vertices, axis=0, return_inverse=True)
    
    # Average normals for shared vertices
    unique_normals = np.zeros_like(unique_vertices)
    for i in range(len(unique_vertices)):
        mask = vertex_indices == i
        unique_normals[i] = np.mean(normals[mask], axis=0)
        norm = np.linalg.norm(unique_normals[i])
        if norm > 0:
            unique_normals[i] /= norm
    
    # Generate texture coordinates based on mapping type
    if texture_mapping == 'planar':
        # XZ plane projection
        min_x, max_x = np.min(unique_vertices[:, 0]), np.max(unique_vertices[:, 0])
        min_z, max_z = np.min(unique_vertices[:, 2]), np.max(unique_vertices[:, 2])
        x_range = max_x - min_x if max_x != min_x else 1.0
        z_range = max_z - min_z if max_z != min_z else 1.0
        
        texture_coords = np.column_stack([
            (unique_vertices[:, 0] - min_x) / x_range,
            (unique_vertices[:, 2] - min_z) / z_range
        ])
        
    elif texture_mapping == 'cylindrical':
        # Cylindrical mapping around Y-axis
        x, y, z = unique_vertices[:, 0], unique_vertices[:, 1], unique_vertices[:, 2]
        u = (np.arctan2(z, x) + np.pi) / (2 * np.pi)  # [0, 1]
        
        min_y, max_y = np.min(y), np.max(y)
        y_range = max_y - min_y if max_y != min_y else 1.0
        v = (y - min_y) / y_range  # [0, 1]
        
        texture_coords = np.column_stack([u, v])
        
    elif texture_mapping == 'spherical':
        # Spherical mapping
        x, y, z = unique_vertices[:, 0], unique_vertices[:, 1], unique_vertices[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        r = np.where(r == 0, 1e-8, r)  # Avoid division by zero
        
        u = (np.arctan2(z, x) + np.pi) / (2 * np.pi)  # [0, 1]
        v = (np.arcsin(np.clip(y / r, -1, 1)) + np.pi/2) / np.pi  # [0, 1]
        
        texture_coords = np.column_stack([u, v])
        
    elif texture_mapping == 'cubic':
        # Cubic mapping (based on dominant normal direction)
        texture_coords = np.zeros((len(unique_vertices), 2))
        
        for i, (vertex, normal) in enumerate(zip(unique_vertices, unique_normals)):
            # Find dominant axis
            abs_normal = np.abs(normal)
            dominant_axis = np.argmax(abs_normal)
            
            if dominant_axis == 0:  # X-dominant
                texture_coords[i] = [(vertex[1] + 1) / 2, (vertex[2] + 1) / 2]
            elif dominant_axis == 1:  # Y-dominant
                texture_coords[i] = [(vertex[0] + 1) / 2, (vertex[2] + 1) / 2]
            else:  # Z-dominant
                texture_coords[i] = [(vertex[0] + 1) / 2, (vertex[1] + 1) / 2]
        
        # Normalize to [0, 1]
        texture_coords = np.clip(texture_coords, 0, 1)
    
    else:
        # Default: simple planar
        texture_coords = np.zeros((len(unique_vertices), 2))
    
    # Write OBJ file (same as above)
    faces = vertex_indices.reshape(-1, 3)
    
    obj_path = Path(obj_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(obj_path, 'w') as f:
        f.write(f"# OBJ file converted from {stl_path}\n")
        f.write(f"# Texture mapping: {texture_mapping}\n")
        f.write(f"# Vertices: {len(unique_vertices)}, Faces: {len(faces)}\n\n")
        
        # Write vertices
        for vertex in unique_vertices:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        
        f.write("\n")
        
        # Write texture coordinates
        for texcoord in texture_coords:
            f.write(f"vt {texcoord[0]:.6f} {texcoord[1]:.6f}\n")
        
        f.write("\n")
        
        # Write normals
        for normal in unique_normals:
            f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
        
        f.write("\n")
        
        # Write faces
        for face in faces:
            f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                   f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                   f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")
    
    print(f"Successfully converted {stl_path} to {obj_path} with {texture_mapping} mapping")


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
    world_dim = np.array([400e3, 400e3, 6e3])

    # DEBUG
    #path4[:,0] -= path4[0,0]
    #path4[:,6] = np.linspace(0, 2*np.pi, path4.shape[0])
    #path4[:,5] = 0
    #path4[:,6] = 0

    time = np.linspace(0, path4[-1][0], path4.shape[0])
    color = np.random.rand(path4.shape[0], 4)
    color = np.column_stack((time, color))  # T x [t,r,g,b,a]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for i in range(4, 7):
        plt.scatter(path2[:,0], path2[:, i], label=f'Column {i}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    obj_plans = [{'id':'radar_0', 'type':'radar', 'path':path1,'color':(255,0,0,0.5),'world_dimensions':world_dim, 'dimension':80e3},
                 {'id':'radar_1', 'type':'radar', 'path':np.zeros_like(path1),'color':(255,0,0,0.5),'world_dimensions':world_dim, 'dimension':[50e3,30e3,5e3]},
                 {'id':'cone_0', 'type':'cone', 'path':path2,'color':(0,255,0,0.5),'world_dimensions':world_dim, 'dimension':60e3},
                 {'id':'torus_0', 'type':'torus', 'path':path3,'color':(0,0,255,0.5),'world_dimensions':world_dim, 'dimension':40e3},
                 {'id':'drone_0', 'type':'drone', 'path':path4,'color':color,'world_dimensions':world_dim, 'dimension':20e3}]
    
    with open(folder/'obj_plans.pkl', 'wb') as f:
        pkl.dump(obj_plans, f)  


def main():
    folder = Path(__file__).parent
    #get_demo_heightmap_from_grid_static()
    get_demo_data_for_obj_plans()
    #convert_stl_to_obj(stl_path=folder/'objects/drone/quad.stl', obj_path=folder/'objects/drone/uav.obj', generate_texture_coords=True)

if __name__ == "__main__":
    main()