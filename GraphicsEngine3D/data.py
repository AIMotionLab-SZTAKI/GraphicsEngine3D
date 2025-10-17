from model import *
from utils import *

import os
from pathlib import Path
import numpy as np
import pickle
import hashlib
import json
import traceback
import re

class Data:
    def __init__(self, app):
        self.app = app
        self.folder = self.app.config['folder']

        self.folder_files = os.listdir(self.folder)

        # grid_seq: np.ndarray with indices: time,x,y,z
        # plans: list of dictionaries for every object to plot
        # heightmap: np.ndarray with shape (x,y) containing the height values

        # In order to load objects to the scene add an item to the dict with the corresponding function
        self.load_dict = {
            'grid': {'func': self.load_grid},
            'plans': {'func': self.load_plans},
            'terrain': {'func': self.load_terrain}
        }

        if 'all' in app.config['scene.objects']:
            app.config['scene.objects'] = list(self.load_dict.keys())
        for key in self.load_dict:
            if key in app.config['scene.objects']:
                try:
                    self.load_dict[key]['func']()
                    pass
                except Exception:
                    if key in app.config['scene.objects']: app.config['scene.objects'].remove(key)

        print(f'Successfully Loaded scene objects: {app.config["scene.objects"]}')

    def load_plans(self):
        # plans.pkl: list of dictionaries with keys: 'path_extracted', 'path_corrected', 'path_interp_BSpline', 'path_interp_MinimumSnapTrajectory', etc.
        # all path_* keys contain a numpy array with shape (time, x, y, z, ?(rotx, roty, rotz)?, ....) (rot not mandatory, but should be located at indices 4,5,6)

        plans_filenames = [fname for fname in self.folder_files if re.match(r'.*plans.*\.pkl', fname)]
        if len(plans_filenames) == 0:
            print(f"No plans file found in {self.folder}. No objects will be loaded.")
            self.plans = None
        else:
            self.plans = []
            for filename in plans_filenames:
                try:
                    with open(self.folder/filename, 'rb') as f:
                        plans = pickle.load(f)
                        if isinstance(plans, list):
                            self.plans.extend(plans)
                            print(f'Loaded {len(plans)} plans from {filename}')
                except Exception as e:
                    traceback.print_exc()
                    raise e
            if len(self.plans) == 0:
                print(f"No valid plans found in {self.folder}. No objects will be loaded.")
                self.plans = None

        try:
            self.world_dimensions = self.plans[0]['world_dimensions']
        except Exception as e:
            self.world_dimensions = np.array([1, 1, 1], dtype=np.float32)
            print(f'No world dimensions found in obj_plans.pkl. Using default dimensions: {self.world_dimensions}')

        # The DefaultOBJ class does not handle the shrinking of the object, so we need to load the objects with the correct scale.
        # Iterate through the object plans and convert the paths to the correct format
        for obj_plan in self.plans:

            print(f"Object plan {obj_plan['id']} loaded, type: {obj_plan['type']}, path shape: {obj_plan['path'].shape}, keys: {list(obj_plan.keys())}")
            world_dim = obj_plan.get('world_dimensions', self.world_dimensions)

            for key in obj_plan:
                if key.startswith('path') and not key.endswith('tcku'):

                    if np.issubdtype(obj_plan[key].dtype, np.floating):
                        print('Path is of type float (SI units), centering/scaling the path')
                        obj_plan[key][:,1:4] = obj_plan[key][:,1:4] - world_dim/2 # center
                        obj_plan[key][:,1:4] = 2 * obj_plan[key][:,1:4] / np.max(world_dim).astype(np.float32)

                    if np.issubdtype(obj_plan[key].dtype, np.integer):
                        print('Path is of type int (Grid units), converting to SI units and centering/scaling the path')
                        obj_plan[key] = obj_plan[key].astype(np.float32)
                        obj_plan[key][:,1:4] = 2 * (obj_plan[key][:,1:4] - np.array(world_dim/2)) / np.max(world_dim)

            # THIS PART IS FOR COMPATIBILITY WITH OLDER VERSIONS, IT ADDS THE NECESSARY SPLINE PATHS TO BE PLOTTED
            # Possible keys: 'path_extracted'[31/255,119/255,180/255,1], 'path_corrected'[44/255,160/255,44/255,1], 'path_interp_BSpline', 'path_interp_MinimumSnapTrajectory'
            if obj_plan['type'] in ('drone','uav'):
                if 'path_interp_BSpline' in obj_plan and 'path_interp_MinimumSnapTrajectory' not in obj_plan:
                    obj_plan['path'] = obj_plan['path_interp_BSpline']
                    self.plans.append({'id':f"{obj_plan['id']}_spline_BSpline", 'type':'spline', 'path':obj_plan['path_interp_BSpline'],'color':[0.8,0.2,0.2,1],'world_dimensions':obj_plan['world_dimensions']})
                elif 'path_interp_MinimumSnapTrajectory' in obj_plan:
                    obj_plan['path'] = obj_plan['path_interp_MinimumSnapTrajectory']
                    self.plans.append({'id':f"{obj_plan['id']}_spline_MinSnap", 'type':'spline', 'path':obj_plan['path_interp_MinimumSnapTrajectory'],'color':[0.8,0.2,0.2,1],'world_dimensions':obj_plan['world_dimensions']})


    def load_grid(self):
        try:
            grid_static = np.load(self.folder/'grid_static.npz')['grid_static']
            grid_seq = np.load(self.folder/'grid_seq.npz')['grid_seq']
            self.grid_shape = grid_static.shape
            print(f'grid_static shape: {grid_static.shape}, grid_seq shape: {grid_seq.shape}')
        except Exception as e:
            print(f'grid_seq.npz OR grid_static.npz not found in {self.folder}. No grid will be loaded')
            raise e
        
        try:
            self.grid_static_instancelist = convert_grid_static_to_instancelist(grid_static)
            self.grid_seq_dynamic_instancelist = convert_grid_seq_to_instancelist(grid_seq)
        except Exception as e:
            print(f'Error during conversion')
            raise e

    def load_terrain(self):
        """Load meshgrid or heightmap and convert to terrain mesh with caching"""
        
        # Define file paths
        terrain_obj_path = self.app.config['root_dir']/'objects/terrain/terrain.obj'
        terrain_texture_path = self.app.config['root_dir']/'objects/terrain/terrain.png'
        cache_file = self.app.config['root_dir']/'objects/terrain/terrain_cache.obj.json'

        def get_file_hash(filepath):
            """Calculate SHA256 hash of a file"""
            if not filepath.exists():
                return None
            
            hash_sha256 = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        
        def load_cache():
            """Load cache information"""
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except:
                    return {}
            return {}
        
        def save_cache(cache_data):
            """Save cache information"""
            cache_file.parent.mkdir(exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        
        try:
            # Load existing cache
            cache = load_cache()
            
            # Check if we need to process files
            needs_processing = False
            current_hashes = {}
            
            # Try to load meshgrid first
            meshgrid_file = self.folder/'meshgrid.npz'
            heightmap_file = self.folder/'heightmap.npy'
            
            if meshgrid_file.exists():
                print("Checking meshgrid.npz for changes...")
                current_hashes['meshgrid'] = get_file_hash(meshgrid_file)
                
                # Check if meshgrid changed
                if (cache.get('meshgrid_hash') != current_hashes['meshgrid'] or
                    not terrain_obj_path.exists() or 
                    not terrain_texture_path.exists()):
                    needs_processing = True
                    source_type = 'meshgrid'
                else:
                    print("Meshgrid unchanged, using cached terrain files.")
                    
            elif heightmap_file.exists():
                print("Checking heightmap.npy for changes...")
                current_hashes['heightmap'] = get_file_hash(heightmap_file)
                
                # Also check world_dimensions if they affect terrain generation
                world_dim_hash = None
                if hasattr(self, 'world_dimensions'):
                    world_dim_str = str(self.world_dimensions.tolist())
                    world_dim_hash = hashlib.sha256(world_dim_str.encode()).hexdigest()
                    current_hashes['world_dimensions'] = world_dim_hash
                
                # Check if heightmap or world dimensions changed
                if (cache.get('heightmap_hash') != current_hashes['heightmap'] or
                    cache.get('world_dimensions_hash') != current_hashes.get('world_dimensions') or
                    not terrain_obj_path.exists() or 
                    not terrain_texture_path.exists()):
                    needs_processing = True
                    source_type = 'heightmap'
                else:
                    print("Heightmap and world dimensions unchanged, using cached terrain files.")
            else:
                raise FileNotFoundError("Neither meshgrid.npz nor heightmap.npy found")
            
            # Process files only if needed
            if needs_processing:
                print(f"Processing terrain from {source_type}...")
                
                if source_type == 'meshgrid':
                    # Load meshgrid data
                    meshgrid_data = np.load(meshgrid_file)
                    X = meshgrid_data['X']
                    Y = meshgrid_data['Y'] 
                    Z = meshgrid_data['Z']
                    print(f'X range: [{X.min():.2f}, {X.max():.2f}], Y range: [{Y.min():.2f}, {Y.max():.2f}], Z range: [{Z.min():.2f}, {Z.max():.2f}]')
                    
                else:  # heightmap
                    # Load and process heightmap
                    heightmap = np.load(heightmap_file)
                    print(f'Heightmap shape: {heightmap.shape}')
                    
                    if 'grid' in self.app.config['scene.objects']:
                        self.world_dimensions_original = self.world_dimensions
                        self.world_dimensions = np.array([127,63,31])

                    # Create meshgrid from heightmap
                    if hasattr(self, 'world_dimensions'):
                        h, w = heightmap.shape
                        x_range = self.world_dimensions[0]
                        y_range = self.world_dimensions[1]
                        
                        x = np.linspace(-x_range/2, x_range/2, w)
                        y = np.linspace(-y_range/2, y_range/2, h)
                        X, Y = np.meshgrid(x, y)
                        Z = heightmap
                        
                        print(f'Using world_dimensions for meshgrid: {self.world_dimensions}, heightmap shape: {heightmap.shape}')
                    else:
                        h, w = heightmap.shape
                        x = np.linspace(-1, 1, w)
                        y = np.linspace(-1, 1, h)
                        X, Y = np.meshgrid(x, y)
                        Z = heightmap
                        print(f'Using fallback normalized meshgrid for heightmap shape: {heightmap.shape}')
                
                # Generate terrain files and clean up old cache files
                print("Generating terrain OBJ and texture files...")
                convert_meshgrid_to_terrain_obj(X, Y, Z, terrain_obj_path) 
                convert_heightmap_to_terrain_texture(Z, terrain_texture_path)
                (self.app.config['root_dir']/'objects/terrain/terrain.obj.bin').unlink(missing_ok=True)
                (self.app.config['root_dir']/'objects/terrain/terrain.obj.json').unlink(missing_ok=True)

                # Update cache
                cache.update(current_hashes)
                if source_type == 'meshgrid':
                    cache['meshgrid_hash'] = current_hashes['meshgrid']
                    cache.pop('heightmap_hash', None)  # Remove old heightmap hash
                    cache.pop('world_dimensions_hash', None)
                else:
                    cache['heightmap_hash'] = current_hashes['heightmap']
                    if 'world_dimensions' in current_hashes:
                        cache['world_dimensions_hash'] = current_hashes['world_dimensions']
                    cache.pop('meshgrid_hash', None)  # Remove old meshgrid hash
                
                save_cache(cache)
                print("Terrain processing complete and cached.")
            
        except Exception as e:
            print(f'Error loading terrain: {e}')
            raise e
