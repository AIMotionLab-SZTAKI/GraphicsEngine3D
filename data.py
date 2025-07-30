from model import *
from utils import *

from pathlib import Path
import numpy as np
import pickle

class Data:
    def __init__(self, app):
        self.app = app
        try:
            self.folder = Path(app.config['folder'])
        except:
            print(f'Invalid folder path: {app.config['folder']}. Please provide a valid path.')

        # grid_seq: np.ndarray with indices: time,x,y,z
        # plans: list of dictionaries with keys: 'path_extracted' ,'path_corrected' ,'path_interp_BSpline', 'path_interp_MinimumSnapTrajectory', etc.
        # heightmap: np.ndarray with shape (x,y) containing the height values

        # In order to load objects to the scene add an item to the dict with the corresponding function
        load_func_dict = {
            'grid': self.load_grid,
            'plans': self.load_plans,
            'obj': self.load_obj_plans,
            'terrain': self.load_terrain
        }

        self.scene = app.config['scene'] if isinstance(app.config['scene'], list) else [app.config['scene']]
        if 'all' in self.scene:
            self.scene = list(load_func_dict.keys())
        for key in load_func_dict:
            if key in self.scene:
                try:
                    load_func_dict[key]()
                    pass
                except Exception:
                    if key in self.scene: self.scene.remove(key)

        print(f'Successfully Loaded scene objects: {self.scene}')

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

    def load_plans(self):
        # plans.pkl: list of dictionaries with keys: 'path_extracted', 'path_corrected', 'path_interp_BSpline', 'path_interp_MinimumSnapTrajectory', etc.
        # all path_* keys contain a numpy array with shape (time, x, y, z, ?(rotx, roty, rotz)?, ......) 
        # (rot not mandatory, but should be located at indices 4,5,6)
        try:
            with open(self.folder/'plans.pkl', 'rb') as f: 
                self.plans = pickle.load(f)
                
                for i, plan in enumerate(self.plans):
                    print(f"Plan {i} keys: {list(plan.keys())}") # DEBUG

                    #if not 'world_dimensions' in self.plans[i]:s
                    #    self.plans[i]['world_dimensions'] = self.plans[i]['grid_shape'] - 1
        
        except Exception as e:
            self.plans = None
            print('plans.pkl not found in {folder}. No plans will be loaded.')
            raise e

    def load_terrain(self):
        ### Load meshgrid or heightmap and convert to terrain mesh
        try:
            # First try to load meshgrid file
            try:
                meshgrid_data = np.load(self.folder/'meshgrid.npz')
                X = meshgrid_data['X']
                Y = meshgrid_data['Y'] 
                Z = meshgrid_data['Z']
                print(f'X range: [{X.min():.2f}, {X.max():.2f}], Y range: [{Y.min():.2f}, {Y.max():.2f}], Z range: [{Z.min():.2f}, {Z.max():.2f}]')
                
            except FileNotFoundError:
                print('meshgrid.npz not found, falling back to heightmap and creating meshgrid...')
                heightmap = np.load(self.folder/'heightmap.npy')
                print(f'Heightmap shape: {heightmap.shape}')
                
                if 'grid' in self.scene:
                    self.world_dimensions_original = self.world_dimensions
                    self.world_dimensions = np.array([127,63,31])

                # Create meshgrid from heightmap
                if hasattr(self, 'world_dimensions'):
                    # Use world_dimensions to scale the meshgrid properly
                    
                    h, w = heightmap.shape
                    x_range = self.world_dimensions[0]  # x dimension from world_dimensions
                    y_range = self.world_dimensions[1]  # y dimension from world_dimensions
                    
                    # Create meshgrid scaled to world dimensions
                    x = np.linspace(-x_range/2, x_range/2, w)
                    y = np.linspace(-y_range/2, y_range/2, h)
                    X, Y = np.meshgrid(x, y)
                    Z = heightmap
                    
                    print(f'Using world_dimensions for meshgrid: {self.world_dimensions}, heightmap shape: {heightmap.shape}')

                else:
                    # Fallback: use heightmap shape to create normalized meshgrid
                    h, w = heightmap.shape
                    x = np.linspace(-1, 1, w)
                    y = np.linspace(-1, 1, h)
                    X, Y = np.meshgrid(x, y)
                    Z = heightmap
                    print(f'Using fallback normalized meshgrid for heightmap shape: {heightmap.shape}')
            
            terrain_obj_path = Path(__file__).parent/'objects/terrain/terrain.obj'
            terrain_texture_path = Path(__file__).parent/'objects/terrain/terrain.png'
            convert_meshgrid_to_terrain_obj(X, Y, Z, terrain_obj_path) 
            convert_heightmap_to_terrain_texture(Z, terrain_texture_path)
            (Path(__file__).parent/'objects/terrain/terrain.obj.bin').unlink(missing_ok=True)
            (Path(__file__).parent/'objects/terrain/terrain.obj.json').unlink(missing_ok=True)
            
        except Exception as e:
            print(f'Neither meshgrid.npz nor heightmap.npy found in {self.folder}. No terrain will be loaded.')
            raise e

    def load_obj_plans(self):
        try:
            with open(self.folder/'obj_plans.pkl', 'rb') as f:
                self.obj_plans = pickle.load(f)
                print(f'Loaded {len(self.obj_plans)} object plans.')
        except Exception as e:
            self.obj_plans = None
            print(f'obj_plans.pkl not found in {self.folder}. No objects will be loaded.')
            raise e

        try:
            self.world_dimensions = self.obj_plans[0]['world_dimensions']
        except Exception as e:
            self.world_dimensions = np.array([1, 1, 1], dtype=np.float32)
            print(f'No world dimensions found in obj_plans.pkl. Using default dimensions: {self.world_dimensions}')

        # The DefaultOBJ class does not handle the shrinking of the object, so we need to load the objects with the correct scale.
        # Iterate through the object plans and convert the paths to the correct format
        for obj_plan in self.obj_plans:

            print(f"Object plan {obj_plan['id']} loaded, type: {obj_plan['type']}, path shape: {obj_plan['path'].shape} start: {obj_plan['path'][0,1:4]}, dimension: {obj_plan['dimension']}, world dimensions: {obj_plan['world_dimensions']}")

            if np.issubdtype(obj_plan['path'].dtype, np.floating):
                print('Path is of type float, time is already in seconds, path is in SI units, no conversion needed')

                obj_plan['path'][:,0] = obj_plan['path'][:,0] / self.app.clock.FPS_animation  # DEBUG: Make it faster 
                obj_plan['path'][:,1:4] = obj_plan['path'][:,1:4] - self.world_dimensions/2 # center
                obj_plan['path'][:,1:4] = 2 * obj_plan['path'][:,1:4] / np.max(self.world_dimensions).astype(np.float32)  # Normalize the position, by the largest dimension of the world

            if np.issubdtype(obj_plan['path'].dtype, np.integer):
                print('Path is of type int, converting time to seconds from timestep with FPS_animation and path is in grid units, converting to SI units')

                obj_plan['path'] = obj_plan['path'].astype(np.float32)  # Convert entire path to float32 first for consistency
                obj_plan['path'][:,0] = obj_plan['path'][:,0] / self.app.clock.FPS_animation  # Convert time to seconds
                obj_plan['path'][:,1:4] = obj_plan['path'][:,1:4] / np.max(self.world_dimensions).astype(np.float32)  # Normalize the position, by the largest dimension of the world


            # Transform the path to the correct format (OpenGL expects Y and Z to be swapped, and X to be flipped)
            obj_plan['path'][:,1] = -obj_plan['path'][:,1]  # Flip x axis
            obj_plan['path'][:,[2,3]] = obj_plan['path'][:,[3,2]]  # Swap y and z
            if obj_plan['path'].shape[1] >= 7: # Rotation information on indices 4,5,6
                obj_plan['path'][:,4] = - obj_plan['path'][:,4]  # Flip x axis
                obj_plan['path'][:,[5,6]] = obj_plan['path'][:,[6,5]]  # Swap y and z
            if isinstance(obj_plan['dimension'], (tuple, list, np.ndarray)):
                obj_plan['dimension'] = np.array(obj_plan['dimension'])
                obj_plan['dimension'][[1, 2]] = obj_plan['dimension'][[2, 1]]  # Swap y and z dimensions

        # If you want to add a custom object .obj file created by an external program,
        # first open the obj file and comment out lines starting with: 'mtllib' and 'usemtl'
        # because the DefaultOBJ class can not handle these lines.