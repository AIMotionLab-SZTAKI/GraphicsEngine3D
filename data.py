from model import *
from utils import convert_grid_static_to_instancelist, convert_grid_seq_to_instancelist, convert_heightmap_to_terrain_obj, convert_heightmap_to_terrain_texture

import sys;
from pathlib import Path
import numpy as np
import pickle


class Data:
    def __init__(self, app, args:str):
        self.app = app
        folder = args.folder
        scene = args.scene
        print(scene)

        try:
            folder = Path(folder)
        except:
            print(f'Invalid folder path: {folder}. Please provide a valid path.')
            sys.exit(1)

        # grid_seq: np.ndarray with indices: time,x,y,z
        # plans: list of dictionaries with keys: 'path_extracted' ,'path_corrected' ,'path_interp_BSpline', 'path_interp_MinimumSnapTrajectory', etc.
        # heightmap: np.ndarray with shape (x,y) containing the height values

        ### Load grid_static
        if 'grid' in scene or 'all' in scene:
            try:
                grid_static = np.load(folder/'grid_static.npz')['grid_static']
                grid_seq = np.load(folder/'grid_seq.npz')['grid_seq']
                self.grid_shape = grid_static.shape
                print(f'grid_static shape: {grid_static.shape}, grid_seq shape: {grid_seq.shape}')
            except FileNotFoundError:
                if 'grid' in scene: scene.remove('grid')
                print(f'grid_seq.npz OR grid_static.npz not found in {folder}. No grid will be loaded')
                sys.exit(1)
                
            try:
                self.grid_static_instancelist = convert_grid_static_to_instancelist(grid_static)
                self.grid_seq_dynamic_instancelist = convert_grid_seq_to_instancelist(grid_seq)
            except Exception as e:
                print(f'Error during conversion')
                sys.exit(1)


        ### Load plans
        if 'plans' in scene or 'all' in scene:
            try:
                with open(folder/'plans.pkl', 'rb') as f: 
                    self.plans = pickle.load(f)

                    for i, plan in enumerate(self.plans):
                        print(f"Plan {i} keys: {list(plan.keys())}")

                    # IMPLEMENTAION NOT READY YET
                    #if isinstance(self.plans, list):
                    #    self.plans = {i: plan for i, plan in enumerate(self.plans)}
            except:
                self.plans = None
                if 'plans' in scene: scene.remove('plans')
                print('plans.pkl not found in {folder}. No plans will be loaded.')

        ### Load the heightmap and convert to terrain mesh
        if 'terrain' in scene or 'all' in scene:
            try: 
                heightmap = np.load(folder/'heightmap.npy')
                print(f'Heightmap shape: {heightmap.shape}')
                terrain_obj_path = Path(__file__).parent/'objects/terrain/terrain.obj'
                terrain_texture_path = Path(__file__).parent/'objects/terrain/terrain.png'
                convert_heightmap_to_terrain_obj(heightmap, terrain_obj_path, resolution=2/128)
                convert_heightmap_to_terrain_texture(heightmap, terrain_texture_path)
                (Path(__file__).parent/'objects/terrain/terrain.obj.bin').unlink(missing_ok=True)
                (Path(__file__).parent/'objects/terrain/terrain.obj.json').unlink(missing_ok=True)
                
            except FileNotFoundError:
                if 'terrain' in scene: scene.remove('terrain')
                print(f'heightmap.npy not found in {folder}. No terrain will be loaded.')

        
