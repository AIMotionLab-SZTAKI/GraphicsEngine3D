import os, sys
import pygame
import numpy as np
import threading
from datetime import datetime
import argparse
from pathlib import Path
import pickle

sys.path.append(str(Path(__file__).resolve().parents[1]))
from GraphicsEngine3D.utils import getColorMap
from GraphicsEngine3D.clock import Clock

class GraphicsEngine2D():
    def __init__(self, grid_seq:np.ndarray=None, extracted_paths:dict=None, FPS:int=30, FPS_animation:int=30, win_size:tuple=(1000,1000)):
        pygame.init()
        # SCREEN
        self.WIN_SIZE = win_size
        self.cmap = getColorMap()
        self.surf = None
        self.render_event = threading.Event()
        # TIME
        self.clock = Clock()

        self.update_data(grid_seq, extracted_paths)
        
        pygame.init()
        self.display = pygame.display.set_mode(win_size)
        pygame.display.set_caption('GraphicsEngine2D')
        
        # Color palette for agents
        self.agent_colors = self._generate_agent_colors()

    def _generate_agent_colors(self):
        # Generate a color palette for up to 20 agents
        base_colors = [
            (255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255),
            (255,128,0), (128,0,255), (0,128,255), (128,255,0), (255,0,128), (0,255,128),
            (128,128,128), (255,128,128), (128,255,128), (128,128,255), (255,255,128), (255,128,255), (128,255,255), (200,200,200)]
        return base_colors

    def update_data(self, grid_seq:np.ndarray=None, extracted_paths:dict=None):
        if grid_seq is not None:
            self.grid_seq = grid_seq
            self.grid_shape = grid_seq[0].shape
            self.pix = (self.WIN_SIZE[0] / self.grid_shape[0] )  # The size of a single grid square in pixels
            self.font = pygame.freetype.SysFont(None, int(self.pix*6))
            self.paths = extracted_paths if extracted_paths is not None else {} # dict of agent_id : np.ndarray (shape: [t,x,y])
            self.clock.time_animation = 0
            self.render_event.set()
        else:
            self.render_event.clear()

    def draw_agent_path(self, surf, path, color): # path: np.ndarray of shape [t,x,y]
        for i in range(path.shape[0]-1):
            pygame.draw.line(surf, color,
                ((path[i][0] + 0.5)*self.pix, (self.grid_shape[1]-path[i][1]-0.5)*self.pix),
                ((path[i+1][0]+0.5)*self.pix, (self.grid_shape[1]-path[i+1][1]-0.5)*self.pix), width=4)
            # Draw a small dot on the path if the agent has waited
            if np.all(path[i] == path[i+1]):
                pygame.draw.circle(surf, color,
                    ((path[i][0]+0.5)*self.pix, (self.grid_shape[1]-(path[i][1]+0.5))*self.pix), self.pix/2)

    def draw_agent(self, surf, pos, color, agent_id=None): # pos: (x, y)
        pygame.draw.circle(surf, color,
            ((pos[0]+0.5)*self.pix, (self.grid_shape[1]-(pos[1]+0.5))*self.pix), self.pix*2)
        # Draw agent id as text at the agent's location
        if agent_id is not None:
            text_surf, _ = self.font.render(str(agent_id), (0,0,0))
            text_rect = text_surf.get_rect(center=((pos[0]+0.5)*self.pix, (self.grid_shape[1]-(pos[1]+0.5))*self.pix))
            surf.blit(text_surf, text_rect)

    def check_events(self):
        for event in pygame.event.get():
            if  event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.display.quit()
                pygame.quit()
                sys.exit()
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:   # R: Reset animation
                self.clock.time_animation = 0
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:   # P: Screenshot
                 self.take_screenshot()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.clock.paused = not self.clock.paused
            elif self.clock.paused and event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.clock.time_animation = self.clock.time_animation + 1 / self.clock.FPS_animation
            elif self.clock.paused and event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.clock.time_animation = self.clock.time_animation - 1 / self.clock.FPS_animation

    def take_screenshot(self, folder='Screenshots'):
        folder = Path(__file__).parent / folder
        os.makedirs(folder, exist_ok=True)
        pygame.image.save(self.surf,f'{folder}/GraphicsEngine2D_{datetime.today().strftime('%Y%m%d_%H%M%S')}.png')
    
    def render(self):
        self.clock.frame_index = min(int(self.clock.time_animation * self.clock.FPS_animation), self.grid_seq.shape[0]-1)
        surf_array = self.grid_seq[self.clock.frame_index]
        surf_array = np.where(surf_array == np.inf, 1, surf_array)
        self.surf = pygame.surfarray.make_surface((np.fliplr(surf_array*255).astype(int)))
        self.surf = pygame.transform.scale(self.surf,self.WIN_SIZE)
        self.surf.set_palette(self.cmap)
        self.surf = self.surf.convert_alpha()
        # Draw all agent paths and agents
        for idx, (agent_id, path) in enumerate(self.paths.items()):
            if path.size > 0:
                path = path[:,1:]  # Remove time dimension
                color = self.agent_colors[idx % len(self.agent_colors)]
                self.draw_agent_path(self.surf, path, color)
                # Draw agent at current position
                frame_index_path = min(self.clock.frame_index, path.shape[0]-1)
                self.draw_agent(self.surf, path[frame_index_path], color, agent_id=agent_id)
        self.display.blit(self.surf,(0,0))   # (0,0) -> center in window
        pygame.display.update()
        self.clock.clock.tick(self.clock.FPS)
        if self.clock.frame_index > self.clock.frame_index_prev:
            pass
            #pygame.image.save(self.surf,f'Screenshots/GraphicsEngine2D_{(self.time*1000):.0f}.png')

    def run(self):
        while True:
            self.render_event.wait()
            self.clock.update_time()
            self.check_events()
            if self.render_event.is_set():
                self.render()
            self.clock.update_delta_time()
            self.clock.update_caption()

def main():
    root_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="GraphicsEngine 2D for plotting path planning results")
    parser.add_argument("--folder", type=str, default=root_dir/"demo/demo_2D", help="Folder containing the grid_seq.npy and paths_dict.pkl")
    args = parser.parse_args()
    folder = Path(args.folder)

    grid_seq = np.load(folder/"grid_seq.npy", allow_pickle=True)
    with open(folder / "paths_dict.pkl", "rb") as f:
        paths = pickle.load(f)

    print("grid_seq shape:", grid_seq.shape)
    for k, v in paths.items():
        print(f"Agent {k}: path shape {v.shape}")

    ge = GraphicsEngine2D(grid_seq, paths)
    ge.run()

if __name__ == "__main__":
    main()