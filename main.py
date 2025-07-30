import pygame as pg
import moderngl as mgl
import os, sys
from datetime import datetime
import argparse
from pathlib import Path

from clock import Clock
from model import *
from camera import Camera
from light import Light
from mesh import Mesh
from scene import Scene
from data import Data
from info_display import InfoDisplay

class GraphicsEngine:
    def __init__(self, win_size=(1920, 1080), **kwargs):
        self.config = kwargs
        self.WIN_SIZE = win_size
        # init pygame modules
        pg.init()
        # set opengl attr
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        # create opengl context
        pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF)
        # mouse settings
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)
        # detect and use existing opengl context
        self.ctx = mgl.create_context()
        # self.ctx.front_face = 'cw'
        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)

        self.clock  = Clock()
        self.data   = Data(self)
        self.light  = Light()
        self.camera = Camera(self, position=(2,0.5,0), yaw=180, pitch=-20)
        self.mesh   = Mesh(self)
        self.scene  = Scene(self, self.data.scene)
        self.info_display = InfoDisplay(self)

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.mesh.destroy()
                self.info_display.destroy()
                pg.display.quit()
                pg.quit()
                sys.exit()
                return
            elif (event.type == pg.ACTIVEEVENT and event.gain == 1 and event.state == 2): # Window regains focus
                pg.event.set_grab(True)
                pg.mouse.set_visible(False)
            elif (event.type == pg.KEYDOWN and event.key == pg.K_r): # R: Reset animation
                self.clock.time_animation = 0
            elif (event.type == pg.KEYDOWN and event.key == pg.K_p): # P: Screenshot
                self.take_screenshot()
            elif (event.type == pg.KEYDOWN and event.key == pg.K_SPACE): # Pause animation
                self.clock.paused = not self.clock.paused
            elif (self.clock.paused and event.type == pg.KEYDOWN and event.key == pg.K_RIGHT):
                self.clock.time_animation = self.clock.time_animation + 1 / self.clock.FPS_animation
            elif (self.clock.paused and event.type == pg.KEYDOWN and event.key == pg.K_LEFT):
                self.clock.time_animation = self.clock.time_animation - 1 / self.clock.FPS_animation
            elif (event.type == pg.KEYDOWN and event.key == pg.K_i): # I: Interpolate camera movement
                self.camera.camera_interp = not self.camera.camera_interp
            elif (event.type == pg.KEYDOWN and event.key == pg.K_h): # H: Toggle info display
                self.info_display.toggle_visibility()

    def render(self):
        self.ctx.clear(color=(0.08, 0.16, 0.18))    # clear framebuffer
        self.scene.render()                         # render scene
        self.info_display.render()                  # render info overlay
        pg.display.flip()                           # swap buffers

    def run(self):
        while True:
            self.clock.update_time()
            self.check_events()
            self.camera.update()
            self.render()
            self.clock.update_delta_time()
            self.clock.update_caption()

    def take_screenshot(self, folder='screenshots'):
        folder = Path(__file__).parent / folder
        data = self.ctx.screen.read()   # Read the pixel data from he framebuffer
        image = pg.image.frombytes(data, self.WIN_SIZE, 'RGB',True)
        os.makedirs(folder, exist_ok=True)
        pg.image.save(image,f'{folder}/GraphicsEngine3D_{datetime.today().strftime("%Y%m%d_%H%M%S")}.png')

def main():
    root_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description='GraphicsEngine3D')
    parser.add_argument('--folder', type=str, default=root_dir/'demo/demo_Mate2',
                        help='Folder containing the necessary files')
    parser.add_argument('--scene', type=eval, default=['all'],
                        help='''List containing scene objects to be loaded ['all', 'grid', 'plans', 'terrain', 'obj']''')
    args = parser.parse_args()
    kwargs = vars(args)  # Convert Namespace to dict

    app = GraphicsEngine(**kwargs)
    app.run()

if __name__ == '__main__':
    main()

# python GraphicsEngine3D/main.py --scene ['plans','terrain', 'obj']