import pygame as pg

class Clock:
    def __init__(self):
        self.clock = pg.time.Clock()
        self.time = 0
        self.delta_time = 0
        self.time_animation = 0

        self.FPS = 30
        self.FPS_animation = 15
        self.delay = 0.3

        self.paused = False
        self.frame_index = 0
        self.frame_index_prev = -1

    def update_time(self):
        self.time_prev = self.time
        self.time = pg.time.get_ticks() * 0.001 # sec
        if not self.paused:
            self.time_animation += self.time - self.time_prev

    def update_delta_time(self):
        self.delta_time = self.clock.tick(self.FPS)

    def get_FPS(self):
        return self.clock.get_fps()
    
    def update_caption(self):
        pg.display.set_caption(f'GraphicsEngine '\
                                f'Time: {self.time:.2f}s, '\
                                f'Animation time: {self.time_animation:.2f}s, '\
                                f'Frame: {self.frame_index}, '\
                                f'FPS: {self.get_FPS():.0f}') 
        # camera.get_camera_str()