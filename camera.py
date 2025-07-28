import glm
import pygame as pg
import numpy as np
from scipy.interpolate import CubicSpline

FOV = 50  # deg
NEAR = 0.1
FAR = 100
SPEED = 0.001/2
SENSITIVITY = 0.04


class Camera:
    def __init__(self, app, position=(0, 0, 3), yaw=-90, pitch=0):
        self.app = app
        self.aspect_ratio = app.WIN_SIZE[0] / app.WIN_SIZE[1]
        self.position = glm.vec3(position)
        self.up = glm.vec3(0, 1, 0)
        self.right = glm.vec3(1, 0, 0)
        self.forward = glm.vec3(0, 0, -1)
        self.yaw = yaw
        self.pitch = pitch
        # view matrix
        self.m_view = self.get_view_matrix()
        # projection matrix
        self.m_proj = self.get_projection_matrix()

        self.camera_interp = False # press key 'i' to activate

    def rotate(self):
        rel_x, rel_y = pg.mouse.get_rel()
        self.yaw += rel_x * SENSITIVITY
        self.pitch -= rel_y * SENSITIVITY
        self.pitch = max(-89, min(89, self.pitch))

    def update_camera_vectors(self):
        yaw, pitch = glm.radians(self.yaw), glm.radians(self.pitch)

        self.forward.x = glm.cos(yaw) * glm.cos(pitch)
        self.forward.y = glm.sin(pitch)
        self.forward.z = glm.sin(yaw) * glm.cos(pitch)

        self.forward = glm.normalize(self.forward)
        self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 1, 0)))
        self.up = glm.normalize(glm.cross(self.right, self.forward))

    def update(self):
        self.move()
        self.rotate()
        if self.camera_interp: self.interpolate()
        self.update_camera_vectors()
        self.m_view = self.get_view_matrix()

    def move(self):
        velocity = SPEED * self.app.clock.delta_time
        keys = pg.key.get_pressed()
        if keys[pg.K_w]:
            self.position += self.forward * velocity
        if keys[pg.K_s]:
            self.position -= self.forward * velocity
        if keys[pg.K_a]:
            self.position -= self.right * velocity
        if keys[pg.K_d]:
            self.position += self.right * velocity
        if keys[pg.K_q]:
            self.position += self.up * velocity
        if keys[pg.K_e]:
            self.position -= self.up * velocity

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.forward, self.up)

    def get_projection_matrix(self):
        return glm.perspective(glm.radians(FOV), self.aspect_ratio, NEAR, FAR)
    
    def get_camera_str(self):
        return f'p: ({self.position.x:.2f}, {self.position.y:.1f}, {self.position.z:.1f}), '\
               f'yaw: {self.yaw:.0f}deg {np.radians(self.yaw):.2f}rad, pitch: {self.pitch:.0f}deg {np.radians(self.pitch):.2f}rad'

    def get_camera_interp_data(self):

        positions = np.array([ # t,x,y,z
            [0,   1.54, 1, 0], 
            [0.5, 1.08, 1, -1.13],
            [1,   0,    1, -1.6]])
        
        yaws = np.array([
            [0, 180],
            [1, 90]])
        
        pitches = np.array([
            [0, -41],
            [1, -38]])

        self.camera_interp_data = interpolate_camera_movement(positions, yaws, pitches)


        


    def interpolate(self):
        if not hasattr(self, 'camera_interp_data'):
            self.get_camera_interp_data()

        t_end = 20
        t = min(1.1, self.app.clock.time_animation/t_end)

        self.position = glm.vec3(self.camera_interp_data["position"](t))
        self.yaw = self.camera_interp_data["yaw"](t)
        self.pitch = self.camera_interp_data["pitch"](t)



def interpolate_camera_movement(positions, yaws, pitches):
    """
    Interpolates camera movement using cubic splines for smooth transitions.
    """
    # Ensure inputs are sorted by time
    positions = positions[np.argsort(positions[:, 0])]
    yaws = yaws[np.argsort(yaws[:, 0])]
    pitches = pitches[np.argsort(pitches[:, 0])]

    # Extract time and values
    time_pos = positions[:, 0]
    x, y, z = positions[:, 1], positions[:, 2], positions[:, 3]
    time_yaw = yaws[:, 0]
    yaw_values = yaws[:, 1]
    time_pitch = pitches[:, 0]
    pitch_values = pitches[:, 1]

    # Create cubic splines for each component
    spline_x = CubicSpline(time_pos, x)
    spline_y = CubicSpline(time_pos, y)
    spline_z = CubicSpline(time_pos, z)
    spline_yaw = CubicSpline(time_yaw, yaw_values)
    spline_pitch = CubicSpline(time_pitch, pitch_values)

    return {"position": lambda t: np.array([spline_x(t), spline_y(t), spline_z(t)]),
            "yaw": spline_yaw,
            "pitch": spline_pitch}












