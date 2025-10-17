import glm


class Light:
    def __init__(self, app, position=(10, 30, 10)):  # (global x,y,z: grid -x,z,-y)
        self.position = glm.vec3(app.config.get('light.position', position))
        self.color = glm.vec3(app.config.get('light.color', (1, 1, 1)))
        # intensities
        self.Ia = app.config.get('light.ambient_intensity', 0.06) * self.color  # ambient
        self.Id = app.config.get('light.diffuse_intensity', 0.8) * self.color   # diffuse
        self.Is = app.config.get('light.specular_intensity', 1.0) * self.color  # specular