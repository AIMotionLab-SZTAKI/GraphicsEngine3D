import moderngl as mgl
import numpy as np
import glm
from pathlib import Path

from OpenGL.GL import *

from vbo import CubeVBO, CubeStaticInstanceVBO, CubeDynamicInstanceVBO, SplineVBO, CoordSysVBO, DefaultSTL_VBO, DefaultOBJ_VBO

class BaseModel:
    def __init__(self, app, vao_name, tex_id, pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), coord_sys=None, **kwargs):
        self.app = app
        self.pos = glm.vec3(pos)
        self.rot = glm.vec3([glm.radians(a) for a in rot])
        self.scale = glm.vec3(tuple(np.broadcast_to(np.array(scale), 3).astype(float)))
        self.coord_sys = coord_sys
        self.m_model = self.get_model_matrix()
        self.tex_id = tex_id
        self.vao = app.mesh.vao.vaos[vao_name]
        self.program = self.vao.program
        self.camera = self.app.camera

        self.initial_pos = self.pos
        self.initial_rot = self.rot
        self.initial_scale = self.scale
        
    def update(self): ...

    def get_model_matrix(self):
        m_model = glm.mat4()
        # translate
        m_model = glm.translate(m_model, self.pos)
        # rotate
        m_model = glm.rotate(m_model, self.rot.z, glm.vec3(0, 0, 1))
        m_model = glm.rotate(m_model, self.rot.y, glm.vec3(0, 1, 0))
        m_model = glm.rotate(m_model, self.rot.x, glm.vec3(1, 0, 0))
        # scale
        m_model = glm.scale(m_model, self.scale)

        if self.coord_sys is not None:
            m_model = self.coord_sys * m_model

        return m_model
    
    def get_instance_matrix(self):
        """Calculate the instance transformation matrix (local coordinate system)"""
        m_instance = glm.mat4()
        # translate
        m_instance = glm.translate(m_instance, self.instance_pos)
        # rotate
        m_instance = glm.rotate(m_instance, self.instance_rot.z, glm.vec3(0, 0, 1))
        m_instance = glm.rotate(m_instance, self.instance_rot.y, glm.vec3(0, 1, 0))
        m_instance = glm.rotate(m_instance, self.instance_rot.x, glm.vec3(1, 0, 0))
        # scale
        m_instance = glm.scale(m_instance, self.instance_scale)
        return m_instance

    def update_uniform_light(self, specular=True):
        self.program['light.position'].write(self.app.light.position)
        self.program['light.Ia'].write(self.app.light.Ia)
        self.program['light.Id'].write(self.app.light.Id)
        if specular: self.program['light.Is'].write(self.app.light.Is)

    def update_uniform_transformation_matrices(self, update_instance=False, update_proj=False, update_camPos=False):
        ''' Update the instance-model-view-projection matrices in the shader '''
        if update_instance: self.program['m_instance'].write(self.get_instance_matrix())  # Add instance matrix
        self.program['m_model'].write(self.m_model)
        self.program['m_view'].write(self.camera.m_view)
        if update_proj: self.program['m_proj'].write(self.camera.m_proj)
        if update_camPos: self.program['camPos'].write(self.camera.position)

    def update_instance_transform_parameters_for_normalization(self, vbo_name):
        if any(k in self.kwargs for k in ['normalize_instance_dimensions', 'center_instance']):
            vertex_data = self.app.mesh.vao.vbo.vbos[vbo_name].vertex_data
            vertex_data = vertex_data.reshape(-1, 8)         # Reshape the vertex data into a (N, 8) array, format: [texcoord, normal, position]
            vertex_positions = vertex_data[:, 5:8]           # Extract the position data
            min_coords = np.min(vertex_positions, axis=0)    # Calculate the min and max for each axis (x, y, z)
            max_coords = np.max(vertex_positions, axis=0)
            bbox = max_coords - min_coords                   # Calculate the bounding box dimensions

            if 'normalize_instance_dimensions' in self.kwargs:
                # Normalize the instance dimensions based on the bounding box
                if self.kwargs['normalize_instance_dimensions'] == 'max':
                    self.instance_scale = float(1 / np.max(bbox)) # Keep aspect ratio, divide by the largest dimension
                elif self.kwargs['normalize_instance_dimensions'] == 'unit':
                    self.instance_scale = 1 / bbox                # Rescale to unit cube
                else:
                    raise ValueError("Invalid value for 'normalize_instance_dimensions'. Use 'max' or 'unit'.")
            
            if 'center_instance' in self.kwargs:
                # Center the instance first, before normalization
                self.instance_pos = - (min_coords + max_coords) / 2

        self.instance_pos = glm.vec3(self.instance_pos * self.instance_scale + self.instance_pos_init)
        self.instance_scale = glm.vec3(self.instance_scale * self.instance_scale_init)
        
    def render(self):
        self.update()
        self.vao.render()

class DefaultOBJ(BaseModel):
    def __init__(self, app, vao_name='default', vbo_name='default', tex_id='test', 
                 path_obj='objects/terrain/terrain.obj', path_texture='objects/terrain/terrain.png',
                 path:np.ndarray=None, rotation_available:bool=False,
                 pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1),
                 instance_pos=(0, 0, 0), instance_rot=(0, 0, 0), instance_scale=(1, 1, 1),
                 coord_sys=None, **kwargs):
        '''kwargs:
        - alpha: float, default 1.0, transparency of the object
        - normalize_instance_dimensions: str, 'max' or 'unit'
        - center_instance: bool, default False, whether to center the object (using its bounding box) before loading it to the vbo
        '''

        # Add vbo if it does not exist
        if not vbo_name in app.mesh.vao.vbo.vbos: # Only create the VBO once
            app.mesh.vao.vbo.vbos[vbo_name] = DefaultOBJ_VBO(app.ctx, file=path_obj)

        # Add texture if tex_id != 'test' and it does not exist already
        if tex_id != 'test' and tex_id not in app.mesh.texture.textures:
            app.mesh.texture.textures[tex_id] = app.mesh.texture.get_texture(path=Path(__file__).parent/path_texture)  # Load texture

        # Init PROGRAM and VAO before calling super().init !!! (vao_name = program_name)
        app.mesh.vao.program.programs[vao_name] = app.mesh.vao.program.get_program('OBJ_default')
        app.mesh.vao.vaos[vao_name] = app.mesh.vao.get_vao(program = app.mesh.vao.program.programs['OBJ_default'],
                                                           vbo = [app.mesh.vao.vbo.vbos[vbo_name]])
    
        self.vao_name = vao_name
        self.path = path
        self.rot_available = rotation_available
        self.kwargs = kwargs

        # Instance transformation parameters
        self.instance_pos = glm.vec3(instance_pos)
        self.instance_rot = glm.vec3([glm.radians(a) for a in instance_rot])
        self.instance_scale = glm.vec3(tuple(np.broadcast_to(np.array(instance_scale), 3).astype(float)))
        self.instance_pos_init = self.instance_pos
        self.instance_rot_init = self.instance_rot
        self.instance_scale_init = self.instance_scale

        super().__init__(app, vao_name, tex_id, pos, rot, scale, coord_sys)
        self.update_instance_transform_parameters_for_normalization(vbo_name)
        self.on_init()

    def on_init(self):
        self.update_model_transform()

        # texture
        self.program['alpha'] = self.kwargs['alpha'] if 'alpha' in self.kwargs else 1.0
        self.texture = self.app.mesh.texture.textures[self.tex_id]
        self.program['u_texture_0'] = 0
        self.texture.use(location = 0)

        self.update_uniform_transformation_matrices(update_instance=True, update_proj=True)
        self.update_uniform_light()

    def update(self):
        self.update_model_transform()
        self.update_uniform_transformation_matrices(update_instance=True, update_camPos=True)
        self.program['alpha'] = self.kwargs['alpha'] if 'alpha' in self.kwargs else 1.0
        self.texture.use()
        
        self.app.info_display.update_info_section(self.vao_name,f'{self.vao_name} pos: {self.pos}, rot: {self.rot}, scale: {self.scale}, ipos: {self.instance_pos}, irot: {self.instance_rot}, iscale: {self.instance_scale}')

    def get_pos(self):
        closest_time_idx = np.argmin(np.abs(self.path[:, 0] - self.app.clock.time_animation))
        path_pos = glm.vec3(self.path[closest_time_idx, 1:4])
        return path_pos + self.initial_pos  # Add initial position as offset

    def get_rot(self):
        closest_time_idx = np.argmin(np.abs(self.path[:, 0] - self.app.clock.time_animation))
        path_rot = glm.vec3(self.path[closest_time_idx, 4:7])  # Assuming the rotation is in radians
        return path_rot + self.initial_rot  # Add initial rotation as offset
    
    def update_model_transform(self):
        if self.path is not None:
            self.pos = self.get_pos()
            if self.rot_available:
                self.rot = self.get_rot()
            self.m_model = self.get_model_matrix()


class CubeStatic(BaseModel):
    def __init__(self, app, vao_name='cubeStatic', tex_id='cube', pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), **kwargs):

        if not 'cube' in app.mesh.vao.vbo.vbos:  # The same VBO is used for all cubes, so it is created only once
            app.mesh.vao.vbo.vbos['cube'] = CubeVBO(app.ctx)
        if not 'cubeStaticInstance' in app.mesh.vao.vbo.vbos:
            app.mesh.vao.vbo.vbos['cubeStaticInstance'] = \
                CubeStaticInstanceVBO(app.ctx, reserve = app.data.grid_static_instancelist.shape[0] * 16) # 4f ~16 byte # not updated at all

        app.mesh.vao.vaos['cubeStatic'] = app.mesh.vao.get_vao(
            program = app.mesh.vao.program.programs['cubeStatic'],
            vbo = [app.mesh.vao.vbo.vbos['cube'],
                   app.mesh.vao.vbo.vbos['cubeStaticInstance']],
            ibo = app.mesh.vao.ibo.ibos['cube'])

        super().__init__(app, vao_name, tex_id, pos, rot, scale, **kwargs)
        self.on_init()

    def update(self):
        self.update_uniform_transformation_matrices(update_proj=True)

    def on_init(self):
        self.grid_static_instancelist = self.app.data.grid_static_instancelist
        self.program['shape'].write(glm.ivec3(*self.app.data.grid_shape))   # uniform variable
        self.app.mesh.vao.vbo.vbos['cubeStaticInstance'].vbo.write(np.array(self.grid_static_instancelist.flatten()).astype('f4'))

        self.update_uniform_transformation_matrices()
        self.update_uniform_light(specular=False)
        
    def render(self):
        self.update()
        self.vao.render(instances = np.prod(self.grid_static_instancelist.shape[0]))

class CubeDynamic(BaseModel):
    def __init__(self, app, vao_name='cubeDynamic', tex_id='cube', pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), **kwargs):

        if not 'cube' in app.mesh.vao.vbo.vbos:  # The same VBO is used for all cubes, so it is created only once
            app.mesh.vao.vbo.vbos['cube'] = CubeVBO(app.ctx)
        if not 'cubeDynamicInstance' in app.mesh.vao.vbo.vbos:
            app.mesh.vao.vbo.vbos['cubeDynamicInstance'] = \
                CubeDynamicInstanceVBO(app.ctx, reserve = app.data.grid_seq_dynamic_instancelist.shape[1] * 16) # 4f ~16 byte # updated with indices

        app.mesh.vao.vaos['cubeDynamic'] = app.mesh.vao.get_vao(
            program = app.mesh.vao.program.programs['cubeDynamic'],
            vbo = [app.mesh.vao.vbo.vbos['cube'],
                   app.mesh.vao.vbo.vbos['cubeDynamicInstance']],
            ibo = app.mesh.vao.ibo.ibos['cube'])

        super().__init__(app, vao_name, tex_id, pos, rot, scale, **kwargs)
        self.on_init()

    def update(self):
        # grid_seq
        frame_index_prev = self.frame_index
        self.frame_index = min(int(self.app.clock.time_animation * self.app.clock.FPS_animation), self.grid_seq_dynamic_instancelist.shape[0] - 1)
        if self.frame_index != frame_index_prev:
            self.frame = self.grid_seq_dynamic_instancelist[self.frame_index]
            self.app.mesh.vao.vbo.vbos['cubeDynamicInstance'].vbo.write(np.array(self.frame.flatten()).astype('f4'))
        
        self.update_uniform_transformation_matrices(update_proj=True, update_camPos=True)

    def on_init(self):

        # grid_seq dynamic
        self.grid_seq_dynamic_instancelist = self.app.data.grid_seq_dynamic_instancelist
        self.program['shape'].write(glm.ivec3(*self.app.data.grid_shape))   # uniform variable
        self.frame_index = 0
        self.frame = self.grid_seq_dynamic_instancelist[self.frame_index]
        self.app.mesh.vao.vbo.vbos['cubeDynamicInstance'].vbo.write(np.array(self.frame.flatten()).astype('f4'))   

        self.update_uniform_transformation_matrices(update_camPos=True)
        self.update_uniform_light(specular=False)

        
    def render(self):
        self.update()
        self.vao.render(instances = self.grid_seq_dynamic_instancelist.shape[1])

class Spline(BaseModel):
    def __init__(self, app, vao_name='spline',
                 pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), 
                 path:np.ndarray=np.zeros(4), color=[1,1,1,1], **kwargs):
        
        # Init VBO and VAO before calling super().init !!! (vbo_name = vao_name)
        app.mesh.vao.vbo.vbos[vao_name] = SplineVBO(app.ctx, reserve = path.shape[0]*12)
        app.mesh.vao.vaos[vao_name] = app.mesh.vao.get_vao(program = app.mesh.vao.program.programs['spline'], 
                                                           vbo = [app.mesh.vao.vbo.vbos[vao_name]])
        
        super().__init__(app, vao_name, None, pos, rot, scale, **kwargs)

        self.app = app
        self.vao_name = vao_name
        self.path = path
        self.color = glm.vec4(color)
        self.on_init()

    def on_init(self):
        path = self.path[:, 1:4]  # Use the path as it is, without normalization
        self.app.mesh.vao.vbo.vbos[self.vao_name].vbo.write((path).flatten().astype('f4'))
        self.update_uniform_transformation_matrices(update_proj=True)

    def update(self):
        self.program['color'].write(self.color) # Color is updated sequentually, since a custom shader object would have been needed to make it static
        self.update_uniform_transformation_matrices()

    def render(self):
        self.update()
        self.app.ctx.line_width = 4
        self.vao.render(mgl.LINE_STRIP)
        self.app.ctx.line_width = 1 # RESET

class CoordSys(BaseModel):
    def __init__(self, app, vao_name='coordsys', tex_id='None', pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), **kwargs):
        self.data=np.array([0,0,0,1,0,0,1,  1,0,0,1,0,0,1,
                            0,0,0,0,1,0,1,  0,1,0,0,1,0,1,
                            0,0,0,0,0,1,1,  0,0,1,0,0,1,1])
        
        # Init VBO and VAO before calling super().init !!! (vbo_name = vao_name)
        app.mesh.vao.vbo.vbos[vao_name] = CoordSysVBO(app.ctx, reserve = self.data.shape[0]*28)
        app.mesh.vao.vaos[vao_name] = app.mesh.vao.get_vao(program = app.mesh.vao.program.programs['coordsys'],
                                                           vbo = [app.mesh.vao.vbo.vbos[vao_name]])
        super().__init__(app, vao_name, tex_id, pos, rot, scale, **kwargs)

        self.vao_name = vao_name

        self.on_init()

    def update(self):
        self.update_uniform_transformation_matrices()

    def on_init(self):           
        self.app.mesh.vao.vbo.vbos[self.vao_name].vbo.write(self.data.astype('f4'))
        self.update_uniform_transformation_matrices(update_proj=True)

    def render(self):
        self.update()
        self.app.ctx.line_width = 3
        self.vao.render(mgl.LINES)
        self.app.ctx.line_width = 1 # RESET

class DefaultSTL(BaseModel):
    def __init__(self, app, vao_name='defaultSTL', vbo_name='defaultSTL', tex_id='test',
                 path_stl='objects/drone/quad.stl',
                 path:np.ndarray=None, rotation_available:bool=False,
                 pos=(0, 0, 0), rot=(0, 180, 0), scale=(1, 1, 1), **kwargs):
        
        # Add vbo if it does not exist
        if not vbo_name in app.mesh.vao.vbo.vbos: # The same VBO is used for all drones, so it is created only once
            app.mesh.vao.vbo.vbos[vbo_name] = DefaultSTL_VBO(app.ctx, file=path_stl)

        # Init PROGRAM and VAO before calling super().init !!! (vao_name = program_name)
        app.mesh.vao.program.programs[vao_name] = app.mesh.vao.program.get_program('STL_default')
        app.mesh.vao.vaos[vao_name] = app.mesh.vao.get_vao(program = app.mesh.vao.program.programs['STL_default'],
                                                           vbo = [app.mesh.vao.vbo.vbos[vbo_name]])

        self.vao_name = vao_name
        self.path = path
        self.rot_available = rotation_available
        self.kwargs = kwargs

        super().__init__(app, vao_name, tex_id, pos, rot, scale, **kwargs)
        #self.update_instance_transform_parameters_for_normalization(vbo_name)
        self.on_init()

    def on_init(self):
        self.update_model_transform()
        self.update_uniform_transformation_matrices(update_proj=True)
        self.update_uniform_light()

    def update(self):
        self.update_model_transform()
        self.update_uniform_transformation_matrices(update_camPos=True)

        self.app.info_display.update_info_section(self.vao_name,f'{self.vao_name} pos: {self.pos}, rot: {self.rot}, scale: {self.scale}')

    def get_pos(self):
        closest_time_idx = np.argmin(np.abs(self.path[:, 0] - self.app.clock.time_animation))
        path_pos = glm.vec3(self.path[closest_time_idx, 1:4])
        return path_pos + self.initial_pos  # Add initial position as offset

    def get_rot(self):
        closest_time_idx = np.argmin(np.abs(self.path[:, 0] - self.app.clock.time_animation))
        path_rot = glm.vec3(self.path[closest_time_idx, 4:7])  # Assuming the rotation is in radians
        return path_rot + self.initial_rot  # Add initial rotation as offset

    def update_model_transform(self):
        if self.path is not None:
            self.pos = self.get_pos()
            if self.rot_available:
                self.rot = self.get_rot()
            self.m_model = self.get_model_matrix()

    