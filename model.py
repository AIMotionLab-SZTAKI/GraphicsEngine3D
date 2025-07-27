import moderngl as mgl
import numpy as np
import glm
from pathlib import Path

from OpenGL.GL import *

from vbo import CubeVBO, CubeStaticInstanceVBO, CubeDynamicInstanceVBO, SplineVBO, CoordSysVBO, DefaultSTL_VBO, DefaultOBJ_VBO

class BaseModel:
    def __init__(self, app, vao_name, tex_id, pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
        self.app = app
        self.pos = glm.vec3(pos)
        self.rot = glm.vec3([glm.radians(a) for a in rot])
        self.scale = glm.vec3(tuple(np.broadcast_to(np.array(scale), 3).astype(float)))
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
        return m_model

    def update_uniform_light(self, specular=True):
        self.program['light.position'].write(self.app.light.position)
        self.program['light.Ia'].write(self.app.light.Ia)
        self.program['light.Id'].write(self.app.light.Id)
        if specular: self.program['light.Is'].write(self.app.light.Is)

    def update_uniform_transformation_matrices(self, update_proj=False, update_camPos=False):
        ''' Update the model-view-projection matrices in the shader '''
        self.program['m_model'].write(self.m_model)
        self.program['m_view'].write(self.camera.m_view)
        if update_proj: self.program['m_proj'].write(self.camera.m_proj)
        if update_camPos: self.program['camPos'].write(self.camera.position)

    def set_offset_parameters_for_dimension_normalization(self, vertex_data):        
        vertex_data = vertex_data.reshape(-1, 3)    # Reshape the vertex data into a (N, 3) array
        min_coords = np.min(vertex_data, axis=0)    # Calculate the min and max for each axis (x, y, z)
        max_coords = np.max(vertex_data, axis=0)

        self.offset_scale = float(1 / np.max(max_coords - min_coords))  # Scale factor to fit within a unit cube

    def update_initial_transform_parameters_for_dimension_normalization(self, vbo_name, normalize_dimensions=False):
        if normalize_dimensions:
            # set scale for dimension normalization
            self.set_offset_parameters_for_dimension_normalization(self.app.mesh.vao.vbo.vbos[vbo_name].vertex_data)
        else:
            self.offset_scale = 1

        self.initial_scale = glm.vec3(tuple(np.broadcast_to(self.offset_scale * np.array(self.initial_scale), 3).astype(float)))

        self.pos = glm.vec3(self.initial_pos)
        self.rot = glm.vec3(self.initial_rot)
        self.scale = glm.vec3(self.initial_scale)

    def render(self):
        self.update()
        self.vao.render()

class DefaultOBJ(BaseModel):
    def __init__(self, app, vao_name='default', vbo_name='default', tex_id='test', 
                 path_obj='objects/terrain/terrain.obj', path_texture='objects/terrain/terrain.png',
                 path:np.ndarray=None, rotation_available:bool=False, 
                 normalize_dimensions:bool=False, center_obj:bool=False,
                 pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
        
        # Center the obj file before loading it to the vbo
        if center_obj and not vbo_name in app.mesh.vao.vbo.vbos:
            if (Path(__file__).parent/path_obj).with_suffix('.obj.bin').exists():
                print('The OBJ file centering is only executed if a new .obj file is added or updated. ' \
                      'Delete the cache files if the .obj file is updated.')
            else:
                from utils import center_obj_file
                center_obj_file(path=Path(__file__).parent/path_obj)
                print(f'OBJ file {path_obj} centered successfully.')

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
    
        self.path = path
        self.rot_available = rotation_available

        super().__init__(app, vao_name, tex_id, pos, rot, scale)
        self.update_initial_transform_parameters_for_dimension_normalization(vbo_name, normalize_dimensions)
        self.on_init()

    def on_init(self):
        self.update_model_transform()

        # texture
        self.program['alpha'] = 1
        self.texture = self.app.mesh.texture.textures[self.tex_id]
        self.program['u_texture_0'] = 0
        self.texture.use(location = 0)

        self.update_uniform_transformation_matrices(update_proj=True)
        self.update_uniform_light()
        self.program['light.Ia'].write(self.app.light.Ia+0.2)

    def update(self):
        self.update_model_transform()
        self.update_uniform_transformation_matrices(update_camPos=True)
        self.texture.use()
        
        self.app.info_display.update_info_section(self.tex_id,f'{self.tex_id} pos: {self.pos}, rot: {self.rot}, scale: {self.scale}')

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

class DroneOBJ(DefaultOBJ):
    def __init__(self, app, vao_name='drone',
                 pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), plan:dict=None):
        
        self.plan = plan
        self.vao_name = vao_name

        super().__init__(app, vao_name=vao_name, vbo_name='drone', tex_id='drone',
                         path_obj='objects/drone/MQ-9.obj',
                         path_texture='objects/drone/MQ-9_Black.png',
                         normalize_dimensions=True,
                         pos=pos, rot=rot, scale=scale)

        self.on_init()

    def on_init(self):
        if 'path_interp_MinimumSnapTrajectory' in self.plan:
            self.path = self.plan['path_interp_MinimumSnapTrajectory'] # t,x,y,z,rotx,roty,rotz
            self.rot_available = True
        elif 'path_interp_BSpline' in self.plan:
            self.path = self.plan['path_interp_BSpline']  # t,x,y,z,rotx,roty,rotz
            self.rot_available = True
        else:
            self.path = self.plan['path_extracted'] # t,x,y,z
            self.rot_available = False

        self.path_frame_multiplier = self.path.shape[0]/self.plan['path_extracted'].shape[0]
        self.shape_scale = np.min( 1 / np.array(self.plan['grid_shape']))

        self.frame_index = 0
        self.pos = self.get_pos()
        if self.rot_available:
            self.rot = self.get_rot()
        self.scale = glm.vec3([x*y for x, y in zip((np.broadcast_to(self.shape_scale*50, 3)), self.scale)])
        self.m_model = self.get_model_matrix()

        # texture
        self.texture = self.app.mesh.texture.textures[self.tex_id]
        self.program['u_texture_0'] = 0
        self.texture.use(location = 0)

        self.update_uniform_transformation_matrices(update_proj=True)
        self.update_uniform_light()

    def update(self):
        self.frame_index = min(int(self.app.clock.time_animation * self.app.clock.FPS_animation * self.path_frame_multiplier), (self.path.shape[0]-1))
        self.update_model_transform()
        self.update_uniform_transformation_matrices(update_camPos=True)
        self.texture.use()

        self.app.info_display.update_info_section(self.vao_name,f'{self.vao_name} pos: {self.pos}, rot: {self.rot}, scale: {self.scale}')

    def get_pos(self):
        self.translation = 2 * self.shape_scale * (self.path[self.frame_index,1:4] - (np.array(self.plan['grid_shape'])-1)/2)
        self.translation = [-self.translation[0], self.translation[2] - 0.5*self.shape_scale, self.translation[1]] # correct for the differences in the coord systems
        return glm.vec3(self.translation + self.initial_pos)
        
    def get_rot(self):
        self.rotation = self.path[:,4:7][self.frame_index]
        self.rotation = [self.rotation[0]-np.pi/2,self.rotation[2]-np.pi/2,self.rotation[1]]  # OPENGL: X,Z,Y
        return glm.vec3(self.rotation + self.initial_rot)

class CubeStatic(BaseModel):
    def __init__(self, app, vao_name='cubeStatic', tex_id='cube', pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):

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

        super().__init__(app, vao_name, tex_id, pos, rot, scale)
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
    def __init__(self, app, vao_name='cubeDynamic', tex_id='cube', pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):

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

        super().__init__(app, vao_name, tex_id, pos, rot, scale)
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
    def __init__(self, app, vao_name='spline', tex_id='None',
                 pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), plan:np.ndarray=None, path_name:str=None, color:glm.vec4=glm.vec4([1,1,1,1])):
        
        # Init VBO and VAO before calling super().init !!! (vbo_name = vao_name)
        app.mesh.vao.vbo.vbos[vao_name] = SplineVBO(app.ctx, reserve = plan[path_name].shape[0]*12)
        app.mesh.vao.vaos[vao_name] = app.mesh.vao.get_vao(program = app.mesh.vao.program.programs['spline'], 
                                                           vbo = [app.mesh.vao.vbo.vbos[vao_name]])
        
        super().__init__(app, vao_name, tex_id, pos, rot, scale)

        self.app = app
        self.vao_name = vao_name
        self.plan = plan
        self.path_name = path_name
        self.path = self.plan[path_name]
        self.color = glm.vec4(color)
        self.on_init()

    def on_init(self):
        self.shape_scale = 1 / np.max(self.plan['grid_shape'])
        path_transformed = 2 * self.shape_scale * (self.path[:, 1:4] - ((np.array(self.plan['grid_shape']) - 1) / 2))
        self.app.mesh.vao.vbo.vbos[self.vao_name].vbo.write((path_transformed).flatten().astype('f4'))
        
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
    def __init__(self, app, vao_name='coordsys', tex_id='None', pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
        self.data=np.array([0,0,0,1,0,0,1,  1,0,0,1,0,0,1,
                            0,0,0,0,1,0,1,  0,1,0,0,1,0,1,
                            0,0,0,0,0,1,1,  0,0,1,0,0,1,1])
        
        # Init VBO and VAO before calling super().init !!! (vbo_name = vao_name)
        app.mesh.vao.vbo.vbos[vao_name] = CoordSysVBO(app.ctx, reserve = self.data.shape[0]*28)
        app.mesh.vao.vaos[vao_name] = app.mesh.vao.get_vao(program = app.mesh.vao.program.programs['coordsys'],
                                                           vbo = [app.mesh.vao.vbo.vbos[vao_name]])
        super().__init__(app, vao_name, tex_id, pos, rot, scale)

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
                 pos=(0, 0, 0), rot=(0, 180, 0), scale=(1, 1, 1)):
        
        # Add vbo if it does not exist
        if not vbo_name in app.mesh.vao.vbo.vbos: # The same VBO is used for all drones, so it is created only once
            app.mesh.vao.vbo.vbos[vbo_name] = DefaultSTL_VBO(app.ctx, file=path_stl)

        # Init PROGRAM and VAO before calling super().init !!! (vao_name = program_name)
        app.mesh.vao.program.programs[vao_name] = app.mesh.vao.program.get_program('STL_default')
        app.mesh.vao.vaos[vao_name] = app.mesh.vao.get_vao(program = app.mesh.vao.program.programs['STL_default'],
                                                           vbo = [app.mesh.vao.vbo.vbos[vbo_name]])

        super().__init__(app, vao_name, tex_id, pos, rot, scale)
        self.update_initial_transform_parameters_for_dimension_normalization(vbo_name, normalize_dimensions=True)
        self.on_init()

    def update(self):
        self.update_uniform_transformation_matrices(update_camPos=True)

    def on_init(self):
        self.update_uniform_transformation_matrices(update_proj=True)
        self.update_uniform_light()

class DroneSTL(DefaultSTL):
    def __init__(self, app, vao_name='droneSTL',
                 pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), plan:dict=None):
        
        self.plan = plan
        self.vao_name = vao_name

        super().__init__(app, vao_name, vbo_name='droneSTL', tex_id='test', 
                         path_stl='objects/drone/quad.stl', 
                         pos=pos, rot=rot, scale=scale)

        self.on_init()

    def on_init(self):
        self.path_interpolated = self.plan['path_interp_MinimumSnapTrajectory'] # if 'path_interp_MinimumSnapTrajectory' in self.plan else self.plan['path_interp_BSpline']
        self.path_frame_multiplier = self.path_interpolated.shape[0]/self.plan['path_extracted'].shape[0]
        self.shape_scale = 1 / np.max(self.plan['grid_shape'])

        self.frame_index = 0
        self.pos = self.get_pos()
        self.scale = glm.vec3(self.initial_scale * self.shape_scale * 8)  # Scale the drone to fit the grid
        self.m_model = self.get_model_matrix()

        self.update_uniform_transformation_matrices(update_proj=True)
        self.update_uniform_light()

    def update(self):
        self.frame_index = min(int(self.app.clock.time_animation * self.app.clock.FPS_animation * self.path_frame_multiplier), (self.path_interpolated.shape[0]-1))
        self.pos = self.get_pos()
        self.rot = self.get_rot()
        self.m_model = self.get_model_matrix()

        self.update_uniform_transformation_matrices(update_camPos=True)

        self.app.info_display.update_info_section(self.vao_name,f'{self.vao_name} pos: {self.pos}, rot: {self.rot}, scale: {self.scale}')

    def get_pos(self):
        self.translation = 2 * self.shape_scale * (self.path_interpolated[self.frame_index,1:4] - (np.array(self.plan['grid_shape'])-1)/2)
        self.translation = [-self.translation[0], self.translation[2] - 0.5*self.shape_scale, self.translation[1]] # correct for the differences in the coord systems
        return glm.vec3(self.translation + np.array(self.initial_pos))

    def get_rot(self):
        if 'path_interp_MinimumSnapTrajectory' in self.plan:
            self.rotation = self.plan['path_interp_MinimumSnapTrajectory'][:,4:7][self.frame_index]
            self.rotation = [-self.rotation[1] + np.pi/2, self.rotation[2] , self.rotation[0] + np.pi]
            return glm.vec3(self.rotation + np.array(self.initial_rot))

# DEPRECATED: Renders the whole grid_seq without making a distinction between static and dynamic objects
class Cube(BaseModel):
    def __init__(self, app, vao_name='cube', tex_id='cube', pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
        super().__init__(app, vao_name, tex_id, pos, rot, scale)
        self.on_init()

    """ To render the cubes with instancing we need to pass the 'frame' 
        which is a numpy array to the shader somehow
        Solution: (the ModernGL way: described in moderngl/examples/instanced_rendering_crates)
        Include all the data needed for the shader to render in the VAO object
        with a format: 3f/i, where the i qualifier refers to instancing
        This means that when instancing some variables of the vertex array buffer 
        is updated in a loop. In our case we would have to include and update """

    def update(self):
        # grid_seq
        frame_index = min(int(self.app.clock.time * self.app.clock.FPS_animation), self.grid_seq.shape[0] - 1)
        self.frame = self.app.data.grid_seq[frame_index].astype('f4')
        self.app.mesh.vao.vbo.vbos['cubeInstanceValue'].vbo.write(np.array(self.frame.flatten()).astype('f4'))

        self.update_uniform_transformation_matrices()

    def on_init(self):
        self.frame = self.app.data.grid_seq[0].astype('f4')

        frame_indices = np.indices(self.frame.shape).reshape((3, -1)).T
        self.program['shape'].write(glm.ivec3(*self.frame.shape))   # uniform variable

        self.app.mesh.vao.vbo.vbos['cubeInstanceIndex'].vbo.write(np.array(frame_indices.flatten()).astype(int))
        self.app.mesh.vao.vbo.vbos['cubeInstanceValue'].vbo.write(np.array(self.frame.flatten()).astype('f4'))      

        self.update_uniform_transformation_matrices()
        self.update_uniform_light(specular=False)

    def render(self):
        self.update()
        self.vao.render(instances = np.prod(self.frame.shape))