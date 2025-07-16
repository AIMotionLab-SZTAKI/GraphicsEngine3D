import moderngl as mgl
import numpy as np
import glm
from pathlib import Path

from OpenGL.GL import *

from vbo import CubeVBO, CubeStaticInstanceVBO, CubeDynamicInstanceVBO, SplineVBO, CoordSysVBO, DefaultSTL_VBO, DefaultOBJ_VBO

class BaseModel:
    def __init__(self, app, vao_name, tex_id, pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
        self.app = app
        self.pos = pos
        self.rot = glm.vec3([glm.radians(a) for a in rot])
        self.scale = scale
        self.m_model = self.get_model_matrix()
        self.tex_id = tex_id
        self.vao = app.mesh.vao.vaos[vao_name]
        self.program = self.vao.program
        self.camera = self.app.camera

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

    def render(self):
        self.update()
        self.vao.render()

class CubeStatic(BaseModel):
    def __init__(self, app, vao_name='cubeStatic', tex_id='cube', pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):

        if not 'cube' in app.mesh.vao.vbo.vbos:  # The same VBO is used for all cubes, so it is created only once
            app.mesh.vao.vbo.vbos['cube'] = CubeVBO(app.ctx)
        if not 'cubeStaticInstance' in app.mesh.vao.vbo.vbos:
            app.mesh.vao.vbo.vbos['cubeStaticInstance'] = \
                CubeStaticInstanceVBO(app.ctx, reserve = app.data.grid_static_instancelist.shape[0] * 16) # 4f ~16 byte # not updated at all

        # cube static vao
        app.mesh.vao.vaos['cubeStatic'] = app.mesh.vao.get_vao(
            program = app.mesh.vao.program.programs['cubeStatic'],
            vbo = [app.mesh.vao.vbo.vbos['cube'],
                   app.mesh.vao.vbo.vbos['cubeStaticInstance']],
            ibo = app.mesh.vao.ibo.ibos['cube'])

        super().__init__(app, vao_name, tex_id, pos, rot, scale)
        self.on_init()

    def update(self):
        # mvp
        self.program['m_proj'].write(self.camera.m_proj)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)
        #self.program['camPos'].write(self.camera.position)

    def on_init(self):

        # grid_static
        self.grid_static_instancelist = self.app.data.grid_static_instancelist
        self.program['shape'].write(glm.ivec3(*self.app.data.grid_shape))   # uniform variable
        self.app.mesh.vao.vbo.vbos['cubeStaticInstance'].vbo.write(np.array(self.grid_static_instancelist.flatten()).astype('f4'))

        # mvp
        #self.program['camPos'].write(self.camera.position)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)
        # light
        self.program['light.position'].write(self.app.light.position)
        self.program['light.Ia'].write(self.app.light.Ia)
        self.program['light.Id'].write(self.app.light.Id)
        #self.program['light.Is'].write(self.app.light.Is)

        
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

        # cube dynamic vao
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
        
        # mvp
        self.program['camPos'].write(self.camera.position)
        self.program['m_proj'].write(self.camera.m_proj)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)

    def on_init(self):

        # grid_seq dynamic
        self.grid_seq_dynamic_instancelist = self.app.data.grid_seq_dynamic_instancelist
        self.program['shape'].write(glm.ivec3(*self.app.data.grid_shape))   # uniform variable
        self.frame_index = 0
        self.frame = self.grid_seq_dynamic_instancelist[self.frame_index]
        self.app.mesh.vao.vbo.vbos['cubeDynamicInstance'].vbo.write(np.array(self.frame.flatten()).astype('f4'))   

        # mvp
        self.program['camPos'].write(self.camera.position)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)
        # light
        self.program['light.position'].write(self.app.light.position)
        self.program['light.Ia'].write(self.app.light.Ia)
        self.program['light.Id'].write(self.app.light.Id)
        #self.program['light.Is'].write(self.app.light.Is)

        
    def render(self):
        self.update()

        '''
        # Disable color writes, keep both depth testing and depth writes on
        self.app.ctx.fbo.color_mask = False, False, False, True
        # depth buffer to contain the depth of the closest visible surface of your transparent cubes
        self.vao.render(instances = self.grid_seq_dynamic_instancelist.shape[1]) # Depth only draw
        # Enable color writes and blending
        self.app.ctx.fbo.color_mask = True, True, True, True
        # renders all of the cubes, only the pixels that are the closest visible surface will pass the depth test
        '''
        self.vao.render(instances = self.grid_seq_dynamic_instancelist.shape[1])

class DroneOBJ(BaseModel):
    def __init__(self, app, vao_name='drone_OBJ', tex_id='drone',
                 pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), plan:dict=None):
        
        if not 'drone_OBJ' in app.mesh.vao.vbo.vbos: # The same VBO is used for all drones, so it is created only once
            app.mesh.vao.vbo.vbos['drone_OBJ'] = DefaultOBJ_VBO(app.ctx, file='objects/drone/MQ-9.obj', modify_vertices=True)

        dir = Path(__file__).parent
        if not tex_id in app.mesh.texture.textures:
            app.mesh.texture.textures[tex_id] = app.mesh.texture.get_texture(path=dir/'objects/drone/MQ-9_Black.png')  # Load texture

        # Init PROGRAM and VAO before calling super().init !!! (vao_name = program_name)
        app.mesh.vao.program.programs[vao_name] = app.mesh.vao.program.get_program('OBJ_default') # OBJ
        app.mesh.vao.vaos[vao_name] = app.mesh.vao.get_vao(program = app.mesh.vao.program.programs[vao_name],
                                                           vbo = [app.mesh.vao.vbo.vbos['drone_OBJ']])
        
        super().__init__(app, vao_name, tex_id, pos, rot, scale)
        self.plan = plan
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
        self.scale = glm.vec3(self.shape_scale * 8)
        self.m_model = self.get_model_matrix()


        # texture
        self.texture = self.app.mesh.texture.textures[self.tex_id]
        self.program['u_texture_0'] = 0
        self.texture.use(location = 0)
        # mvp
        self.program['m_proj'].write(self.camera.m_proj)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)
        # light
        self.program['light.position'].write(self.app.light.position)
        self.program['light.Ia'].write(self.app.light.Ia)
        self.program['light.Id'].write(self.app.light.Id)
        self.program['light.Is'].write(self.app.light.Is)

    def update(self):
        self.frame_index = min(int(self.app.clock.time_animation * self.app.clock.FPS_animation * self.path_frame_multiplier), (self.path.shape[0]-1))
        self.pos = self.get_pos()
        if self.rot_available:
            self.rot = self.get_rot()
        self.m_model = self.get_model_matrix()

        # mvp
        self.texture.use()
        self.program['camPos'].write(self.camera.position)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)

    def get_pos(self):
        self.translation = 2 * self.shape_scale * (self.path[self.frame_index,1:4] - (np.array(self.plan['grid_shape'])-1)/2)
        self.translation = [-self.translation[0], self.translation[2] - 0.5*self.shape_scale, self.translation[1]] # correct for the differences in the coord systems
        return glm.vec3(self.translation)

    def get_rot(self):
        self.rotation = self.path[:,4:7][self.frame_index]
        self.rotation = [self.rotation[0]-np.pi/2,self.rotation[2]-np.pi/2,self.rotation[1]]  # OPENGL: X,Z,Y
        return glm.vec3(self.rotation)

class DroneSTL(BaseModel):
    def __init__(self, app, vao_name='droneSTL', tex_id='None',
                 pos=(0, 0, 0), rot=(90, 0, 180), scale=(1, 1, 1), plan:dict=None):
        
        if not 'drone_STL' in app.mesh.vao.vbo.vbos: # The same VBO is used for all drones, so it is created only once
            app.mesh.vao.vbo.vbos['drone_STL'] = DefaultSTL_VBO(app.ctx, file='objects/drone/quad.stl', modify_vertices=True)

        # Init PROGRAM and VAO before calling super().init !!! (vao_name = program_name)
        app.mesh.vao.program.programs[vao_name] = app.mesh.vao.program.get_program('STL_default')
        app.mesh.vao.vaos[vao_name] = app.mesh.vao.get_vao(program = app.mesh.vao.program.programs[vao_name],
                                                           vbo = [app.mesh.vao.vbo.vbos['drone_STL']])

        super().__init__(app, vao_name, tex_id, pos, rot, scale)

        self.plan = plan
        self.on_init()

    def on_init(self):
        self.path_interpolated = self.plan['path_interp_MinimumSnapTrajectory'] # if 'path_interp_MinimumSnapTrajectory' in self.plan else self.plan['path_interp_BSpline']
        self.path_frame_multiplier = self.path_interpolated.shape[0]/self.plan['path_extracted'].shape[0]
        self.shape_scale = np.min( 1 / np.array(self.plan['grid_shape']))

        self.frame_index = 0
        self.pos = self.get_pos()
        self.scale = glm.vec3(self.shape_scale * 4)
        self.m_model = self.get_model_matrix()

        # mvp
        self.program['m_proj'].write(self.camera.m_proj)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)
        # light
        self.program['light.position'].write(self.app.light.position)
        self.program['light.Ia'].write(self.app.light.Ia)
        self.program['light.Id'].write(self.app.light.Id)
        self.program['light.Is'].write(self.app.light.Is)

    def update(self):
        self.frame_index = min(int(self.app.clock.time_animation * self.app.clock.FPS_animation * self.path_frame_multiplier), (self.path_interpolated.shape[0]-1))
        self.pos = self.get_pos()
        self.rot = self.get_rot()
        self.m_model = self.get_model_matrix()

        # mvp
        self.program['camPos'].write(self.camera.position)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)

    def get_pos(self):
        self.translation = 2 * self.shape_scale * (self.path_interpolated[self.frame_index,1:4] - (np.array(self.plan['grid_shape'])-1)/2)
        self.translation = [-self.translation[0], self.translation[2] - 0.5*self.shape_scale, self.translation[1]] # correct for the differences in the coord systems
        return glm.vec3(self.translation)

    def get_rot(self):
        if 'path_interp_MinimumSnapTrajectory' in self.plan:
            self.rotation = self.plan['path_interp_MinimumSnapTrajectory'][:,4:7][self.frame_index]
            self.rotation = [-self.rotation[1] + np.pi/2, self.rotation[2] , self.rotation[0] + np.pi]
            return glm.vec3(self.rotation)

class Spline(BaseModel):
    def __init__(self, app, vao_name='spline', tex_id='None',
                 pos=(0, 0, 0), rot=(90, 0, 180), scale=(1, 1, 1), plan:np.ndarray=None, path_name:str=None, color:glm.vec4=glm.vec4([1,1,1,1])):
        
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
        self.color = color
        self.on_init()

    def update(self):
        self.program['color'].write(self.color) # Color is updated sequentually, since a custom shader object would have been needed to make it static
        # mvp
        #self.program['camPos'].write(self.camera.position)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)

    def on_init(self):
        self.shape_scale = np.min( 1 / np.array(self.plan['grid_shape']))

        path_transformed = np.zeros((self.path.shape[0],3))
        for i in range(len(self.path)):
            path_transformed[i] = 2 * self.shape_scale * (self.path[i,1:4] - (np.array(self.plan['grid_shape'])-1)/2)

        self.app.mesh.vao.vbo.vbos[self.vao_name].vbo.write((path_transformed).flatten().astype('f4'))
        

        # mvp
        self.program['m_proj'].write(self.camera.m_proj)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)

    def render(self):
        self.update()
        self.app.ctx.line_width = 4 # 3
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
        # mvp
        #self.program['camPos'].write(self.camera.position)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)

    def on_init(self):           
        self.app.mesh.vao.vbo.vbos[self.vao_name].vbo.write(self.data.astype('f4'))
        
        # mvp
        self.program['m_proj'].write(self.camera.m_proj)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)

    def render(self):
        self.update()
        self.app.ctx.line_width = 3
        self.vao.render(mgl.LINES)
        self.app.ctx.line_width = 1 # RESET

class DefaultOBJ(BaseModel):
    def __init__(self, app, vao_name='default', vbo_name='default', tex_id='test', path_obj='objects/terrain/terrain.obj', path_texture='objects/terrain/terrain.png',
                 pos=(0, 0, 0), rot=(-90, 0, 0), scale=(1, 1, 1)):
        
        # Add vbo if it does not exist
        if not vbo_name in app.mesh.vao.vbo.vbos: # Only create the VBO once
            app.mesh.vao.vbo.vbos[vbo_name] = DefaultOBJ_VBO(app.ctx, file=path_obj, modify_vertices=False)

        # Add texture if tex_id != 'test' and it does not exist already
        if tex_id != 'test' and tex_id not in app.mesh.texture.textures:
            app.mesh.texture.textures[tex_id] = app.mesh.texture.get_texture(path=Path(__file__).parent/path_texture)  # Load texture

        # Init PROGRAM and VAO before calling super().init !!! (vao_name = program_name)
        app.mesh.vao.program.programs[vao_name] = app.mesh.vao.program.get_program('OBJ_default')
        app.mesh.vao.vaos[vao_name] = app.mesh.vao.get_vao(program = app.mesh.vao.program.programs['OBJ_default'],
                                                           vbo = [app.mesh.vao.vbo.vbos[vbo_name]])
        
        super().__init__(app, vao_name, tex_id, pos, rot, scale)
        self.on_init()

    

    def on_init(self):
        # texture
        self.texture = self.app.mesh.texture.textures[self.tex_id]
        self.program['u_texture_0'] = 0
        self.texture.use(location = 0)
        # mvp
        self.program['m_proj'].write(self.camera.m_proj)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)
        # light
        self.program['light.position'].write(self.app.light.position)
        self.program['light.Ia'].write(self.app.light.Ia+0.2)
        self.program['light.Id'].write(self.app.light.Id)
        self.program['light.Is'].write(self.app.light.Is)

    def update(self):
        self.texture.use()
        self.program['camPos'].write(self.camera.position)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)

class DefaultSTL(BaseModel):
    def __init__(self, app, vao_name='defaultSTL', tex_id='None',
                 pos=(0, 0, 0), rot=(0, 180, 0), scale=(1, 1, 1)):
        super().__init__(app, vao_name, tex_id, pos, rot, scale)
        self.on_init()

    def update(self):
        self.program['camPos'].write(self.camera.position)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)

    def on_init(self):
        # mvp
        self.program['m_proj'].write(self.camera.m_proj)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)
        # light
        self.program['light.position'].write(self.app.light.position)
        self.program['light.Ia'].write(self.app.light.Ia)
        self.program['light.Id'].write(self.app.light.Id)
        self.program['light.Is'].write(self.app.light.Is)

# DEPRECATED: Renders the whole grid_seq without making a distinction between static and dynamic objects
class Cube(BaseModel):
    def __init__(self, app, vao_name='cube', tex_id='cube', pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
        super().__init__(app, vao_name, tex_id, pos, rot, scale)
        self.on_init()

    """ To render the cubes with instancing we need to pass the 'frame' 
        which is a numpy array to the shader somehow
        Solution 1: (the legacy OpenGL way: described in moderngl/examples/external_texture, 3D_OpenGL_v2) 
            Create a texture3D object, and bind it to the uniform buffer,
            in that way all instances can acces the matrix, and access their relavant
            data fields with indexing and texelFetch()
        Solution 2: (the ModernGL way: described in moderngl/examples/instanced_rendering_crates)
            Include all the data needed for the shader to render in the VAO object
            with a format: 3f/i, where the i qualifier refers to instancing
            This means that when instancing some variables of the vertex array buffer 
            is updated in a loop. In our case we would have to include and update 
            4f amount of data per vertex to make this solution work, so instead i used Sol1
    """

    def update(self):
        # grid_seq
        frame_index = min(int(self.app.clock.time * self.app.clock.FPS_animation), self.grid_seq.shape[0] - 1)
        self.frame = self.grid_seq[frame_index].astype('f4')
        self.app.mesh.vao.vbo.vbos['cubeInstanceValue'].vbo.write(np.array(self.frame.flatten()).astype('f4'))

        # mvp
        self.program['m_proj'].write(self.camera.m_proj)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)

    def on_init(self):

        # grid_seq
        self.grid_seq = self.app.data.grid_seq
        self.frame = self.grid_seq[0].astype('f4')

        frame_indices = np.indices(self.frame.shape).reshape((3, -1)).T
        self.program['shape'].write(glm.ivec3(*self.frame.shape))   # uniform variable

        self.app.mesh.vao.vbo.vbos['cubeInstanceIndex'].vbo.write(np.array(frame_indices.flatten()).astype(int))
        self.app.mesh.vao.vbo.vbos['cubeInstanceValue'].vbo.write(np.array(self.frame.flatten()).astype('f4'))      

        # mvp
        #self.program['camPos'].write(self.camera.position)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)
        # light
        self.program['light.position'].write(self.app.light.position)
        self.program['light.Ia'].write(self.app.light.Ia)
        self.program['light.Id'].write(self.app.light.Id)
        #self.program['light.Is'].write(self.app.light.Is)

        
    def render(self):
        self.update()
        self.vao.render(instances = np.prod(self.frame.shape))



    def timeToFrame():
        pass