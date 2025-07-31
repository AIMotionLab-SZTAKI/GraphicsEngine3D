from turtle import pos
from model import *
from utils import create_texture_from_rgba

class Scene:
    def __init__(self, app, scene=['all']):
        self.app = app
        self.scene = scene
        self.objects = []
        self.load()

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self):
        app = self.app

        coord_transform = glm.mat4(
                -1,  0,  0,  0,  # negate x
                 0,  0,  1,  0,  # z -> y
                 0,  1,  0,  0,  # y -> z
                 0,  0,  0,  1)

        self.add_object(CoordSys(app, vao_name='coordsys_WORLD', pos=(0,0,0.2), scale=0.1, coord_sys=coord_transform))
        self.add_object(CoordSys(app, vao_name='coordsys_WORLD_OPENGL', pos=(0,0.5,0), scale=0.1))
        self.add_object(CoordSys(app, vao_name='coordsys_MAP_ORIGIN', pos=(-1,-1,0), scale=0.1, coord_sys=coord_transform))
        #self.add_object(CoordSys(app, vao_name='coordsys_MAP_ORIGIN_OPENGL', pos=(1,0,-1), rot=(0,0,0), scale=0.1))

        if 'grid' in self.scene:
            self.add_object(CubeStatic(app, coord_sys=coord_transform))    # total instanced cube size: 2x2x2

        if 'plans' in self.scene:
            plans = self.app.data.plans
            for plan_idx in range(len(plans)):
                #self.add_object(Spline(app, vao_name='spline1'+str(plan_idx), plan=plans[plan_idx], path_name='path_extracted', color=[31/255,119/255,180/255,1], rot=(90,0,180)))
                #self.add_object(Spline(app, vao_name='spline2'+str(plan_idx), plan=plans[plan_idx], path_name='path_corrected', color=[44/255,160/255,44/255,1], rot=(90,0,180)))
                if 'path_interp_BSpline' in plans[plan_idx] and 'path_interp_MinimumSnapTrajectory' not in plans[plan_idx]:
                    self.add_object(DefaultOBJ(app, vao_name='drone'+str(plan_idx), vbo_name='drone', tex_id='drone',
                                               path_obj='objects/obj/drone.obj',
                                               path_texture='objects/drone/MQ-9_Diffuse.jpg',
                                               path=plans[plan_idx]['path_interp_BSpline'], rotation_available=True,
                                               rot=(0,0,90), scale=0.01, coord_sys=coord_transform))
                    self.add_object(Spline(app, vao_name='spline3'+str(plan_idx), path=plans[plan_idx]['path_interp_BSpline'], color=[0.8,0.2,0.2,1], coord_sys=coord_transform))
                if 'path_interp_MinimumSnapTrajectory' in plans[plan_idx]:
                    self.add_object(Spline(app, vao_name='spline4'+str(plan_idx), path=plans[plan_idx]['path_interp_MinimumSnapTrajectory'], color=[0.8,0.2,0.2,1], rot=(90,0,180)))
                    self.add_object(DefaultSTL(app, vao_name='droneSTL'+str(plan_idx), path_stl='objects/drone/quad.stl',
                                               path=plans[plan_idx]['path_interp_MinimumSnapTrajectory'], rotation_available=True,
                                               coord_sys=coord_transform, rot=(0,0,90)))

        if 'grid' in self.scene:
            self.add_object(CubeDynamic(app, coord_sys=coord_transform))   # total instanced cube size: 2x2x2

        if 'terrain' in self.scene:
            rot = (0,0,0) if 'grid' not in self.scene else (0,0,-90) # SOLUTION NEEDED TO DISCTINGUISH BETWEEN GRID AND CONTINOUS WORLD
            self.add_object(DefaultOBJ(app, vao_name='terrain', vbo_name='terrain', tex_id='terrain',
                                    path_obj='objects/terrain/terrain.obj',
                                    path_texture='objects/terrain/terrain.png',
                                    rot=rot, scale=1,
                                    coord_sys=coord_transform))

        if 'obj' in self.scene:
            for obj_plan in self.app.data.obj_plans:
                if obj_plan['type'] == 'drone':
                    tex_id = 'drone' # same texture for all drones
                    path_texture = 'objects/drone/MQ-9_Diffuse.jpg'

                    self.add_object(Spline(app, vao_name='spline_'+obj_plan['id'], path=obj_plan['path'], color=[1,0,0,1],
                                           coord_sys=coord_transform))
                else:
                    self.app.mesh.texture.textures[obj_plan['id']] = create_texture_from_rgba(self.app.ctx, rgba=obj_plan['color'])
                    tex_id = obj_plan['id'] # separete texture for each object
                    path_texture = None

                if isinstance(obj_plan['dimension'], (float, int)): # If dimension is a single value, normalize dimensions with keeping aspect ratio
                    normalize_dimensions = 'max' # normalize by the largest dimension of the instance
                else:
                    normalize_dimensions = 'unit' # rescale to unit cube

                self.add_object(DefaultOBJ(app, vao_name=obj_plan['id'],
                                            vbo_name=obj_plan['type'],
                                            tex_id= tex_id,
                                            path_obj=f'objects/obj/{obj_plan["type"]}.obj',
                                            path_texture=path_texture,
                                            path=obj_plan['path'],
                                            rotation_available=True,
                                            scale=2*np.array(obj_plan['dimension'])/np.max(obj_plan['world_dimensions']),
                                            instance_rot=(0,0,0),
                                            coord_sys=coord_transform,
                                            normalize_instance_dimensions=normalize_dimensions,
                                            center_instance=True,
                                            alpha=obj_plan['color'][3]))
        '''
        self.add_object(DefaultOBJ(app, vao_name='drone_WORLD_OPENGL', vbo_name='drone', tex_id='cat',
                                   path_obj='objects/obj/drone.obj',
                                   path_texture='objects/drone/MQ-9_Diffuse.jpg',
                                   pos=(0, 0.1, 0), scale=0.01))
        self.add_object(DefaultOBJ(app, vao_name='drone_WORLD', vbo_name='drone', tex_id='cat',
                                   path_obj='objects/obj/drone.obj',
                                   path_texture='objects/drone/MQ-9_Diffuse.jpg',
                                   coord_sys=coord_transform, instance_rot=(90, 0, 180),
                                   pos=(0, 0, 0.4), scale=0.01))
        '''
        
                                   
    def render(self):
        for obj in self.objects:
            obj.render()