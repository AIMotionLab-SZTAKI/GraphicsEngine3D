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
        self.add_object(CoordSys(app, vao_name='coordsys_WORLD', pos=(0,0.4,0), rot=(90,0,180), scale=0.1))
        self.add_object(CoordSys(app, vao_name='coordsys_WORLD_OPENGL', pos=(0,0.1,0), rot=(0,0,0), scale=0.1))
        #self.add_object(CoordSys(app, vao_name='coordsys_MAP_ORIGIN', pos=(1,0,-1), rot=(90,0,180), scale=0.1))
        #self.add_object(CoordSys(app, vao_name='coordsys_MAP_ORIGIN_OPENGL', pos=(1,0,-1), rot=(0,0,0), scale=0.1))

        if 'grid' in self.scene:
            self.add_object(CubeStatic(app, rot=(90,0,180)))    # total instanced cube size: 2x2x2
        
        if 'plans' in self.scene:
            plans = self.app.data.plans
            for plan_idx in range(len(plans)):

                if 'path_extracted' in plans[plan_idx] and not 'grid' in self.scene: # Continous world (plan only contains 'path_extracted' and 'grid_shape')
                    self.add_object(Spline(app, vao_name='spline1'+str(plan_idx), plan=plans[plan_idx], path_name='path_extracted', color=[31/255,119/255,180/255,1], rot=(90,0,180)))
                    self.add_object(DroneOBJ(app, vao_name='drone'+str(plan_idx), plan = plans[plan_idx], path_name='path_extracted'))
                
                if 'path_corrected' in plans[plan_idx] and 'grid' in self.scene: # Grid based world (pland contains 'path_corrected' and 'grid_shape')
                    #self.add_object(Spline(app, vao_name='spline2'+str(plan_idx), plan=plans[plan_idx], path_name='path_corrected', color=[44/255,160/255,44/255,1], rot=(90,0,180)))
                    pass
                if 'path_interp_BSpline' in plans[plan_idx] and 'path_interp_MinimumSnapTrajectory' not in plans[plan_idx]:
                    self.add_object(DroneOBJ(app, vao_name='drone'+str(plan_idx), plan = plans[plan_idx], path_name='path_interp_BSpline'))
                    self.add_object(Spline(app, vao_name='spline3'+str(plan_idx), plan=plans[plan_idx], path_name='path_interp_BSpline', color=[0.8,0.2,0.2,1], rot=(90,0,180)))
                if 'path_interp_MinimumSnapTrajectory' in plans[plan_idx]:
                    self.add_object(DroneSTL(app, vao_name='drone'+str(plan_idx), plan = plans[plan_idx], path_name='path_interp_MinimumSnapTrajectory'))
                    self.add_object(Spline(app, vao_name='spline4'+str(plan_idx), plan=plans[plan_idx], path_name='path_interp_MinimumSnapTrajectory', color=[0.8,0.2,0.2,1], rot=(90,0,180)))

        if 'grid' in self.scene:
            self.add_object(CubeDynamic(app, rot=(90,0,180)))   # total instanced cube size: 2x2x2

        if 'terrain' in self.scene:
            rot = (90,0,180) if 'grid' not in self.scene else (90,90,180) # SOLUTION NEEDED TO DISCTINGUISH BETWEEN GRID AND CONTINOUS WORLD
            self.add_object(DefaultOBJ(app, vao_name='terrain', vbo_name='terrain', tex_id='terrain',
                                    path_obj='objects/terrain/terrain.obj',
                                    path_texture='objects/terrain/terrain.png',
                                    rot=rot, scale=1))

        if 'obj' in self.scene:  
            for obj_plan in self.app.data.obj_plans:
                if obj_plan['type'] == 'drone':
                    tex_id = 'drone' # same texture for all drones
                    path_texture = 'objects/drone/MQ-9_Diffuse.jpg'
                    rot=(0,180,0) # rotate drone to face the correct X+ direction

                    self.add_object(Spline(app, vao_name='spline_'+obj_plan['id'], 
                                           plan=obj_plan, path_name='path', 
                                           color=[1,0,0,1]))
                else:
                    self.app.mesh.texture.textures[obj_plan['id']] = create_texture_from_rgba(self.app.ctx, rgba=obj_plan['color'])
                    tex_id = obj_plan['id'] # separete texture for each object
                    path_texture = None
                    rot=(0,0,0) # Possibly no rotation needed for other objects, but need to check with an object that has a detectable orientation in Z

                self.add_object(DefaultOBJ(app, vao_name=obj_plan['id'],
                                            vbo_name=obj_plan['type'],
                                            tex_id= tex_id,
                                            path_obj=f'objects/obj/{obj_plan["type"]}.obj',
                                            path_texture=path_texture,
                                            path=obj_plan['path'], 
                                            rotation_available=True,
                                            normalize_dimensions=True,
                                            center_obj=False,
                                            rot=rot,
                                            scale=2*obj_plan['dimension']/np.max(obj_plan['world_dimensions']),
                                            alpha=obj_plan['color'][3]))
             
        '''self.add_object(DefaultOBJ(app, vao_name='cat', vbo_name='cat', tex_id='cat', path_obj='objects/cat/20430_Cat_v1_NEW.obj',
                                      path_texture='objects/cat/20430_cat_diff_v1.jpg', normalize_dimensions=True))'''
        
    def render(self):
        for obj in self.objects:
            obj.render()