from pathlib import Path
import numpy as np
import pywavefront
import stl # pip install numpy-stl

class VBO:
    def __init__(self, ctx):
        self.vbos = {}

    def destroy(self):
        [vbo.destroy() for vbo in self.vbos.values()]

class BaseVBO:
    def __init__(self, ctx, reserve = None):
        self.ctx = ctx
        self.vbo = self.get_vbo(reserve)
        self.format: str = None
        self.attribs: list = None

    def get_vbo(self, reserve=None):
        if reserve is None: 
            self.vertex_data = self.get_vertex_data()
        vbo = self.ctx.buffer(self.vertex_data) if reserve is None else self.ctx.buffer(reserve=reserve) # optionally RESERVE
        return vbo

    def get_vertex_data(self): ...

    def destroy(self):
        self.vbo.release()

class CubeVBO(BaseVBO):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.format = '3f 3f'
        self.attribs = ['a_position', 'a_normal']

    @staticmethod
    def get_data(vertices, indices):
        data = [vertices[ind] for triangle in indices for ind in triangle]
        return np.array(data, dtype='f4')

    def get_vertex_data(self):
        vertices =  [[-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1], [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1]]                    # vertices (points) defining the object
        normals =   [[0,0,1], [1,0,0], [0,0,-1], [-1,0,0], [0,1,0], [0,-1,0]]                                               # normal vectors for each face of the cube
        faces =     [[0,1,2,3], [1,5,6,2], [5,4,7,6], [4,0,3,7], [3,2,6,7], [1,0,4,5]]                                      # indices for each face of the cube

        attr_array = []                             # len 24: 6 face * 4 vertex
        for face_index in range(len(faces)):        # loops the 6 faces of the cube
            for vertex_index in faces[face_index]:  # loops the 4 vertices of each face
                attr_array += [*vertices[vertex_index], *normals[face_index]]

        return np.array(attr_array, dtype=np.float32) # vertex attributes

        ''' for face_index in range(6): This outer loop iterates over the six faces of the cube. 
        The cube is made up of six faces: front, back, left, right, top, and bottom.
        for vertex_index in range(6): This inner loop iterates over the six vertices of each face. Each face consists of four vertices.
        [0, 1, 2, 0, 2, 3][vertex_index]: This part is indexing into a list [0, 1, 2, 0, 2, 3] based on vertex_index. 
        This list represents the indices of the vertices that form a quad (two triangles) for each face. For example, 
        [0, 1, 2, 0, 2, 3] represents the vertices of a quad as two triangles: (0, 1, 2) and (0, 2, 3). 
        By indexing into this list with vertex_index, you get the indices of the vertices for each iteration of the inner loop.
        face_index*4: This part is used to adjust the indices based on the current face being processed. 
        Since each face has four vertices and the indices are continuous for each face, 
        we need to multiply face_index by 4 to get the correct starting index for each face.

        Putting it all together, the indices will contain indices for rendering each face of the cube.
        Each face will have its vertices specified in counter-clockwise order to define two triangles forming a quad.
        This list of indices will be used along with the vertex data to render the cube properly. '''

class CubeStaticInstanceVBO(BaseVBO):
    def __init__(self, ctx, reserve):
        super().__init__(ctx, reserve)
        self.format = '3f f/i'
        self.attribs = ['a_instance_index', 'a_instance_value']
    
class CubeDynamicInstanceVBO(BaseVBO):
    def __init__(self, ctx, reserve):
        super().__init__(ctx, reserve)
        self.format = '3f f/i'
        self.attribs = ['a_instance_index', 'a_instance_value']

class DefaultOBJ_VBO(BaseVBO):
    def __init__(self, app, file:str='objects/.../xxx.obj'):
        self.file = Path(__file__).parent / file
        super().__init__(app)
        self.format = '2f 3f 3f'
        self.attribs = ['in_texcoord_0', 'in_normal', 'in_position']
        
    def get_vertex_data(self):
        objs = pywavefront.Wavefront(file_name=self.file, cache=True, parse=True)
        obj = objs.materials.popitem()[1]
        vertex_data = obj.vertices
        vertex_data = np.array(vertex_data, dtype='f4')
        return vertex_data
    
class DefaultSTL_VBO(BaseVBO):
    def __init__(self, app, file:str='objects/.../xxx.stl'):
        self.file = Path(__file__).parent / file

        super().__init__(app)
        self.format = '3f 3f'  # Assuming normals, vertices, and texture coordinates
        self.attribs = ['in_normal', 'in_position']
        

    def get_vertex_data(self):
        try:
            mesh_data = stl.mesh.Mesh.from_file(self.file)  # Load the STL file
            vertices = mesh_data.vectors.reshape(-1, 3)  # Extract vertex data
            normals = mesh_data.normals.reshape(-1, 3)   # Extract normals
            
            normals = np.repeat(normals, 3, axis=0)
            vertex_data = np.hstack((normals, vertices)).astype('f4')
            return vertex_data
        
        except Exception as e:
            print(f"Error loading STL file: {e}")
            return None

class SplineVBO(BaseVBO):
    def __init__(self, ctx, reserve):
        super().__init__(ctx, reserve)
        self.format = '3f'
        self.attribs = ['in_vert']

class CoordSysVBO(BaseVBO):
    def __init__(self, ctx, reserve):
        super().__init__(ctx, reserve)
        self.format = '3f 4f'
        self.attribs = ['in_vert','in_color']
