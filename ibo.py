import numpy as np
import moderngl as mgl


class IBO:
    def __init__(self, ctx):
        self.ibos = {}
        self.ibos['cube'] = CubeIBO(ctx)

    def destroy(self):
        [ibo.destroy() for ibo in self.ibos.values()]


class BaseIBO:
    def __init__(self, ctx):
        self.ctx = ctx
        self.ibo = self.get_ibo()

    def get_index_data(self): ...

    def get_ibo(self):
        indices = self.get_index_data()
        ibo = self.ctx.buffer(indices)
        return ibo

    def destroy(self):
        self.ibo.release()


class CubeIBO(BaseIBO):
    def __init__(self, ctx):
        super().__init__(ctx)

    def get_index_data(self):
        indices = [face_index*4+[0, 1, 2, 0, 2, 3][vertex_index] for face_index in range(6) for vertex_index in range(6)]   # len 36: 6 face * 6 vertex
        return np.array(indices, dtype=np.uint32)
        