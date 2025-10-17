from vbo import VBO
from ibo import IBO
from shader_program import ShaderProgram


class VAO:
    def __init__(self, ctx):
        self.ctx = ctx
        self.vbo = VBO(ctx)
        self.ibo = IBO(ctx)
        self.program = ShaderProgram(ctx)
        self.vaos = {}
        
    def get_vao(self, program, vbo, ibo=None):
        
        vao = self.ctx.vertex_array(program = program, 
                                    content = [(vbo[i].vbo, vbo[i].format, *vbo[i].attribs) for i in range(len(vbo))],
                                    index_buffer = ibo.ibo if ibo is not None else None)
        return vao

    def destroy(self):
        self.vbo.destroy()
        self.program.destroy()