import os
from pathlib import Path

class ShaderProgram:
    def __init__(self, ctx):
        self.ctx = ctx
        self.programs = {}

        dir = Path(__file__).parent / 'shaders'
        for file in dir.glob('*.vert'):
            name = file.stem
            frag_file = dir / f'{name}.frag'
            if frag_file.exists():
                self.programs[name] = self.get_program(name)


    def get_program(self, shader_program_name):

        dir = Path(__file__).parent
       
        with open(dir/f'shaders/{shader_program_name}.vert') as file:
            vertex_shader = file.read()

        with open( dir/f'shaders/{shader_program_name}.frag') as file:
            fragment_shader = file.read()

        program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        return program

    def destroy(self):
        [program.release() for program in self.programs.values()]
