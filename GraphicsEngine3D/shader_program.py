from pathlib import Path

class ShaderProgram:
    def __init__(self, ctx):
        self.ctx = ctx
        self.programs = {}

        self.shader_dir = Path(__file__).parents[0] / 'shaders'
        for file in self.shader_dir.glob('*.vert'):
            name = file.stem
            frag_file = self.shader_dir / f'{name}.frag'
            if frag_file.exists():
                self.programs[name] = self.get_program(name)


    def get_program(self, shader_program_name):
       
        with open(self.shader_dir/f'{shader_program_name}.vert') as file:
            vertex_shader = file.read()

        with open(self.shader_dir/f'{shader_program_name}.frag') as file:
            fragment_shader = file.read()

        program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        return program

    def destroy(self):
        [program.release() for program in self.programs.values()]
