import pygame as pg
import moderngl as mgl
from pathlib import Path

class Texture:
    def __init__(self, ctx):
        self.ctx = ctx
        self.textures = {}
        
        self.textures['test'] = self.get_texture(path=Path(__file__).parents[0]/'objects/cube/test.png')

    def get_texture(self, path):
        # Load image and preserve alpha channel if present
        texture = pg.image.load(path)
        
        # Check if the original image has alpha channel
        has_alpha = texture.get_flags() & pg.SRCALPHA or texture.get_bitsize() == 32
        
        if has_alpha:
            texture = texture.convert_alpha() # Also adds alpha channel if it is not present
            components = 4
            format_str = 'RGBA'
        else:
            texture = texture.convert()
            components = 3
            format_str = 'RGB'
        
        texture = pg.transform.flip(texture, flip_x=False, flip_y=True)
        texture = self.ctx.texture(size=texture.get_size(),
                                   components=components,
                                   data=pg.image.tostring(texture, format_str))
        # mipmaps
        texture.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
        texture.build_mipmaps()
        # AF
        texture.anisotropy = 32.0
        return texture

    def get_texture_cube(self, dir_path, ext='png'):
        faces = ['right', 'left', 'top', 'bottom'] + ['front', 'back'][::-1]
        # textures = [pg.image.load(dir_path + f'{face}.{ext}').convert() for face in faces]
        textures = []
        for face in faces:
            texture = pg.image.load(dir_path / f'{face}.{ext}').convert()
            if face in ['right', 'left', 'front', 'back']:
                texture = pg.transform.flip(texture, flip_x=True, flip_y=False)
            else:
                texture = pg.transform.flip(texture, flip_x=False, flip_y=True)
            textures.append(texture)

        size = textures[0].get_size()
        texture_cube = self.ctx.texture_cube(size=size, components=3, data=None)

        for i in range(6):
            texture_data = pg.image.tostring(textures[i], 'RGB')
            texture_cube.write(face=i, data=texture_data)

        return texture_cube

    def add(self, tex_id: str, texture):
        if tex_id not in self.textures and tex_id != 'test':
            self.textures[tex_id] = texture

    def destroy(self):
        [tex.release() for tex in self.textures.values()]