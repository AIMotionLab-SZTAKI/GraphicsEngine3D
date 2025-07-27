import pygame as pg
import moderngl as mgl
from pathlib import Path

class Texture:
    def __init__(self, ctx):
        self.ctx = ctx
        self.textures = {}

        self.textures['test'] = self.get_texture(path=Path(__file__).parent/'objects/cube/test.png')

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

    def destroy(self):
        [tex.release() for tex in self.textures.values()]