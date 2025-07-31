import pygame as pg
import moderngl as mgl
from typing import Tuple
import numpy as np

class InfoDisplay:
    """
    A text overlay system for displaying information on top of the 3D graphics.
    Uses pygame for text rendering and ModernGL for overlay rendering.
    """
    
    def __init__(self, graphics_engine):
        self.app = graphics_engine
        self.ctx = graphics_engine.ctx
        self.win_size = graphics_engine.WIN_SIZE
        
        # Initialize pygame font
        pg.font.init()
        self.font_size = 20
        self.font = pg.font.Font(None, self.font_size)
        
        # Display settings
        self.visible = False
        self.background_alpha = 0.5
        self.text_color = (255, 255, 255)  # White
        self.background_color = (0, 0, 0)  # Black
        
        # Text content storage - now using dict-based sections
        self.info_sections = []  # List of dicts with section IDs and f-string templates
        self.custom_texts = {}  # For positioned custom text
        
        # Setup OpenGL resources for overlay rendering
        self._setup_overlay_rendering()
        
        # Initialize default info sections
        self._setup_default_info()
    
    def _setup_overlay_rendering(self):
        """Setup OpenGL resources for rendering the text overlay."""
        # Vertex shader for overlay quad
        vertex_shader = """
        #version 330 core
        in vec2 in_position;
        in vec2 in_texcoord;
        out vec2 texcoord;
        
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            texcoord = in_texcoord;
        }
        """
        
        # Fragment shader for overlay
        fragment_shader = """
        #version 330 core
        in vec2 texcoord;
        out vec4 fragColor;
        uniform sampler2D u_texture;
        uniform float u_alpha;
        
        void main() {
            vec4 tex_color = texture(u_texture, texcoord);
            fragColor = vec4(tex_color.rgb, tex_color.a * u_alpha);
        }
        """
        
        self.overlay_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        
        # Create overlay surface
        self.overlay_surface = pg.Surface(self.win_size, pg.SRCALPHA)
        
        # Create texture for overlay
        self.overlay_texture = self.ctx.texture(self.win_size, 4)
        self.overlay_texture.filter = (mgl.LINEAR, mgl.LINEAR)
        
        # Create quad for overlay
        vertices = np.array([
            # Position    # TexCoord
            -1.0, -1.0,   0.0, 1.0,
             1.0, -1.0,   1.0, 1.0,
             1.0,  1.0,   1.0, 0.0,
            -1.0,  1.0,   0.0, 0.0,
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        self.overlay_vbo = self.ctx.buffer(vertices.tobytes())
        self.overlay_ibo = self.ctx.buffer(indices.tobytes())
        
        self.overlay_vao = self.ctx.vertex_array(
            self.overlay_program,
            [(self.overlay_vbo, '2f 2f', 'in_position', 'in_texcoord')],
            self.overlay_ibo
        )
    
    def _setup_default_info(self):
        """Setup default information sections using f-string templates."""
        self.info_sections = [
            {'controls': '''Controls: \n ESC - Exit \n H - Info display \n P - Screenshot \n I - Camera movement interpolation \n'''\
             ''' R - Reset animation \n SPACE - Pause/Resume animation\n ''' \
             ''' LEFT/RIGHT - Step when paused \n UP/DOWN - Change animation speed multiplier'''},
            {'separator': ''},
            {'camera': 'Camera: {self.app.camera.get_camera_str()}' if hasattr(self.app, 'camera') else 'Camera Position: N/A'},
        ]
    
    def toggle_visibility(self):
        self.visible = not self.visible
    
    def add_info_section(self, section_dict: dict):
        """Add a section dictionary to the info display."""
        self.info_sections.append(section_dict)
    
    def update_info_section(self, section_id: str, text: str):
        """Update a specific section in the info display by its ID."""
        for section in self.info_sections:
            if section_id in section:
                section[section_id] = text
                return
        # If section doesn't exist, add it
        self.add_info_section({section_id: text})
    
    def remove_info_section(self, section_id: str):
        """Remove a section from the info display by its ID."""
        self.info_sections = [section for section in self.info_sections if section_id not in section]
    
    def set_custom_text(self, key: str, text: str, position: Tuple[int, int], 
                       font_size: int = None, color: Tuple[int, int, int] = None):
        """Set custom positioned text on the overlay."""
        self.custom_texts[key] = {
            'text': text,
            'position': position,
            'font_size': font_size or self.font_size,
            'color': color or self.text_color
        }
    
    def remove_custom_text(self, key: str):
        """Remove custom text from the overlay."""
        if key in self.custom_texts:
            del self.custom_texts[key]
    
    def _evaluate_f_string(self, template: str):
        """Evaluate f-string templates at render time."""
        try:
            # Use eval with f-string syntax
            return eval(f'f"""{template}"""')
        except Exception as e:
            # If evaluation fails, return the template with error info
            return f"{template} [Error: {str(e)}]"
    
    def _render_section_text(self, section_dict: dict):
        """Render text from a section dictionary, evaluating f-strings."""
        rendered_lines = []
        
        for section_id, content in section_dict.items():
            if section_id == 'separator':
                rendered_lines.append('')
            elif isinstance(content, str):
                if '{' in content and '}' in content:
                    # This looks like an f-string template
                    rendered_text = self._evaluate_f_string(content)
                else:
                    # Regular string
                    rendered_text = content
                
                # Handle multi-line content
                if '\n' in rendered_text:
                    rendered_lines.extend(rendered_text.split('\n'))
                else:
                    rendered_lines.append(rendered_text)
        
        return rendered_lines
    
    def render(self):
        """Render the info display overlay."""
        if not self.visible:
            return
        
        # Clear the overlay surface
        self.overlay_surface.fill((0, 0, 0, 0))
        
        # Render info sections
        y_offset = 10
        line_height = self.font_size + 2
        
        # Process each section and render its content
        for section in self.info_sections:
            rendered_lines = self._render_section_text(section)
            
            for line in rendered_lines:
                if line.strip():  # Skip empty lines for rendering but keep spacing
                    text_surface = self.font.render(line, True, self.text_color)
                    # Add background for better readability
                    bg_rect = text_surface.get_rect()
                    bg_rect.x = 8
                    bg_rect.y = y_offset - 2
                    bg_rect.width += 4
                    bg_rect.height += 2
                    
                    pg.draw.rect(self.overlay_surface, (*self.background_color, int(255 * self.background_alpha)), bg_rect)
                    self.overlay_surface.blit(text_surface, (10, y_offset))
                
                y_offset += line_height
        
        # Render custom texts
        for custom_text in self.custom_texts.values():
            font_to_use = pg.font.Font(None, custom_text['font_size'])
            text_surface = font_to_use.render(custom_text['text'], True, custom_text['color'])
            
            # Add background
            bg_rect = text_surface.get_rect()
            bg_rect.x = custom_text['position'][0] - 2
            bg_rect.y = custom_text['position'][1] - 2
            bg_rect.width += 4
            bg_rect.height += 2
            
            pg.draw.rect(self.overlay_surface, (*self.background_color, int(255 * self.background_alpha)), bg_rect)
            self.overlay_surface.blit(text_surface, custom_text['position'])
        
        # Convert pygame surface to OpenGL texture
        text_data = pg.image.tostring(self.overlay_surface, 'RGBA')
        self.overlay_texture.write(text_data)
        
        # Render overlay using OpenGL
        self.ctx.disable(mgl.DEPTH_TEST)
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
        
        self.overlay_texture.use(0)
        self.overlay_program['u_texture'] = 0
        self.overlay_program['u_alpha'] = 1.0
        
        self.overlay_vao.render()
        
        self.ctx.enable(mgl.DEPTH_TEST)
    
    def destroy(self):
        """Clean up OpenGL resources."""
        if hasattr(self, 'overlay_vbo'):
            self.overlay_vbo.release()
        if hasattr(self, 'overlay_ibo'):
            self.overlay_ibo.release()
        if hasattr(self, 'overlay_vao'):
            self.overlay_vao.release()
        if hasattr(self, 'overlay_texture'):
            self.overlay_texture.release()
        if hasattr(self, 'overlay_program'):
            self.overlay_program.release()
