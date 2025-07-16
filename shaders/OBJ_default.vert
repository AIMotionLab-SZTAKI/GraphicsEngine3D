#version 330 core

layout (location = 0) in vec2 in_texcoord_0;    // texture coordinate
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec3 in_position;

out vec2 uv_0;      // texture coordinate
out vec3 normal;    
out vec3 fragPos;   // fragment (pixel) position

uniform mat4 m_proj;    // projection transformation matrix
uniform mat4 m_view;    // view transformation matrix
uniform mat4 m_model;   // model transformation matrix

void main() {
    uv_0 = in_texcoord_0;
    fragPos = vec3(m_model * vec4(in_position, 1.0));
    normal = mat3(transpose(inverse(m_model))) * normalize(in_normal);
    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
}