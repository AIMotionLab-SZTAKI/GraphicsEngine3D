#version 330 core

in vec3 in_vert;

// Uniforms
uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 m_model;

void main() 
{
    gl_Position = m_proj * m_view * m_model * vec4(in_vert, 1.0);;
}