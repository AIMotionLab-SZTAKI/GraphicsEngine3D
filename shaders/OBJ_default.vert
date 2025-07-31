#version 330 core

in vec2 in_texcoord_0;    // texture coordinate
in vec3 in_normal;
in vec3 in_position;

out vec2 uv_0;      // texture coordinate
out vec3 normal;    // fragment (pixel) normal
out vec3 fragPos;   // fragment (pixel) position
out vec3 viewPos;   // position in view space

uniform mat4 m_proj;        // projection transformation matrix
uniform mat4 m_view;        // view transformation matrix
uniform mat4 m_model;       // model transformation matrix
uniform mat4 m_instance;    // instance transformation matrix

void main() {
    // Transform vertex from instance space -> model space -> world space -> view space
    vec4 world_pos = m_model * m_instance * vec4(in_position, 1.0);
    vec4 view_pos = m_view * world_pos;

    uv_0 = in_texcoord_0;
    fragPos = world_pos.xyz;
    viewPos = view_pos.xyz;
    normal = normalize(transpose(inverse(mat3(m_model * m_instance))) * in_normal); // Transform normal to world space // Use normal matrix for proper normal transformation
    gl_Position = m_proj * view_pos;
}