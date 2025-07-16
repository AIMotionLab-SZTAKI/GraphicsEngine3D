#version 450 core

// Attributes
in vec3 a_position;         // attribute position vector in the instance coordinate system
in vec3 a_normal;           // attribute normal   vector in the instance coordinate system
in vec3 a_instance_index;   // attribute instance index // changes for each rendered instance
in float a_instance_value;  // attribute instance index // changes for each rendered instance

out vec3 v_position;     // v: variable, passed to the fragment shader
out vec3 v_normal;
out vec4 v_color;
out vec3 v_fragPos;

// Uniforms
uniform ivec3 shape;    // shape of the whole instanced grid
uniform mat4 m_model;   // model matrix
uniform mat4 m_view;    // view matrix
uniform mat4 m_proj;    // projection matrix


vec3 HUEtoRGB(in float H){
    float R = abs(H * 6.0 - 3.0) - 1.0;
    float G = 2.0 - abs(H * 6.0 - 2.0);
    float B = 2.0 - abs(H * 6.0 - 4.0);
    return clamp( vec3(R,G,B), 0.0, 1.0 );
}

vec3 Colormap(in float val){
    float R = -2.667*val*val + 2.784*val + 0.184;
    float G = -2.129*val*val + 1.789*val + 0.491;
    float B = -1.273*val*val + 1.034*val + 0.264;
    return clamp( vec3(R,G,B), 0.0, 1.0 );
}

void main()
{
    // scale the whole grid to fit in a unit sized cube in the model coordinate system
    vec3 scale = 1.0 / vec3(shape);         
    scale = vec3(min(scale.x, min(scale.y, scale.z)));

    // Translation vector:  (index - (shape-1)/2) -> move to center, then apply the scaling
    vec3 translation = 2.0 * scale * ( vec3(a_instance_index) - vec3(shape-1)/2.0 ); 

    // transform the vertices from instance coord system to the model space
    mat4 m_instance = mat4(  
        vec4(scale.x, 0.0, 0.0, 0.0),
        vec4(0.0, scale.y, 0.0, 0.0),
        vec4(0.0, 0.0, scale.z, 0.0),
        vec4(translation,       1.0)
    );

    mat4 model_view = m_view * m_model * m_instance;          // matrix to transfrom from instance -> model -> view coordinate system
    vec4 view_pos   = model_view * vec4(a_position.xyz, 1.0); // position in the view coord system

    v_fragPos       = vec3(m_model * vec4(a_position, 1.0));  // fragment coord in the global system
    v_position      = view_pos.xyz;                           // position in the view coord system

    v_normal      = transpose(inverse(mat3(model_view))) * normalize(a_normal);  // normal in the camera (view) system (legacy)
    //v_normal =      vec3(m_model*m_instance*vec4(normalize(a_normal), 1.0));
    v_color         = vec4(0.6 * HUEtoRGB(a_instance_value), 0.5);

    gl_Position     = m_proj * view_pos;
}