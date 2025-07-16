#version 450 core

in vec3 v_position; // position in the view coordinate system
in vec3 v_normal;   // normal vector of the plane stretched out by vertices
in vec4 v_color;    // input color of the vertex plane which is later shaded
in vec3 v_fragPos;    // fragment (pixel) position in the global system

layout (location = 0) out vec4 fragColor;

struct Light {
    vec3 position;  // light position in the global system
    vec3 Ia;        // ambient intensity
    vec3 Id;        // diffuse intensity
    //vec3 Is;        // specular intensity
};

uniform Light light;                // uniform: same for all instances
uniform vec3 camPos;                // camera position in the global coordinate system
uniform mat4 m_view;

vec3 getLight(vec3 color) {
    vec3 Normal = normalize(v_normal);  // normal global system

    // ambient light
    vec3 ambient = light.Ia + 0.22;

    // diffuse light (from Light source)
    vec3 lightDir = normalize(light.position - v_fragPos);  
    float diff = max(0, dot(lightDir, Normal));
    vec3 diffuse = diff * light.Id;


    // diffuse light (from Camera position --- flashlight effect)
    vec3 viewDir = normalize(camPos - v_fragPos);  
    float diff2 = max(0, dot(viewDir, Normal));
    vec3 diffuse2 = diff2 * vec3(0.15);

    // specular light
    //vec3 viewDir = normalize(camPos - v_fragPos);
    //vec3 reflectDir = reflect(-lightDir, Normal);
    //float spec = pow(max(dot(viewDir, reflectDir), 0), 32);
    //vec3 specular = spec * light.Is;

    return v_color.xyz * (ambient + diffuse * 0.01 + diffuse2*0.01);
}

void main() {
    float gamma = 2.2;
    //color = pow(v_color, vec3(gamma)); // inverse gamma correction (not needed for RGB colors, only textures)
    vec3 color = getLight(v_color.xyz);                 // Lightning
    fragColor = vec4(pow(color, 1 / vec3(gamma)), 0.16); // gamma correction (0.18)
}
