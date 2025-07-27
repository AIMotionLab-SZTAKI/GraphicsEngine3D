#version 330 core

in vec2 uv_0;       // texture coordinate
in vec3 normal;     // normal vector of the plane
in vec3 fragPos;    // fragment (pixel) position

layout (location = 0) out vec4 fragColor;

struct Light {
    vec3 position;  // light position
    vec3 Ia;        // ambient intensity
    vec3 Id;        // diffuse intensity
    vec3 Is;        // specular intensity
};

uniform Light light;                // uniform: same for all instances 
uniform sampler2D u_texture_0;      
uniform vec3 camPos;                // camera position in the global coordinate system
uniform float alpha;                // alpha value for transparency


vec3 getLight(vec3 color) {
    vec3 Normal = normalize(normal);

    // ambient light
    vec3 ambient = light.Ia;

    // diffuse light
    vec3 lightDir = normalize(light.position - fragPos);
    float diff = max(0, dot(lightDir, Normal));
    vec3 diffuse = diff * light.Id;

    // specular light
    vec3 viewDir = normalize(camPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, Normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0), 32);
    vec3 specular = spec * light.Is;

    return color * (ambient + diffuse + specular);
}


void main() {
    float gamma = 2.2;                      // gamma correction

    // Alter the color
    vec3 color = texture(u_texture_0, uv_0).rgb;

    color = pow(color, vec3(gamma));        // gamma correction
    color = getLight(color);                // Lightning
    color = pow(color, 1 / vec3(gamma));    // gamma correction

    fragColor = vec4(color, alpha);
}










