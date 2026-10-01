#version 450

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 vertexColor;
layout(location = 0) out vec4 outColor;

layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 baseColor;
} ubo;

void main() {
    vec3 tangentX = dFdx(worldPosition);
    vec3 tangentY = dFdy(worldPosition);
    vec3 normal = normalize(-cross(tangentX, tangentY));
    vec3 directionToLight = normalize(vec3(1.0, 0.3, 1.0));
    float diffuse = max(dot(normal, directionToLight), 0.0);
    float brightness = 0.38 + 0.62 * diffuse;
    outColor = vec4(ubo.baseColor.rgb * vertexColor * brightness, 1.0);
}
