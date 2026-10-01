#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 vertexColor;

layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 baseColor;
} ubo;

void main() {
    vec4 world = ubo.model * vec4(inPosition, 1.0);
    worldPosition = world.xyz;
    vertexColor = vec3(0.35) + 0.65 * (inPosition + vec3(1.0)) * 0.5;
    gl_Position =
        ubo.projection *
        ubo.view *
        world;
}
