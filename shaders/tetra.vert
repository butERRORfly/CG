#version 450

// Нормаль N = normalize((B-A) × (C-A)) вычислена на CPU и передана атрибутом.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;

// Общие параметры камеры и освещения находятся в наборе дескрипторов set 0.
// Порядок полей и std140 должны совпадать с C++ SceneUniforms.
layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 viewDirection;
    vec4 ambientColorIntensity;
    vec4 lightDirection;
    vec4 lightColorIntensity;
    ivec4 lightCounts;
} scene;

// Параметры конкретной модели отделены в set 1.
layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specularShininess;
} modelData;

void main() {
    // w=1 включает перенос точки: локальная позиция -> мировая.
    vec4 world = modelData.model * vec4(inPosition, 1.0);
    worldPosition = world.xyz;
    // Для нормали нужна обратная транспонированная линейная часть model.
    // Перенос на направление не действует; mat3 берёт только верхний блок 3x3.
    worldNormal = mat3(modelData.normalMatrix) * inNormal;
    // p_clip = P*V*M*p_local. После шейдера GPU делит x, y и z на w.
    gl_Position = scene.viewProjection * world;
}
