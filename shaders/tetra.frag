#version 450

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 0) out vec4 outColor;

// Общий UBO сцены: камера, ambient, directional и числа активных источников.
layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 viewDirection;
    vec4 ambientColorIntensity;
    vec4 lightDirection;
    vec4 lightColorIntensity;
    ivec4 lightCounts;
} scene;

// Материал: Kd (albedo), Ks (RGB блика), alpha (shininess, степень блика).
layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specularShininess;
} modelData;

// SSBO с раскладкой std430 хранят массивы света без фиксированной длины в шейдере.
struct PointLight {
    vec4 position;
    vec4 colorIntensity;
};
struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 colorIntensity;
    vec4 angleCosines;
};

// Массив источников.
layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight pointLights[];
};
layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spotLights[];
};

vec3 safeNormalize(vec3 value) {
    // Нулевая direction или совпадение позиции камеры/света с точкой не дадут NaN.
    return value * inversesqrt(max(dot(value, value), 0.000001));
}

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance) {
    // Закон Ламберта: диффузный вклад = Kd * max(N·L, 0), N и L нормализованы.
    float diffuse = max(dot(N, L), 0.0);
    if (diffuse <= 0.0)
        return vec3(0.0); // Свет за поверхностью не создаёт диффузную часть и блик.

    // H — половинчатый вектор между направлением к камере и к свету.
    vec3 H = safeNormalize(V + L);
    float specular = pow(max(dot(N, H), 0.0), modelData.specularShininess.w);
    // Блинн–Фонг: C = I * (Kd*max(N·L,0) + Ks*max(N·H,0)^alpha).
    return radiance * (modelData.albedo.rgb * diffuse +
                       modelData.specularShininess.rgb * specular);
}

void main() {
    // Интерполяция и масштаб могут изменить длину нормали: перед dot нормализуем.
    vec3 N = safeNormalize(worldNormal);
    vec3 V = scene.cameraPosition.w > 0.5
        ? safeNormalize(scene.cameraPosition.xyz - worldPosition)
        : safeNormalize(scene.viewDirection.xyz); // В ортографии лучи параллельны.

    // Фоновый вклад: C_ambient = цвет окружения * интенсивность * альбедо.
    vec3 color = scene.ambientColorIntensity.rgb * scene.ambientColorIntensity.w * modelData.albedo.rgb;

    // Направленный свет: лучи параллельны, затухания с расстоянием нет.
    // В UI direction задаёт ход лучей, а L должен указывать ОТ поверхности К свету.
    vec3 L = safeNormalize(-scene.lightDirection.xyz);
    color += blinnPhong(N, V, L, scene.lightColorIntensity.rgb * scene.lightColorIntensity.w);

    // Расчет света.
    for (int i = 0; i < scene.lightCounts.x; ++i) {
        PointLight light = pointLights[i];
        vec3 toLight = light.position.xyz - worldPosition;
        // Затухание falloff=1/D²; ограничиваем D² снизу, чтобы избежать деления на ноль.
        float distanceSquared = max(dot(toLight, toLight), 0.0001);
        vec3 radiance = light.colorIntensity.rgb * light.colorIntensity.w / distanceSquared;
        color += blinnPhong(N, V, safeNormalize(toLight), radiance);
    }

    for (int i = 0; i < scene.lightCounts.y; ++i) {
        SpotLight light = spotLights[i];
        vec3 toLight = light.position.xyz - worldPosition;
        float distanceSquared = max(dot(toLight, toLight), 0.0001);
        vec3 directionToLight = safeNormalize(toLight);
        // -L направлен от прожектора к поверхности: сравниваем его с осью конуса.
        float cosine = dot(safeNormalize(light.direction.xyz), -directionToLight);
        float innerCosine = light.angleCosines.x;
        float outerCosine = light.angleCosines.y;
        // Между outer и inner плавно меняем вклад 0..1. При равных углах край резкий.
        float cone = innerCosine > outerCosine
            ? smoothstep(outerCosine, innerCosine, cosine)
            : step(outerCosine, cosine);
        vec3 radiance = light.colorIntensity.rgb * light.colorIntensity.w * cone / distanceSquared;
        color += blinnPhong(N, V, directionToLight, radiance);
    }

    // Альфа 1 задаёт непрозрачность; видимость фрагмента определяет тест глубины.
    outColor = vec4(color, 1.0);
}
