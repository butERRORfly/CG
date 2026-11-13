#version 450

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 textureUV;
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

// Один дескриптор объединяет изображение и настройки выборки.
layout(set = 1, binding = 1) uniform sampler2D albedoTexture;

// Параметры эффектов передаются через push constants перед рисованием модели.
// Порядок и размер (28 байт) совпадают с C++ SamplingConstants.
layout(push_constant, std430) uniform SamplingConstants {
    int enabled;
    float time;
    float strength;
    float uvScale;
    int tvEnabled;
    float tvTime;
    float tvStrength;
} sampling;

// Процедурный псевдослучайный отсчёт: одинаковые координаты дают одинаковый шум.
// Дискретный номер кадра меняет рисунок помех примерно 30 раз в секунду.
float tvRandom(vec2 value) {
    return fract(sin(dot(value, vec2(127.1, 311.7))) * 43758.5453);
}

vec3 sampleImage(vec2 uv) {
    if (sampling.tvEnabled == 0)
        return texture(albedoTexture, uv).rgb;

    // Цветовые каналы одного изображения читаем по различным UV.
    vec2 separation = vec2(0.004 * sampling.tvStrength, 0.0);
    return vec3(texture(albedoTexture, uv + separation).r,
                texture(albedoTexture, uv).g,
                texture(albedoTexture, uv - separation).b);
}

vec3 sampleAlbedo() {
    vec2 uv = textureUV * sampling.uvScale;
    float frame = mod(floor(sampling.tvTime * 30.0), 1024.0);
    float band = 0.0;
    if (sampling.tvEnabled != 0) {
        // Полоса плохой синхронизации бежит по поверхности, отдельные строки
        // дрожат независимо. REPEAT обрабатывает выход искажённых UV за [0,1].
        float distanceToBand = abs(fract(textureUV.y - sampling.tvTime * 0.2 + 0.5) - 0.5);
        band = 1.0 - smoothstep(0.015, 0.07, distanceToBand);
        float row = floor(textureUV.y * 60.0);
        float jitter = (tvRandom(vec2(row, frame)) - 0.5) * 0.018;
        uv.x += sampling.tvStrength * (jitter + 0.06 * band);
    }

    vec3 base = sampleImage(uv);
    vec3 albedo = base;
    // То же изображение дополнительно читаем по двум другим UV.
    // Сдвиги, смешивание и поканальная модуляция меняются со временем.
    // Все функции периодичны по 2*pi, как и время, накапливаемое на CPU.
    if (sampling.enabled != 0) {
        float t = sampling.time;
        vec2 uvA = uv + 0.12 * vec2(sin(t), cos(t));
        vec2 uvB = uv * 1.7 + 0.15 * vec2(cos(2.0 * t), -sin(t));
        vec3 first = sampleImage(uvA);
        vec3 second = sampleImage(uvB);
        float blend = 0.5 + 0.5 * sin(t);
        vec3 modulation = 0.75 + 0.25 * sin(vec3(t) + vec3(0.0, 2.0943951, 4.1887902));
        vec3 animated = mix(first, second, blend) * modulation;
        albedo = mix(base, animated, sampling.strength);
    }

    if (sampling.tvEnabled != 0) {
        // «Снег» берётся из виртуальной сетки 320x240; scanlines затемняют строки.
        // При уменьшении фигуры fwidth ослабляет мелкие полосы против муара.
        vec2 cell = floor(textureUV * vec2(320.0, 240.0));
        float grain = tvRandom(cell + vec2(17.0, 53.0) * frame);
        float lineContrast = 1.0 - smoothstep(0.5, 1.0, fwidth(textureUV.y) * 240.0);
        float scanlines = 1.0 - 0.14 * lineContrast *
            (0.5 + 0.5 * cos(textureUV.y * 240.0 * 6.2831853));
        float flicker = 0.92 + 0.08 * tvRandom(vec2(frame, 19.0));
        vec3 interference = albedo * scanlines * flicker * (1.0 - 0.35 * band);
        interference += vec3((grain - 0.5) * 0.35 + step(0.985, grain) * 0.35);
        albedo = mix(albedo, clamp(interference, 0.0, 1.0), sampling.tvStrength);
    }
    // Оба эффекта меняют альбедо до расчёта света, а не создают emissive-вклад.
    return albedo * modelData.albedo.rgb;
}

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

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo) {
    // Закон Ламберта: диффузный вклад = Kd * max(N*L, 0), N и L нормализованы.
    float diffuse = max(dot(N, L), 0.0);
    if (diffuse <= 0.0)
        return vec3(0.0); // Свет за поверхностью не создаёт диффузную часть и блик.

    // H — половинчатый вектор между направлением к камере и к свету.
    vec3 H = safeNormalize(V + L);
    float specular = pow(max(dot(N, H), 0.0), modelData.specularShininess.w);
    // Блинн–Фонг: C = I * (Kd*max(N*L,0) + Ks*max(N*H,0)^alpha).
    return radiance * (albedo * diffuse +
                       modelData.specularShininess.rgb * specular);
}

void main() {
    // Альбедо вычисляем один раз на фрагмент и используем для всех источников.
    // Формат sRGB автоматически переводит RGB текстуры в линейное пространство.
    vec3 albedo = sampleAlbedo();
    // Интерполяция и масштаб могут изменить длину нормали: перед dot нормализуем.
    vec3 N = safeNormalize(worldNormal);
    vec3 V = scene.cameraPosition.w > 0.5
        ? safeNormalize(scene.cameraPosition.xyz - worldPosition)
        : safeNormalize(scene.viewDirection.xyz); // В ортографии лучи параллельны.

    // Фоновый вклад: C_ambient = цвет окружения * интенсивность * альбедо.
    vec3 color = scene.ambientColorIntensity.rgb * scene.ambientColorIntensity.w * albedo;

    // Направленный свет: лучи параллельны, затухания с расстоянием нет.
    // В UI direction задаёт ход лучей, а L должен указывать ОТ поверхности К свету.
    vec3 L = safeNormalize(-scene.lightDirection.xyz);
    color += blinnPhong(N, V, L, scene.lightColorIntensity.rgb * scene.lightColorIntensity.w, albedo);

    // Расчет света.
    for (int i = 0; i < scene.lightCounts.x; ++i) {
        PointLight light = pointLights[i];
        vec3 toLight = light.position.xyz - worldPosition;
        // Затухание falloff=1/D²; ограничиваем D² снизу, чтобы избежать деления на ноль.
        float distanceSquared = max(dot(toLight, toLight), 0.0001);
        vec3 radiance = light.colorIntensity.rgb * light.colorIntensity.w / distanceSquared;
        color += blinnPhong(N, V, safeNormalize(toLight), radiance, albedo);
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
        color += blinnPhong(N, V, directionToLight, radiance, albedo);
    }

    // Альфа 1 задаёт непрозрачность; видимость фрагмента определяет тест глубины.
    outColor = vec4(color, 1.0);
}
