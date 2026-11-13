#include "application.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <imgui.h>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

// PNG/JPEG декодируется на CPU в плотно упакованный массив пикселей RGBA8.
// Реализация stb_image включается ровно в одном cpp-файле.
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace {
// Точка/направление в 3D. Для плоских граней нормаль вычислим через cross двух
// рёбер.
struct Vec3 {
  float x;
  float y;
  float z;
};

// UV задаёт положение вершины в двумерном пространстве текстуры.
struct Vertex {
  Vec3 position;
  Vec3 normal;
  float uv[2];
};

// GLSL mat4 хранит 4 столбца по 4 float; индекс элемента: column * 4 + row.
struct alignas(16) Mat4 {
  float m[16];
};

// set 0 содержит общие для всей сцены параметры камеры и освещения.
// Используем группы по 4 числа, чтобы раскладка C++ совпадала с std140 в GLSL.
struct alignas(16) SceneUniforms {
  Mat4 viewProjection;
  float cameraPosition[4]; // w=1 для перспективы, w=0 для ортографии.
  float viewDirection[4];  // Направление к наблюдателю при параллельных лучах.
  float ambientColorIntensity[4];
  float lightDirection[4];
  float lightColorIntensity[4];
  std::int32_t lightCounts[4]; // x — число point lights, y — число spot lights.
};

// set 1: преобразование и материал именно рисуемой модели.
struct alignas(16) ModelUniforms {
  Mat4 model;
  Mat4 normalMatrix;
  float albedo[4];
  float
      specularShininess[4]; // RGB блика и показатель степени m из Блинн–Фонга.
};

// Короткие параметры сэмплирования записываются в команды GPU,
// а матрицы и материал по-прежнему хранятся в динамическом UBO.
struct SamplingConstants {
  std::int32_t enabled;
  float time;
  float strength;
  float uvScale;
  std::int32_t tvEnabled;
  float tvTime;
  float tvStrength;
};

// Точечный источник задаётся позицией, цветом и интенсивностью; массив
// источников передаётся через SSBO с раскладкой std430.
struct alignas(16) PointLight {
  float position[4];
  float colorIntensity[4]; // RGB — цвет, w — интенсивность.
};

struct alignas(16) SpotLight {
  float position[4];
  float direction[4]; // Направление лучей ОТ прожектора в мир.
  float colorIntensity[4];
  float angleCosines[4]; // x — cos(inner), y — cos(outer); остальные —
                         // выравнивание.
};

// VkBuffer — ресурс, allocation — выделенная ему память, capacity — размер в
// байтах.
struct Buffer {
  VkBuffer handle = VK_NULL_HANDLE;
  VmaAllocation allocation = nullptr;
  VkDeviceSize capacity = 0;
};

struct Texture {
  VkImage image = VK_NULL_HANDLE;
  VmaAllocation allocation = nullptr;
  VkImageView view = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  std::string path; // Фактический файл; пустая строка означает служебную текстуру.
};

static_assert(sizeof(Vertex) == 32);
static_assert(offsetof(Vertex, uv) == 24);
static_assert(sizeof(SamplingConstants) == 28);
static_assert(offsetof(SamplingConstants, tvEnabled) == 16);
static_assert(sizeof(Mat4) == 64);
static_assert(sizeof(SceneUniforms) == 160);
static_assert(offsetof(SceneUniforms, lightCounts) == 144);
static_assert(sizeof(ModelUniforms) == 160);
static_assert(offsetof(ModelUniforms, albedo) == 128);
static_assert(sizeof(PointLight) == 32);
static_assert(sizeof(SpotLight) == 64);

constexpr Mat4 kIdentity{{
    1.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    1.0f,
}};

constexpr float kRadiansPerDegree = std::numbers::pi_v<float> / 180.0f;

enum class ProjectionMode {
  Perspective,
  Orthographic,
};

ProjectionMode projectionMode = ProjectionMode::Perspective;
float objectPosition[3] = {0.0f, 0.0f, 0.0f};
float objectRotationDegrees[3] = {20.0f, -25.0f, 0.0f};
float objectScale[3] = {1.0f, 1.0f, 1.0f};
float objectColor[3] = {1.0f, 1.0f, 1.0f}; // Белый множитель сохраняет цвет текстуры.
float specularColor[3] = {1.0f, 1.0f, 1.0f};
float shininess = 32.0f;

// Камера имеет собственные положение и ориентацию, независимо от модели.
float cameraPosition[3] = {0.0f, 0.0f, 3.5f};
float cameraRotationDegrees[3] = {0.0f, 0.0f, 0.0f};
float cameraFovDegrees = 60.0f;
constexpr float cameraNear = 0.1f; // Фиксированная ближняя граница видимости.
constexpr float cameraFar = 20.0f; // Фиксированная дальняя граница видимости.
float orthographicHalfHeight = 3.5f * std::tan(30.0f * kRadiansPerDegree);

float ambientColor[3] = {1.0f, 1.0f, 1.0f};
float ambientIntensity = 0.12f;
bool directionalLightEnabled = true;
float lightDirection[3] = {-1.0f, -0.3f, -1.0f};
float lightColor[3] = {1.0f, 1.0f, 1.0f};
float lightIntensity = 0.8f;
// При пустых массивах точечных источников и прожекторов поверхность освещают
// только направленный источник и фоновый свет.
std::vector<PointLight> pointLights;
std::vector<SpotLight> spotLights;
VkDeviceSize maxStorageBufferBytes = 0;

constexpr std::array<const char *, 3> kTexturePaths{
    "textures/1", "textures/2", "textures/3"};
std::array<Texture, kTexturePaths.size()> textures{};
int selectedTexture = 0;
bool samplingEnabled = false;
float samplingSpeed = 1.0f;
float samplingStrength = 0.7f;
float uvScale = 1.0f;
double samplingTime = 0.0;
bool tvNoiseEnabled = false;
float tvNoiseSpeed = 1.0f;
float tvNoiseStrength = 0.7f;
double tvNoiseTime = 0.0;
// Один буфер рассчитан на массив моделей; сейчас используется слот 0.
constexpr std::uint32_t kMaxModels = 1024;
constexpr std::uint32_t kModelIndex = 0;
VkDeviceSize modelUniformStride = 0;

bool animationPlaying = false;
float animationSpeedY = 0.8f;
float animationSpeedX = 1.3f;
double animationPhaseY = 0.0;
double animationPhaseX = 0.0;
double previousUpdateTime = -1.0;

constexpr float t = 1.0f / 3.0f;

// Усечённый тетраэдр: 12 уникальных точек, 4 треугольные и 4 шестиугольные
// грани.
constexpr std::array<Vec3, 12> kVertices{{
    {1.0f, t, t}, //  0: p01
    {t, 1.0f, t}, //  1: p02
    {t, t, 1.0f}, //  2: p03

    {1.0f, -t, -t}, //  3: p10
    {t, -t, -1.0f}, //  4: p12
    {t, -1.0f, -t}, //  5: p13

    {-t, 1.0f, -t}, //  6: p20
    {-t, t, -1.0f}, //  7: p21
    {-1.0f, t, -t}, //  8: p23

    {-t, -t, 1.0f}, //  9: p30
    {-t, -1.0f, t}, // 10: p31
    {-1.0f, -t, t}, // 11: p32
}};

constexpr std::array<std::uint16_t, 60> kIndices{{
    // Четыре треугольные грани
    0,
    1,
    2,
    5,
    4,
    3,
    6,
    7,
    8,
    11,
    10,
    9,

    // Шестиугольник a1-a2-a3
    5,
    10,
    11,
    5,
    11,
    8,
    5,
    8,
    7,
    5,
    7,
    4,

    // Шестиугольник a0-a2-a3
    1,
    6,
    8,
    1,
    8,
    11,
    1,
    11,
    9,
    1,
    9,
    2,

    // Шестиугольник a0-a1-a3
    2,
    9,
    10,
    2,
    10,
    5,
    2,
    5,
    3,
    2,
    3,
    0,

    // Шестиугольник a0-a1-a2
    0,
    3,
    4,
    0,
    4,
    7,
    0,
    7,
    6,
    0,
    6,
    1,
}};

static_assert(kIndices.size() % 3 == 0);

// У соседних граней одна позиция, но разные нормали. Поэтому для GPU дублируем
// углы треугольников: 60 записей position+normal+UV, геометрических точек всё
// ещё 12.
std::array<Vertex, kIndices.size()> meshVertices{};
std::array<std::uint16_t, kIndices.size()> meshIndices{};

void generateMeshNormals() {
  for (std::size_t i = 0; i < kIndices.size(); i += 3) {
    const Vec3 a = kVertices[kIndices[i]];
    const Vec3 b = kVertices[kIndices[i + 1]];
    const Vec3 c = kVertices[kIndices[i + 2]];
    const Vec3 u{b.x - a.x, b.y - a.y, b.z - a.z};
    const Vec3 v{c.x - a.x, c.y - a.y, c.z - a.z};
    // cross(b-a, c-a) перпендикулярен плоскости; порядок индексов задаёт
    // наружу.
    Vec3 normal{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z,
                u.x * v.y - u.y * v.x};
    const float length = std::sqrt(normal.x * normal.x + normal.y * normal.y +
                                   normal.z * normal.z);
    normal.x /= length;
    normal.y /= length;
    normal.z /= length;

    // Проецируем позицию на две взаимно перпендикулярные оси плоскости грани.
    // Оси зависят только от нормали, поэтому внутри шестиугольника UV общие
    // для всех четырёх треугольников, а не отдельная картинка на каждом.
    const Vec3 reference = std::abs(normal.y) < 0.9f
                               ? Vec3{0.0f, 1.0f, 0.0f}
                               : Vec3{1.0f, 0.0f, 0.0f};
    Vec3 axisU{reference.y * normal.z - reference.z * normal.y,
               reference.z * normal.x - reference.x * normal.z,
               reference.x * normal.y - reference.y * normal.x};
    const float uLength = std::sqrt(axisU.x * axisU.x + axisU.y * axisU.y +
                                     axisU.z * axisU.z);
    axisU.x /= uLength;
    axisU.y /= uLength;
    axisU.z /= uLength;
    const Vec3 axisV{normal.y * axisU.z - normal.z * axisU.y,
                     normal.z * axisU.x - normal.x * axisU.z,
                     normal.x * axisU.y - normal.y * axisU.x};
    const float uvFactor = 0.5f / std::sqrt(1.0f + 2.0f * t * t);
    for (std::size_t corner = 0; corner < 3; ++corner) {
      const Vec3 p = kVertices[kIndices[i + corner]];
      meshVertices[i + corner] = {
          p, normal,
          {0.5f + (p.x * axisU.x + p.y * axisU.y + p.z * axisU.z) * uvFactor,
           0.5f - (p.x * axisV.x + p.y * axisV.y + p.z * axisV.z) * uvFactor}};
      meshIndices[i + corner] = static_cast<std::uint16_t>(i + corner);
    }
  }
}

// Ресурсы Vulkan создаются один раз; меняющиеся параметры записываются в
// буферы.
VkShaderModule vertexShader = VK_NULL_HANDLE;
VkShaderModule fragmentShader = VK_NULL_HANDLE;
VkDescriptorSetLayout sceneSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout modelSetLayout = VK_NULL_HANDLE;
VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipeline graphicsPipeline = VK_NULL_HANDLE;
Buffer vertexBuffer;
Buffer indexBuffer;
Buffer sceneBuffer;
Buffer modelBuffer;
Buffer pointLightBuffer;
Buffer spotLightBuffer;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet sceneDescriptorSet = VK_NULL_HANDLE;
VkSampler textureSampler = VK_NULL_HANDLE;
VkCommandPool uploadCommandPool = VK_NULL_HANDLE;
VkCommandBuffer uploadCommandBuffer = VK_NULL_HANDLE;
VkFence uploadFence = VK_NULL_HANDLE;

bool loadShaderModule(const char *path, VkShaderModule &result) {
  // Читаем уже скомпилированный glslc двоичный SPIR-V, не текст GLSL.
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Cannot open shader: " << path << '\n';
    return false;
  }

  const std::streampos endPosition = file.tellg();
  if (endPosition <= 0) {
    std::cerr << "Shader is empty: " << path << '\n';
    return false;
  }

  const std::size_t byteCount = static_cast<std::size_t>(endPosition);
  if (byteCount % sizeof(std::uint32_t) != 0) {
    std::cerr << "Invalid SPIR-V size: " << path << '\n';
    return false;
  }

  std::vector<std::uint32_t> code(byteCount / sizeof(std::uint32_t));
  file.seekg(0);

  if (!file.read(reinterpret_cast<char *>(code.data()),
                 static_cast<std::streamsize>(byteCount))) {
    std::cerr << "Cannot read shader: " << path << '\n';
    return false;
  }

  const VkShaderModuleCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = byteCount,
      .pCode = code.data(),
  };

  const VkResult status = vkCreateShaderModule(
      graphics::internal::context.device, &info, nullptr, &result);

  if (status != VK_SUCCESS) {
    std::cerr << "Cannot create shader module: " << path << '\n';
    return false;
  }

  return true;
}

void destroyPipelineSetup() {
  VkDevice device = graphics::internal::context.device;

  if (graphicsPipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    graphicsPipeline = VK_NULL_HANDLE;
  }

  if (pipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    pipelineLayout = VK_NULL_HANDLE;
  }
  if (sceneSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device, sceneSetLayout, nullptr);
    sceneSetLayout = VK_NULL_HANDLE;
  }
  if (modelSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device, modelSetLayout, nullptr);
    modelSetLayout = VK_NULL_HANDLE;
  }
  if (fragmentShader != VK_NULL_HANDLE) {
    vkDestroyShaderModule(device, fragmentShader, nullptr);
    fragmentShader = VK_NULL_HANDLE;
  }
  if (vertexShader != VK_NULL_HANDLE) {
    vkDestroyShaderModule(device, vertexShader, nullptr);
    vertexShader = VK_NULL_HANDLE;
  }
}

bool createGraphicsPipeline() {
  // Графический конвейер: вершинный и фрагментный шейдеры, треугольники,
  // проверка глубины, область изображения viewport и прямоугольник scissor.
  auto &context = graphics::internal::context;

  const VkPipelineShaderStageCreateInfo stages[2] = {
      {
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT,
          .module = vertexShader,
          .pName = "main",
      },
      {
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = fragmentShader,
          .pName = "main",
      },
  };

  const VkVertexInputBindingDescription vertexBinding{
      .binding = 0,
      .stride = static_cast<std::uint32_t>(sizeof(Vertex)),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
  };

  // location 2 передаёт UV; шаг между вершинами составляет 32 байта.
  const VkVertexInputAttributeDescription attributes[3] = {
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
      {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
      {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
  };

  const VkPipelineVertexInputStateCreateInfo vertexInput{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertexBinding,
      .vertexAttributeDescriptionCount = 3,
      .pVertexAttributeDescriptions = attributes,
  };

  const VkPipelineInputAssemblyStateCreateInfo assembly{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
  };

  const VkPipelineViewportStateCreateInfo viewportState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
  };

  const VkPipelineRasterizationStateCreateInfo rasterization{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_CLOCKWISE,
      .lineWidth = 1.0f,
  };

  const VkPipelineMultisampleStateCreateInfo multisampling{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
  };

  const VkPipelineDepthStencilStateCreateInfo depthStencil{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
  };

  const VkPipelineColorBlendAttachmentState colorAttachment{
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
  };

  const VkPipelineColorBlendStateCreateInfo colorBlend{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &colorAttachment,
  };

  const VkDynamicState dynamicStates[2] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
  };

  const VkPipelineDynamicStateCreateInfo dynamicState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2,
      .pDynamicStates = dynamicStates,
  };

  const VkGraphicsPipelineCreateInfo pipelineInfo{
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vertexInput,
      .pInputAssemblyState = &assembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterization,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depthStencil,
      .pColorBlendState = &colorBlend,
      .pDynamicState = &dynamicState,
      .layout = pipelineLayout,
      .renderPass = context.render_pass,
      .subpass = 0,
  };

  if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
                                &pipelineInfo, nullptr,
                                &graphicsPipeline) != VK_SUCCESS) {
    std::cerr << "Cannot create graphics pipeline\n";
    return false;
  }

  return true;
}

bool createHostBuffer(const void *source, VkDeviceSize byteCount,
                        VkBufferUsageFlags usage, Buffer &buffer) {
  auto allocator = graphics::internal::context.allocator;

  const VkBufferCreateInfo bufferInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = byteCount,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };

  const VmaAllocationCreateInfo allocationInfo{
      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO,
  };

  VkBuffer newBuffer = VK_NULL_HANDLE;
  VmaAllocation newAllocation = nullptr;

  if (vmaCreateBuffer(allocator, &bufferInfo, &allocationInfo, &newBuffer,
                      &newAllocation, nullptr) != VK_SUCCESS) {
    return false;
  }

  // VMA отображает HOST_VISIBLE память, копирует данные CPU и делает flush,
  // если память не HOST_COHERENT. nullptr нужен для пока пустого SSBO.
  if (source != nullptr &&
      vmaCopyMemoryToAllocation(allocator, source, newAllocation, 0,
                                byteCount) != VK_SUCCESS) {
    vmaDestroyBuffer(allocator, newBuffer, newAllocation);
    return false;
  }

  buffer = {newBuffer, newAllocation, byteCount};
  return true;
}

// Вызывается только до отправки кадров или после ожидания fence/queue idle.
void destroyBuffer(Buffer &buffer) {
  if (buffer.handle != VK_NULL_HANDLE) {
    vmaDestroyBuffer(graphics::internal::context.allocator, buffer.handle,
                     buffer.allocation);
    buffer = {};
  }
}

void destroyMeshBuffers() {
  destroyBuffer(indexBuffer);
  destroyBuffer(vertexBuffer);
}

// Отдельный command buffer используется для первоначальной загрузки ресурсов.
// Графическая очередь также умеет копировать; перенос между семействами не нужен.
bool createUploadResources() {
  auto &context = graphics::internal::context;
  const VkCommandPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
               VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = context.graphics_queue_index,
  };
  if (vkCreateCommandPool(context.device, &poolInfo, nullptr,
                          &uploadCommandPool) != VK_SUCCESS)
    return false;
  const VkCommandBufferAllocateInfo bufferInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = uploadCommandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  if (vkAllocateCommandBuffers(context.device, &bufferInfo,
                                &uploadCommandBuffer) != VK_SUCCESS)
    return false;
  const VkFenceCreateInfo fenceInfo{
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
  };
  return vkCreateFence(context.device, &fenceInfo, nullptr,
                         &uploadFence) == VK_SUCCESS;
}

bool beginUpload() {
  if (vkResetCommandBuffer(uploadCommandBuffer, 0) != VK_SUCCESS)
    return false;
  const VkCommandBufferBeginInfo info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  return vkBeginCommandBuffer(uploadCommandBuffer, &info) == VK_SUCCESS;
}

bool finishUpload() {
  auto &context = graphics::internal::context;
  if (vkEndCommandBuffer(uploadCommandBuffer) != VK_SUCCESS ||
      vkResetFences(context.device, 1, &uploadFence) != VK_SUCCESS)
    return false;
  const VkSubmitInfo submitInfo{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &uploadCommandBuffer,
  };
  if (vkQueueSubmit(context.graphics_queue, 1, &submitInfo,
                      uploadFence) != VK_SUCCESS)
    return false;
  // Fence синхронизирует CPU с GPU: staging нельзя удалять до конца копирования.
  return vkWaitForFences(context.device, 1, &uploadFence, VK_TRUE,
                           UINT64_MAX) == VK_SUCCESS;
}

// Постоянная геометрия размещается в DEVICE_LOCAL памяти.
// CPU заполняет только staging; конечный буфер получает данные командой GPU.
bool createDeviceBuffer(const void *source, VkDeviceSize byteCount,
                        VkBufferUsageFlags usage, Buffer &buffer) {
  auto &context = graphics::internal::context;
  Buffer staging;
  if (!createHostBuffer(source, byteCount, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         staging))
    return false;
  const VkBufferCreateInfo bufferInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = byteCount,
      .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  const VmaAllocationCreateInfo allocationInfo{
      .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
      .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
  };
  bool success = vmaCreateBuffer(context.allocator, &bufferInfo, &allocationInfo,
                                  &buffer.handle, &buffer.allocation,
                                  nullptr) == VK_SUCCESS;
  if (success) {
    buffer.capacity = byteCount;
    success = beginUpload();
    if (success) {
      const VkBufferCopy copy{0, 0, byteCount};
      vkCmdCopyBuffer(uploadCommandBuffer, staging.handle, buffer.handle, 1, &copy);
      // Копирование должно завершить запись до чтения вершин/индексов конвейером.
      const VkBufferMemoryBarrier barrier{
          .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
          .dstAccessMask = usage == VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                               ? VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT
                               : VK_ACCESS_INDEX_READ_BIT,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .buffer = buffer.handle,
          .offset = 0,
          .size = byteCount,
      };
      vkCmdPipelineBarrier(uploadCommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, nullptr,
                            1, &barrier, 0, nullptr);
      success = finishUpload();
    }
  }
  destroyBuffer(staging);
  return success;
}

bool createTexture(const unsigned char *pixels, std::uint32_t width,
                   std::uint32_t height, Texture &texture) {
  auto &context = graphics::internal::context;
  // Копируем распакованные RGBA8. 
  const VkDeviceSize byteCount = VkDeviceSize{width} * height * 4;
  Buffer staging;
  if (!createHostBuffer(pixels, byteCount, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         staging))
    return false;
  const VkImageCreateInfo imageInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_SRGB,
      .extent = {width, height, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };
  const VmaAllocationCreateInfo allocationInfo{
      .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
      .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
  };
  bool success = vmaCreateImage(context.allocator, &imageInfo, &allocationInfo,
                                 &texture.image, &texture.allocation,
                                 nullptr) == VK_SUCCESS;
  if (success) {
    success = beginUpload();
    if (success) {
      // Сначала готовим изображение к копированию, затем переводим в режим чтения.
      // Layout и маски доступа описывают разные стороны одного перехода.
      VkImageMemoryBarrier barrier{
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
          .srcAccessMask = 0,
          .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
          .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = texture.image,
          .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
      vkCmdPipelineBarrier(uploadCommandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                            0, nullptr, 1, &barrier);
      const VkBufferImageCopy copy{
          .bufferOffset = 0,
          .bufferRowLength = 0, // Плотная упаковка: ширина строки берётся из extent.
          .bufferImageHeight = 0,
          .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
          .imageOffset = {0, 0, 0},
          .imageExtent = {width, height, 1},
      };
      vkCmdCopyBufferToImage(uploadCommandBuffer, staging.handle, texture.image,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      vkCmdPipelineBarrier(uploadCommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                            0, nullptr, 1, &barrier);
      success = finishUpload();
    }
  }
  destroyBuffer(staging);
  if (!success)
    return false;

  // VkImageView открывает шейдеру цветной слой и единственный mip-уровень.
  const VkImageViewCreateInfo viewInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = texture.image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_SRGB,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
  };
  return vkCreateImageView(context.device, &viewInfo, nullptr,
                             &texture.view) == VK_SUCCESS;
}

bool createTextureResources() {
  auto &context = graphics::internal::context;
  VkFormatProperties formatProperties{};
  vkGetPhysicalDeviceFormatProperties(context.physical_device,
                                      VK_FORMAT_R8G8B8A8_SRGB, &formatProperties);
  const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
      VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
  if ((formatProperties.optimalTilingFeatures & required) != required) {
    std::cerr << "GPU does not support linear sampling of RGBA8 sRGB textures\n";
    return false;
  }
  const VkSamplerCreateInfo samplerInfo{
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .minLod = 0.0f,
      .maxLod = 0.0f,
  };
  if (vkCreateSampler(context.device, &samplerInfo, nullptr,
                       &textureSampler) != VK_SUCCESS)
    return false;

  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(context.physical_device, &properties);
  for (std::size_t i = 0; i < textures.size(); ++i) {
    int width = 0, height = 0;
    unsigned char *pixels = nullptr;
    // Размер проверяем до декодирования, чтобы не загружать заведомо слишком
    // большое для данного устройства изображение.
    for (const char *extension : {".png", ".jpg", ".jpeg"}) {
      const std::string path = std::string(kTexturePaths[i]) + extension;
      if (stbi_info(path.c_str(), &width, &height, nullptr) &&
          width > 0 && height > 0 &&
          static_cast<std::uint32_t>(width) <= properties.limits.maxImageDimension2D &&
          static_cast<std::uint32_t>(height) <= properties.limits.maxImageDimension2D)
        pixels = stbi_load(path.c_str(), &width, &height, nullptr, STBI_rgb_alpha);
      if (pixels != nullptr) {
        textures[i].path = path;
        break;
      }
    }
    // Пока пользователь не добавил изображение, валидный белый тексель заменяет файл.
    const unsigned char whitePixel[4] = {255, 255, 255, 255};
    if (pixels == nullptr)
      std::cerr << "Cannot load " << kTexturePaths[i]
                << ".png/.jpg/.jpeg; using a white fallback "
                   "(missing, invalid or too large image)\n";
    const bool success = createTexture(pixels ? pixels : whitePixel,
                                        pixels ? width : 1, pixels ? height : 1,
                                        textures[i]);
    stbi_image_free(pixels);
    if (!success)
      return false;
  }
  return true;
}

void destroyTextureResources() {
  auto &context = graphics::internal::context;
  for (Texture &texture : textures) {
    if (texture.view != VK_NULL_HANDLE)
      vkDestroyImageView(context.device, texture.view, nullptr);
    if (texture.image != VK_NULL_HANDLE)
      vmaDestroyImage(context.allocator, texture.image, texture.allocation);
    texture = {};
  }
  if (textureSampler != VK_NULL_HANDLE)
    vkDestroySampler(context.device, textureSampler, nullptr);
  textureSampler = VK_NULL_HANDLE;
  if (uploadFence != VK_NULL_HANDLE)
    vkDestroyFence(context.device, uploadFence, nullptr);
  uploadFence = VK_NULL_HANDLE;
  if (uploadCommandPool != VK_NULL_HANDLE)
    vkDestroyCommandPool(context.device, uploadCommandPool, nullptr);
  uploadCommandPool = VK_NULL_HANDLE;
  uploadCommandBuffer = VK_NULL_HANDLE;
}

void destroyUniformResources() {
  auto &context = graphics::internal::context;

  if (descriptorPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
    descriptorPool = VK_NULL_HANDLE;
    sceneDescriptorSet = VK_NULL_HANDLE;
    for (Texture &texture : textures)
      texture.descriptorSet = VK_NULL_HANDLE;
  }

  destroyBuffer(spotLightBuffer);
  destroyBuffer(pointLightBuffer);
  destroyBuffer(modelBuffer);
  destroyBuffer(sceneBuffer);
}

// Дескриптор хранит ссылку на VkBuffer и диапазон, а не сами матрицы/источники.
void writeBufferDescriptor(VkDescriptorSet set, std::uint32_t binding,
                           VkDescriptorType type, const Buffer &buffer) {
  const VkDescriptorBufferInfo info{
      .buffer = buffer.handle,
      .offset = 0,
      // Диапазон динамического дескриптора охватывает одну структуру модели.
      .range = type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                   ? sizeof(ModelUniforms) : buffer.capacity,
  };
  const VkWriteDescriptorSet write{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = set,
      .dstBinding = binding,
      .descriptorCount = 1,
      .descriptorType = type,
      .pBufferInfo = &info,
  };
  vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0,
                         nullptr);
}

bool createUniformResources() {
  auto &context = graphics::internal::context;
  const SceneUniforms scene{};
  const PointLight emptyPoint{};
  const SpotLight emptySpot{};
  // Даже при нуле источников descriptor должен ссылаться на валидный буфер.
  // Пустой элемент не читается: количество активных источников равно нулю.
  if (!createHostBuffer(&scene, sizeof(scene),
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sceneBuffer) ||
      !createHostBuffer(nullptr, modelUniformStride * kMaxModels,
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, modelBuffer) ||
      !createHostBuffer(&emptyPoint, sizeof(emptyPoint),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          pointLightBuffer) ||
      !createHostBuffer(&emptySpot, sizeof(emptySpot),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          spotLightBuffer)) {
    return false;
  }

  constexpr std::uint32_t materialCount = kTexturePaths.size();
  const VkDescriptorPoolSize poolSizes[4] = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, materialCount},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, materialCount},
  };

  const VkDescriptorPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1 + materialCount,
      .poolSizeCount = 4,
      .pPoolSizes = poolSizes,
  };

  if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr,
                             &descriptorPool) != VK_SUCCESS) {
    return false;
  }

  // Один set сцены и по одному set на материал. Все материалы ссылаются
  // на один динамический UBO, но содержат разные пары image view + sampler.
  std::array<VkDescriptorSetLayout, 1 + materialCount> layouts{};
  layouts.fill(modelSetLayout);
  layouts[0] = sceneSetLayout;
  const VkDescriptorSetAllocateInfo setInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptorPool,
      .descriptorSetCount = static_cast<std::uint32_t>(layouts.size()),
      .pSetLayouts = layouts.data(),
  };

  std::array<VkDescriptorSet, 1 + materialCount> sets{};
  if (vkAllocateDescriptorSets(context.device, &setInfo, sets.data()) != VK_SUCCESS) {
    return false;
  }

  sceneDescriptorSet = sets[0];
  writeBufferDescriptor(sceneDescriptorSet, 0,
                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sceneBuffer);
  writeBufferDescriptor(sceneDescriptorSet, 1,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pointLightBuffer);
  writeBufferDescriptor(sceneDescriptorSet, 2,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, spotLightBuffer);
  for (std::size_t i = 0; i < textures.size(); ++i) {
    textures[i].descriptorSet = sets[i + 1];
    writeBufferDescriptor(textures[i].descriptorSet, 0,
                          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, modelBuffer);
    const VkDescriptorImageInfo imageInfo{
        .sampler = textureSampler,
        .imageView = textures[i].view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = textures[i].descriptorSet,
        .dstBinding = 1,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &imageInfo,
    };
    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
  }
  return true;
}

// Длина массива в GLSL не фиксируется. Когда CPU-массив вырос, создаём больший
// SSBO и обновляем его descriptor. render вызывается ПОСЛЕ prepare/fence.
bool uploadLightBuffer(Buffer &buffer, const void *data, std::size_t count,
                       std::size_t elementSize, std::uint32_t binding) {
  if (count > maxStorageBufferBytes / elementSize) {
    std::cerr << "Light array exceeds maxStorageBufferRange\n";
    return false;
  }
  const VkDeviceSize required = std::max<std::size_t>(count, 1) * elementSize;
  if (required > buffer.capacity) {
    // Удвоение ёмкости сокращает число перевыделений при последовательном Add.
    const VkDeviceSize capacity = std::min(
        maxStorageBufferBytes, std::max(required, buffer.capacity * 2));
    Buffer larger;
    if (!createHostBuffer(nullptr, capacity,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, larger))
      return false;
    destroyBuffer(buffer);
    buffer = larger;
    writeBufferDescriptor(sceneDescriptorSet, binding,
                          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffer);
  }
  return count == 0 ||
         vmaCopyMemoryToAllocation(graphics::internal::context.allocator, data,
                                   buffer.allocation, 0,
                                   count * elementSize) == VK_SUCCESS;
}

Mat4 makePerspective(float aspect, float nearPlane, float farPlane) {
  // Перед камерой Z положительная; проекция переводит глубину near..far в 0..1.
  const float scale =
      1.0f / std::tan(cameraFovDegrees * kRadiansPerDegree * 0.5f);

  Mat4 result{};
  result.m[0] = scale / aspect;
  result.m[5] = scale;
  result.m[10] = farPlane / (farPlane - nearPlane);
  result.m[11] = 1.0f;
  result.m[14] = -nearPlane * farPlane / (farPlane - nearPlane);
  return result;
}

Mat4 makeOrthographic(float aspect, float halfHeight, float nearPlane,
                      float farPlane) {
  // В ортографии w=1: размер по X/Y не зависит от расстояния до камеры.
  Mat4 result = kIdentity;
  result.m[0] = 1.0f / (aspect * halfHeight);
  result.m[5] = 1.0f / halfHeight;
  result.m[10] = 1.0f / (farPlane - nearPlane);
  result.m[14] = -nearPlane / (farPlane - nearPlane);
  return result;
}

Mat4 multiply(const Mat4 &left, const Mat4 &right) {
  // Строка left умножается на столбец right. left * right действует справа
  // налево.
  Mat4 result{};
  for (int column = 0; column < 4; ++column) {
    for (int row = 0; row < 4; ++row) {
      for (int k = 0; k < 4; ++k) {
        result.m[column * 4 + row] +=
            left.m[k * 4 + row] * right.m[column * 4 + k];
      }
    }
  }
  return result;
}

Mat4 makeTranslation(const float position[3]) {
  // Четвёртый столбец прибавляет смещение точке с w=1.
  Mat4 result = kIdentity;
  result.m[12] = position[0];
  result.m[13] = position[1];
  result.m[14] = position[2];
  return result;
}

Mat4 makeScale(const float scale[3]) {
  // Диагональ независимо растягивает X, Y и Z относительно центра модели.
  Mat4 result = kIdentity;
  result.m[0] = scale[0];
  result.m[5] = scale[1];
  result.m[10] = scale[2];
  return result;
}

Mat4 makeRotationX(float angle) {
  // Ось X сохраняется, sin/cos смешивают координаты Y и Z. angle — в радианах.
  const float cosine = std::cos(angle);
  const float sine = std::sin(angle);
  Mat4 result = kIdentity;
  result.m[5] = cosine;
  result.m[6] = sine;
  result.m[9] = -sine;
  result.m[10] = cosine;
  return result;
}

Mat4 makeRotationY(float angle) {
  // Ось Y сохраняется, вращается плоскость XZ.
  const float cosine = std::cos(angle);
  const float sine = std::sin(angle);
  Mat4 result = kIdentity;
  result.m[0] = cosine;
  result.m[2] = -sine;
  result.m[8] = sine;
  result.m[10] = cosine;
  return result;
}

Mat4 makeRotationZ(float angle) {
  // Ось Z сохраняется: x'=cos*x-sin*y, y'=sin*x+cos*y.
  const float cosine = std::cos(angle);
  const float sine = std::sin(angle);
  Mat4 result = kIdentity;
  result.m[0] = cosine;
  result.m[1] = sine;
  result.m[4] = -sine;
  result.m[5] = cosine;
  return result;
}

Mat4 makeModelMatrix() {
  // Формируем мировую позицию точки.

  // M = T * Ry_anim * Rx_anim * Rz * Ry * Rx * S; к точке применяем справа
  // налево.
  constexpr float radiansPerDegree = std::numbers::pi_v<float> / 180.0f;
  const float angleX = objectRotationDegrees[0] * radiansPerDegree;
  const float angleY = objectRotationDegrees[1] * radiansPerDegree;
  const float angleZ = objectRotationDegrees[2] * radiansPerDegree;

  const Mat4 manualRotation =
      multiply(makeRotationZ(angleZ),
               multiply(makeRotationY(angleY), makeRotationX(angleX)));
  return multiply(
      makeTranslation(objectPosition),
      multiply(makeRotationY(static_cast<float>(animationPhaseY)),
               multiply(makeRotationX(static_cast<float>(animationPhaseX)),
                        multiply(manualRotation, makeScale(objectScale)))));
}

Mat4 makeViewMatrix() {
  // Переводим мировые координаты в СК камеры.

  // View — обратная матрица преобразования камеры: V = C^-1.
  // Если C=T*Rz*Ry*Rx, то C^-1=Rx^-1*Ry^-1*Rz^-1*T^-1.
  const float inversePosition[3] = {-cameraPosition[0], -cameraPosition[1],
                                    -cameraPosition[2]};
  const Mat4 inverseRotation = multiply(
      makeRotationX(-cameraRotationDegrees[0] * kRadiansPerDegree),
      multiply(makeRotationY(-cameraRotationDegrees[1] * kRadiansPerDegree),
               makeRotationZ(-cameraRotationDegrees[2] * kRadiansPerDegree)));
  // Согласуем оси: экранная Y направлена вниз, видимая глубина — вдоль +Z.
  Mat4 cameraAxes = kIdentity;
  cameraAxes.m[5] = -1.0f;
  cameraAxes.m[10] = -1.0f;
  return multiply(cameraAxes,
                  multiply(inverseRotation, makeTranslation(inversePosition)));
}

Mat4 makeNormalMatrix(const Mat4 &model) {
  // Нормаль должна остаться перпендикулярной грани даже при разном масштабе
  // осей. Для M=T*R*S матрица нормалей (M^-1)^T имеет линейную часть R*S^-1.
  // Столбец model уже равен R_column*s: деление на s^2 даёт R_column/s.
  Mat4 result = kIdentity;
  for (int column = 0; column < 3; ++column)
    for (int row = 0; row < 3; ++row)
      result.m[column * 4 + row] = model.m[column * 4 + row] /
                                   (objectScale[column] * objectScale[column]);
  return result;
}

void drawCameraControls() {
  if (!ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
    return;
  ImGui::PushID("camera");
  ImGui::DragFloat3("Position", cameraPosition, 0.02f);
  ImGui::DragFloat3("Pitch / yaw / roll", cameraRotationDegrees, 0.5f);
  ImGui::SliderFloat("FOV (degrees)", &cameraFovDegrees, 15.0f, 120.0f, "%.1f",
                     ImGuiSliderFlags_AlwaysClamp);
  if (ImGui::RadioButton("Perspective",
                         projectionMode == ProjectionMode::Perspective))
    projectionMode = ProjectionMode::Perspective;
  ImGui::SameLine();
  if (ImGui::RadioButton("Orthographic",
                         projectionMode == ProjectionMode::Orthographic))
    projectionMode = ProjectionMode::Orthographic;
  if (projectionMode == ProjectionMode::Orthographic)
    ImGui::SliderFloat("Ortho half-height", &orthographicHalfHeight, 0.1f,
                       10.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
  if (ImGui::Button("Reset camera")) {
    cameraPosition[0] = cameraPosition[1] = 0.0f;
    cameraPosition[2] = 3.5f;
    cameraRotationDegrees[0] = cameraRotationDegrees[1] =
        cameraRotationDegrees[2] = 0.0f;
    cameraFovDegrees = 60.0f;
    orthographicHalfHeight = 3.5f * std::tan(30.0f * kRadiansPerDegree);
  }
  ImGui::PopID();
}

void drawModelControls() {
  if (!ImGui::CollapsingHeader("Model and material",
                               ImGuiTreeNodeFlags_DefaultOpen))
    return;
  ImGui::PushID("model");
  // Масштаб, поворот и перенос задают преобразование модели в мировую систему.
  ImGui::DragFloat3("Position", objectPosition, 0.01f);
  ImGui::DragFloat3("Rotation (degrees)", objectRotationDegrees, 0.5f);
  ImGui::SliderFloat3("Scale", objectScale, 0.1f, 3.0f, "%.2f",
                      ImGuiSliderFlags_AlwaysClamp);
  if (ImGui::Button("Reset transform")) {
    for (int component = 0; component < 3; ++component) {
      objectPosition[component] = 0.0f;
      objectRotationDegrees[component] = 0.0f;
      objectScale[component] = 1.0f;
    }
  }
  // Альбедо берём из выбранной текстуры; белый множитель сохраняет её цвета.
  ImGui::Combo("Texture", &selectedTexture, kTexturePaths.data(),
                static_cast<int>(kTexturePaths.size()));
  if (textures[selectedTexture].path.empty())
    ImGui::TextWrapped("PNG/JPG unavailable: white fallback. Add the file and restart.");
  else
    ImGui::Text("File: %s", textures[selectedTexture].path.c_str());
  ImGui::ColorEdit3("Albedo tint", objectColor);
  ImGui::SliderFloat("UV scale", &uvScale, 0.25f, 8.0f, "%.2f",
                     ImGuiSliderFlags_AlwaysClamp);
  // UI меняет параметры эффекта на CPU; render передаёт их через push constants.
  ImGui::Checkbox("Animated sampling", &samplingEnabled);
  ImGui::SliderFloat("Sampling strength", &samplingStrength, 0.0f, 1.0f, "%.2f",
                     ImGuiSliderFlags_AlwaysClamp);
  ImGui::SliderFloat("Sampling speed", &samplingSpeed, 0.0f, 3.0f, "%.2f",
                     ImGuiSliderFlags_AlwaysClamp);
  // Помехи телевизора — отдельный слой с собственными временем и силой.
  ImGui::Checkbox("Old TV interference", &tvNoiseEnabled);
  ImGui::SliderFloat("TV strength", &tvNoiseStrength, 0.0f, 1.0f, "%.2f",
                     ImGuiSliderFlags_AlwaysClamp);
  ImGui::SliderFloat("TV speed", &tvNoiseSpeed, 0.0f, 3.0f, "%.2f",
                     ImGuiSliderFlags_AlwaysClamp);
  // Материал Блинн–Фонга: Ks — цвет блика, alpha — степень блеска.
  ImGui::ColorEdit3("Specular color (Ks)", specularColor);
  ImGui::SliderFloat("Shininess", &shininess, 1.0f, 256.0f, "%.1f",
                     ImGuiSliderFlags_AlwaysClamp);
  if (ImGui::Button(animationPlaying ? "Pause" : "Play"))
    animationPlaying = !animationPlaying;
  ImGui::SameLine();
  if (ImGui::Button("Reset animation")) {
    animationPlaying = false;
    animationPhaseX = animationPhaseY = 0.0;
  }
  ImGui::SliderFloat("Y speed (rad/s)", &animationSpeedY, 0.0f, 3.0f, "%.2f",
                     ImGuiSliderFlags_AlwaysClamp);
  ImGui::SliderFloat("X speed (rad/s)", &animationSpeedX, 0.0f, 3.0f, "%.2f",
                     ImGuiSliderFlags_AlwaysClamp);
  ImGui::PopID();
}

void drawPointLightControls() {
  if (!ImGui::CollapsingHeader("Point lights"))
    return;
  ImGui::PushID("pointLights");
  if (ImGui::Button("Add point light"))
    pointLights.push_back({{2.0f, 2.0f, 2.0f, 1.0f}, {1.0f, 0.8f, 0.6f, 6.0f}});
  ImGui::SameLine();
  ImGui::Text("Count: %zu", pointLights.size());
  // При удалении не увеличиваем i: на это место сдвинулся следующий источник.
  for (std::size_t i = 0; i < pointLights.size();) {
    ImGui::PushID(static_cast<int>(i));
    const bool open = ImGui::TreeNode("Point", "Point light %zu", i + 1);
    ImGui::SameLine();
    const bool remove = ImGui::SmallButton("Remove");
    if (open) {
      PointLight &light = pointLights[i];
      ImGui::DragFloat3("Position", light.position, 0.02f);
      ImGui::ColorEdit3("Color", light.colorIntensity);
      ImGui::DragFloat("Intensity", &light.colorIntensity[3], 0.1f, 0.0f,
                       100.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
      ImGui::TreePop();
    }
    ImGui::PopID();
    if (remove)
      pointLights.erase(pointLights.begin() + static_cast<std::ptrdiff_t>(i));
    else
      ++i;
  }
  ImGui::PopID();
}

void drawSpotLightControls() {
  if (!ImGui::CollapsingHeader("Spot lights"))
    return;
  ImGui::PushID("spotLights");
  if (ImGui::Button("Add spot light")) {
    // Начальное направление (-2,-2,-2) смотрит из (2,2,2) в центр фигуры.
    spotLights.push_back({{2.0f, 2.0f, 2.0f, 1.0f},
                          {-2.0f, -2.0f, -2.0f, 0.0f},
                          {0.6f, 0.8f, 1.0f, 10.0f},
                          {std::cos(15.0f * kRadiansPerDegree),
                           std::cos(25.0f * kRadiansPerDegree), 0.0f, 0.0f}});
  }
  ImGui::SameLine();
  ImGui::Text("Count: %zu", spotLights.size());
  for (std::size_t i = 0; i < spotLights.size();) {
    ImGui::PushID(static_cast<int>(i));
    const bool open = ImGui::TreeNode("Spot", "Spot light %zu", i + 1);
    ImGui::SameLine();
    const bool remove = ImGui::SmallButton("Remove");
    if (open) {
      SpotLight &light = spotLights[i];
      ImGui::DragFloat3("Position", light.position, 0.02f);
      ImGui::DragFloat3("Direction", light.direction, 0.02f);
      ImGui::ColorEdit3("Color", light.colorIntensity);
      ImGui::DragFloat("Intensity", &light.colorIntensity[3], 0.1f, 0.0f,
                       100.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
      // Пользователь задаёт ПОЛУуглы конуса в градусах; shader сравнивает cos.
      float inner = std::acos(std::clamp(light.angleCosines[0], -1.0f, 1.0f)) /
                    kRadiansPerDegree;
      float outer = std::acos(std::clamp(light.angleCosines[1], -1.0f, 1.0f)) /
                    kRadiansPerDegree;
      if (ImGui::SliderFloat("Inner half-angle", &inner, 0.0f, outer,
                             "%.1f deg", ImGuiSliderFlags_AlwaysClamp))
        light.angleCosines[0] = std::cos(inner * kRadiansPerDegree);
      if (ImGui::SliderFloat("Outer half-angle", &outer, std::max(inner, 0.1f),
                             89.0f, "%.1f deg", ImGuiSliderFlags_AlwaysClamp))
        light.angleCosines[1] = std::cos(outer * kRadiansPerDegree);
      ImGui::TreePop();
    }
    ImGui::PopID();
    if (remove)
      spotLights.erase(spotLights.begin() + static_cast<std::ptrdiff_t>(i));
    else
      ++i;
  }
  ImGui::PopID();
}

void drawLightingControls() {
  if (ImGui::CollapsingHeader("Ambient and directional light",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::ColorEdit3("Ambient color", ambientColor);
    ImGui::SliderFloat("Ambient intensity", &ambientIntensity, 0.0f, 1.0f,
                       "%.2f", ImGuiSliderFlags_AlwaysClamp);
    ImGui::Checkbox("Directional enabled", &directionalLightEnabled);
    ImGui::DragFloat3("Light ray direction", lightDirection, 0.02f);
    ImGui::ColorEdit3("Light color", lightColor);
    ImGui::SliderFloat("Light intensity", &lightIntensity, 0.0f, 5.0f, "%.2f",
                       ImGuiSliderFlags_AlwaysClamp);
  }
  drawPointLightControls();
  drawSpotLightControls();
}
} // namespace

namespace application {
bool initialize() {
  auto &context = graphics::internal::context;
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(context.physical_device, &properties);
  maxStorageBufferBytes = properties.limits.maxStorageBufferRange;
  // std140 задаёт раскладку полей, лимит GPU — выравнивание начала слота модели.
  const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
  modelUniformStride = (sizeof(ModelUniforms) + alignment - 1) / alignment * alignment;

  if (!loadShaderModule("shaders/tetra.vert.spv", vertexShader) ||
      !loadShaderModule("shaders/tetra.frag.spv", fragmentShader)) {
    destroyPipelineSetup();
    return false;
  }

  // set 0: общий UBO + два runtime-массива источников в SSBO.
  const VkDescriptorSetLayoutBinding sceneBindings[3] = {
      {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT,
       nullptr},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT,
       nullptr},
  };

  const VkDescriptorSetLayoutCreateInfo setLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3,
      .pBindings = sceneBindings,
  };

  if (vkCreateDescriptorSetLayout(context.device, &setLayoutInfo, nullptr,
                                  &sceneSetLayout) != VK_SUCCESS) {
    std::cerr << "Cannot create descriptor set layout\n";
    destroyPipelineSetup();
    return false;
  }

  // set 1 связывает текстуру материала и выбранный слот общего буфера моделей.
  const VkDescriptorSetLayoutBinding modelBindings[2] = {
      {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
       VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
  };
  const VkDescriptorSetLayoutCreateInfo modelLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = modelBindings,
  };
  if (vkCreateDescriptorSetLayout(context.device, &modelLayoutInfo, nullptr,
                                  &modelSetLayout) != VK_SUCCESS) {
    destroyPipelineSetup();
    return false;
  }
  const VkDescriptorSetLayout layouts[2] = {sceneSetLayout, modelSetLayout};
  const VkPushConstantRange samplingRange{
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      .offset = 0,
      .size = sizeof(SamplingConstants),
  };
  const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 2,
      .pSetLayouts = layouts,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &samplingRange,
  };

  if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr,
                             &pipelineLayout) != VK_SUCCESS) {
    std::cerr << "Cannot create pipeline layout\n";
    destroyPipelineSetup();
    return false;
  }

  if (!createGraphicsPipeline()) {
    destroyPipelineSetup();
    return false;
  }

  generateMeshNormals();
  if (!createUploadResources() ||
      !createDeviceBuffer(meshVertices.data(), sizeof(meshVertices),
                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer) ||
      !createDeviceBuffer(meshIndices.data(), sizeof(meshIndices),
                          VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer) ||
      !createTextureResources() || !createUniformResources()) {
    std::cerr << "Cannot upload geometry, textures or create descriptors\n";
    shutdown();
    return false;
  }

  return true;
}

void shutdown() {
  // GPU должен закончить читать наши буферы до их уничтожения.
  auto &context = graphics::internal::context;
  vkQueueWaitIdle(context.graphics_queue);
  destroyUniformResources();
  destroyTextureResources();
  destroyMeshBuffers();
  destroyPipelineSetup();
}

void update(double time) {
  // Угол += скорость * delta time: одинаковая скорость при разной частоте
  // кадров.
  if (previousUpdateTime >= 0.0) {
    const double elapsed = time - previousUpdateTime;
    if (elapsed > 0.0) {
      constexpr double fullTurn = 2.0 * std::numbers::pi_v<double>;
      if (animationPlaying) {
        animationPhaseY =
            std::fmod(animationPhaseY + animationSpeedY * elapsed, fullTurn);
        animationPhaseX =
            std::fmod(animationPhaseX + animationSpeedX * elapsed, fullTurn);
      }
      // Эффект текстуры имеет своё время и не зависит от паузы вращения модели.
      if (samplingEnabled)
        samplingTime = std::fmod(samplingTime + samplingSpeed * elapsed, fullTurn);
      // Ограничиваем время для точности float; шум и скачки строк специально
      // дискретны. Отдельные часы позволяют независимо заморозить каждый эффект.
      if (tvNoiseEnabled)
        tvNoiseTime = std::fmod(tvNoiseTime + tvNoiseSpeed * elapsed, 5120.0);
    }
  }
  // Обновляем даже во время паузы, чтобы Play не давал скачок накопленного
  // времени.
  previousUpdateTime = time;

  // UI меняет только CPU-параметры. GPU-буферы обновляются позже, после fence.
  ImGui::SetNextWindowSize(ImVec2(380.0f, 660.0f), ImGuiCond_FirstUseEver);
  ImGui::Begin("Lab 3 - Texture mapping");
  ImGui::Text("12 geometric vertices, 60 GPU vertices, 20 triangles");
  drawCameraControls();
  drawModelControls();
  drawLightingControls();
  ImGui::End();
}

void render(const graphics::internal::FrameData &fd) {
  auto &context = graphics::internal::context;
  const VkExtent2D extent = context.swapchain_extent;
  if (extent.width == 0 || extent.height == 0) {
    std::cerr << "Cannot render a zero-sized framebuffer\n";
    std::abort();
  }

  // Сначала камера переводит точки в свой базис; P*V общая для всех моделей.
  const Mat4 view = makeViewMatrix();
  const float aspect =
      static_cast<float>(extent.width) / static_cast<float>(extent.height);
  const Mat4 projection = projectionMode == ProjectionMode::Perspective
                              ? makePerspective(aspect, cameraNear, cameraFar)
                              : makeOrthographic(aspect, orthographicHalfHeight,
                                                 cameraNear, cameraFar);
  const SceneUniforms scene{
      .viewProjection = multiply(projection, view),
      .cameraPosition = {cameraPosition[0], cameraPosition[1],
                         cameraPosition[2],
                         projectionMode == ProjectionMode::Perspective ? 1.0f
                                                                       : 0.0f},
      .viewDirection = {-view.m[2], -view.m[6], -view.m[10], 0.0f},
      .ambientColorIntensity = {ambientColor[0], ambientColor[1],
                                ambientColor[2], ambientIntensity},
      .lightDirection = {lightDirection[0], lightDirection[1],
                         lightDirection[2], 0.0f},
      .lightColorIntensity = {lightColor[0], lightColor[1], lightColor[2],
                              directionalLightEnabled ? lightIntensity : 0.0f},
      .lightCounts = {static_cast<std::int32_t>(pointLights.size()),
                      static_cast<std::int32_t>(spotLights.size()), 0, 0},
  };
  const Mat4 modelMatrix = makeModelMatrix();
  const ModelUniforms model{
      .model = modelMatrix,
      .normalMatrix = makeNormalMatrix(modelMatrix),
      .albedo = {objectColor[0], objectColor[1], objectColor[2], 1.0f},
      .specularShininess = {specularColor[0], specularColor[1],
                            specularColor[2], shininess},
  };
  // CPU записывает модель в слот, который будет выбран смещением при рисовании.
  const std::uint32_t modelOffset =
      static_cast<std::uint32_t>(kModelIndex * modelUniformStride);

  // prepare() дождался предыдущего GPU-кадра: теперь можно менять UBO, SSBO
  // и descriptors, включая перевыделение выросших массивов источников.
  if (!uploadLightBuffer(pointLightBuffer, pointLights.data(),
                         pointLights.size(), sizeof(PointLight), 1) ||
      !uploadLightBuffer(spotLightBuffer, spotLights.data(), spotLights.size(),
                         sizeof(SpotLight), 2) ||
      vmaCopyMemoryToAllocation(context.allocator, &scene,
                                sceneBuffer.allocation, 0,
                                sizeof(scene)) != VK_SUCCESS ||
      vmaCopyMemoryToAllocation(context.allocator, &model,
                                modelBuffer.allocation, modelOffset,
                                sizeof(model)) != VK_SUCCESS) {
    std::cerr << "Cannot upload scene, model or lights\n";
    std::abort();
  }

  const VkCommandBuffer commandBuffer = fd.command_buffer;
  if (vkResetCommandBuffer(commandBuffer, 0) != VK_SUCCESS) {
    std::cerr << "Cannot reset command buffer\n";
    std::abort();
  }

  const VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
  };
  if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
    std::cerr << "Cannot begin command buffer\n";
    std::abort();
  }

  // Проход рисования очищает цвет и устанавливает depth=1 — дальнюю глубину.
  VkClearValue clearValues[2]{};
  clearValues[0].color.float32[0] = 0.04f;
  clearValues[0].color.float32[1] = 0.05f;
  clearValues[0].color.float32[2] = 0.08f;
  clearValues[0].color.float32[3] = 1.0f;
  clearValues[1].depthStencil.depth = 1.0f;
  clearValues[1].depthStencil.stencil = 0;

  const VkRenderPassBeginInfo renderPassInfo{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = context.render_pass,
      .framebuffer = fd.framebuffer,
      .renderArea =
          {
              .offset = {0, 0},
              .extent = extent,
          },
      .clearValueCount = 2,
      .pClearValues = clearValues,
  };
  vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
                       VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    graphicsPipeline);

  const VkViewport viewport{
      .x = 0.0f,
      .y = 0.0f,
      .width = static_cast<float>(extent.width),
      .height = static_cast<float>(extent.height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };
  const VkRect2D scissor{
      .offset = {0, 0},
      .extent = extent,
  };
  vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
  vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

  const VkDeviceSize vertexOffset = 0;
  vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer.handle,
                         &vertexOffset);
  vkCmdBindIndexBuffer(commandBuffer, indexBuffer.handle, 0,
                       VK_INDEX_TYPE_UINT16);
  // Смещение модели задаётся при привязке готового набора дескрипторов.
  // Другая текстура выбирает готовый набор материала с тем же динамическим UBO.
  const VkDescriptorSet sets[2] = {
      sceneDescriptorSet, textures[selectedTexture].descriptorSet};
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelineLayout, 0, 2, sets, 1, &modelOffset);
  const SamplingConstants sampling{
      .enabled = samplingEnabled ? 1 : 0,
      .time = static_cast<float>(samplingTime),
      .strength = samplingStrength,
      .uvScale = uvScale,
      .tvEnabled = tvNoiseEnabled ? 1 : 0,
      .tvTime = static_cast<float>(tvNoiseTime),
      .tvStrength = tvNoiseStrength,
  };
  vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                      0, sizeof(sampling), &sampling);
  vkCmdDrawIndexed(commandBuffer, static_cast<std::uint32_t>(kIndices.size()),
                   1, 0, 0, 0);

  vkCmdEndRenderPass(commandBuffer);
  if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
    std::cerr << "Cannot end command buffer\n";
    std::abort();
  }
}
} // namespace application
