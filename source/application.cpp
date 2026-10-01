#include "application.hpp"

#include <imgui.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numbers>
#include <vector>

namespace
{
    struct Vertex
    {
        float x;
        float y;
        float z;
    };

    struct alignas(16) Mat4
    {
        float m[16];
    };

    struct alignas(16) GlobalUniforms
    {
        Mat4 model;
        Mat4 view;
        Mat4 projection;
        float baseColor[4];
    };

    static_assert(sizeof(Mat4) == 64);
    static_assert(sizeof(GlobalUniforms) == 208);
    static_assert(offsetof(GlobalUniforms, view) == 64);
    static_assert(offsetof(GlobalUniforms, projection) == 128);
    static_assert(offsetof(GlobalUniforms, baseColor) == 192);

    constexpr Mat4 kIdentity{
        {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        }
    };

    constexpr float kCameraDistance = 3.5f;
    constexpr float kHalfVerticalFov = std::numbers::pi_v<float> / 6.0f;

    enum class ProjectionMode
    {
        Perspective,
        Orthographic,
    };

    ProjectionMode projectionMode = ProjectionMode::Perspective;
    float objectPosition[3] = {0.0f, 0.0f, 0.0f};
    float objectRotationDegrees[3] = {20.0f, -25.0f, 0.0f};
    float objectScale[3] = {1.0f, 1.0f, 1.0f};
    float objectColor[3] = {0.9f, 0.6f, 0.2f};

    bool animationPlaying = false;
    float animationSpeedY = 0.8f;
    float animationSpeedX = 1.3f;
    double animationPhaseY = 0.0;
    double animationPhaseX = 0.0;
    double previousUpdateTime = -1.0;

    constexpr float t = 1.0f / 3.0f;

    constexpr std::array<Vertex, 12> kVertices{
        {
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
        }
    };

    constexpr std::array<std::uint16_t, 60> kIndices{
        {
            // Четыре треугольные грани
            0, 1, 2,
            5, 4, 3,
            6, 7, 8,
            11, 10, 9,

            // Шестиугольник a1-a2-a3
            5, 10, 11,
            5, 11, 8,
            5, 8, 7,
            5, 7, 4,

            // Шестиугольник a0-a2-a3
            1, 6, 8,
            1, 8, 11,
            1, 11, 9,
            1, 9, 2,

            // Шестиугольник a0-a1-a3
            2, 9, 10,
            2, 10, 5,
            2, 5, 3,
            2, 3, 0,

            // Шестиугольник a0-a1-a2
            0, 3, 4,
            0, 4, 7,
            0, 7, 6,
            0, 6, 1,
        }
    };

    static_assert(kIndices.size() % 3 == 0);

    VkShaderModule vertexShader = VK_NULL_HANDLE;
    VkShaderModule fragmentShader = VK_NULL_HANDLE;
    VkDescriptorSetLayout globalSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VmaAllocation vertexAllocation = nullptr;

    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VmaAllocation indexAllocation = nullptr;

    VkBuffer uniformBuffer = VK_NULL_HANDLE;
    VmaAllocation uniformAllocation = nullptr;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet globalDescriptorSet = VK_NULL_HANDLE;

    bool loadShaderModule(const char* path, VkShaderModule& result)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
        {
            std::cerr << "Cannot open shader: " << path << '\n';
            return false;
        }

        const std::streampos endPosition = file.tellg();
        if (endPosition <= 0)
        {
            std::cerr << "Shader is empty: " << path << '\n';
            return false;
        }

        const std::size_t byteCount = static_cast<std::size_t>(endPosition);
        if (byteCount % sizeof(std::uint32_t) != 0)
        {
            std::cerr << "Invalid SPIR-V size: " << path << '\n';
            return false;
        }

        std::vector<std::uint32_t> code(byteCount / sizeof(std::uint32_t));
        file.seekg(0);

        if (!file.read(
            reinterpret_cast<char*>(code.data()),
            static_cast<std::streamsize>(byteCount)))
        {
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

        if (status != VK_SUCCESS)
        {
            std::cerr << "Cannot create shader module: " << path << '\n';
            return false;
        }

        return true;
    }

    void destroyPipelineSetup()
    {
        VkDevice device = graphics::internal::context.device;

        if (graphicsPipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device, graphicsPipeline, nullptr);
            graphicsPipeline = VK_NULL_HANDLE;
        }

        if (pipelineLayout != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            pipelineLayout = VK_NULL_HANDLE;
        }
        if (globalSetLayout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device, globalSetLayout, nullptr);
            globalSetLayout = VK_NULL_HANDLE;
        }
        if (fragmentShader != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(device, fragmentShader, nullptr);
            fragmentShader = VK_NULL_HANDLE;
        }
        if (vertexShader != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(device, vertexShader, nullptr);
            vertexShader = VK_NULL_HANDLE;
        }
    }

    bool createGraphicsPipeline()
    {
        auto& context = graphics::internal::context;

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

        const VkVertexInputAttributeDescription positionAttribute{
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = static_cast<std::uint32_t>(offsetof(Vertex, x)),
        };

        const VkPipelineVertexInputStateCreateInfo vertexInput{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &vertexBinding,
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions = &positionAttribute,
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
            .colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT,
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

        if (vkCreateGraphicsPipelines(
            context.device,
            VK_NULL_HANDLE,
            1,
            &pipelineInfo,
            nullptr,
            &graphicsPipeline) != VK_SUCCESS)
        {
            std::cerr << "Cannot create graphics pipeline\n";
            return false;
        }

        return true;
    }

    bool createStaticBuffer(
        const void* source,
        VkDeviceSize byteCount,
        VkBufferUsageFlags usage,
        VkBuffer& buffer,
        VmaAllocation& allocation
    )
    {
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

        if (vmaCreateBuffer(
            allocator, &bufferInfo, &allocationInfo,
            &newBuffer, &newAllocation, nullptr) != VK_SUCCESS)
        {
            return false;
        }

        void* mapped = nullptr;
        if (vmaMapMemory(allocator, newAllocation, &mapped) != VK_SUCCESS)
        {
            vmaDestroyBuffer(allocator, newBuffer, newAllocation);
            return false;
        }

        std::memcpy(mapped, source, static_cast<std::size_t>(byteCount));

        const VkResult flushResult =
            vmaFlushAllocation(allocator, newAllocation, 0, byteCount);

        vmaUnmapMemory(allocator, newAllocation);

        if (flushResult != VK_SUCCESS)
        {
            vmaDestroyBuffer(allocator, newBuffer, newAllocation);
            return false;
        }

        buffer = newBuffer;
        allocation = newAllocation;
        return true;
    }

    void destroyMeshBuffers()
    {
        auto allocator = graphics::internal::context.allocator;

        if (indexBuffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(allocator, indexBuffer, indexAllocation);
            indexBuffer = VK_NULL_HANDLE;
            indexAllocation = nullptr;
        }

        if (vertexBuffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(allocator, vertexBuffer, vertexAllocation);
            vertexBuffer = VK_NULL_HANDLE;
            vertexAllocation = nullptr;
        }
    }

    void destroyUniformResources()
    {
        auto& context = graphics::internal::context;

        if (descriptorPool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
            descriptorPool = VK_NULL_HANDLE;
            globalDescriptorSet = VK_NULL_HANDLE;
        }

        if (uniformBuffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(context.allocator, uniformBuffer, uniformAllocation);
            uniformBuffer = VK_NULL_HANDLE;
            uniformAllocation = nullptr;
        }
    }

    bool createUniformResources()
    {
        auto& context = graphics::internal::context;

        const GlobalUniforms initialValues{
            kIdentity,
            kIdentity,
            kIdentity,
            {objectColor[0], objectColor[1], objectColor[2], 1.0f},
        };

        if (!createStaticBuffer(
            &initialValues, sizeof(initialValues),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            uniformBuffer, uniformAllocation))
        {
            return false;
        }

        const VkDescriptorPoolSize poolSize{
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
        };

        const VkDescriptorPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &poolSize,
        };

        if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr,
                                   &descriptorPool) != VK_SUCCESS)
        {
            return false;
        }

        const VkDescriptorSetAllocateInfo setInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &globalSetLayout,
        };

        if (vkAllocateDescriptorSets(context.device, &setInfo,
                                     &globalDescriptorSet) != VK_SUCCESS)
        {
            return false;
        }

        const VkDescriptorBufferInfo bufferInfo{
            .buffer = uniformBuffer,
            .offset = 0,
            .range = sizeof(GlobalUniforms),
        };

        const VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = globalDescriptorSet,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &bufferInfo,
        };

        vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
        return true;
    }

    Mat4 makePerspective(float aspect, float nearPlane, float farPlane)
    {
        const float scale = 1.0f / std::tan(kHalfVerticalFov);

        Mat4 result{};
        result.m[0] = scale / aspect;
        result.m[5] = scale;
        result.m[10] = farPlane / (farPlane - nearPlane);
        result.m[11] = 1.0f;
        result.m[14] = -nearPlane * farPlane / (farPlane - nearPlane);
        return result;
    }

    Mat4 makeOrthographic(float aspect, float halfHeight,
                          float nearPlane, float farPlane)
    {
        Mat4 result = kIdentity;
        result.m[0] = 1.0f / (aspect * halfHeight);
        result.m[5] = 1.0f / halfHeight;
        result.m[10] = 1.0f / (farPlane - nearPlane);
        result.m[14] = -nearPlane / (farPlane - nearPlane);
        return result;
    }

    Mat4 multiply(const Mat4& left, const Mat4& right)
    {
        Mat4 result{};
        for (int column = 0; column < 4; ++column)
        {
            for (int row = 0; row < 4; ++row)
            {
                for (int k = 0; k < 4; ++k)
                {
                    result.m[column * 4 + row] +=
                        left.m[k * 4 + row] * right.m[column * 4 + k];
                }
            }
        }
        return result;
    }

    Mat4 makeTranslation(const float position[3])
    {
        Mat4 result = kIdentity;
        result.m[12] = position[0];
        result.m[13] = position[1];
        result.m[14] = position[2];
        return result;
    }

    Mat4 makeScale(const float scale[3])
    {
        Mat4 result = kIdentity;
        result.m[0] = scale[0];
        result.m[5] = scale[1];
        result.m[10] = scale[2];
        return result;
    }

    Mat4 makeRotationX(float angle)
    {
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        Mat4 result = kIdentity;
        result.m[5] = cosine;
        result.m[6] = sine;
        result.m[9] = -sine;
        result.m[10] = cosine;
        return result;
    }

    Mat4 makeRotationY(float angle)
    {
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        Mat4 result = kIdentity;
        result.m[0] = cosine;
        result.m[2] = -sine;
        result.m[8] = sine;
        result.m[10] = cosine;
        return result;
    }

    Mat4 makeRotationZ(float angle)
    {
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        Mat4 result = kIdentity;
        result.m[0] = cosine;
        result.m[1] = sine;
        result.m[4] = -sine;
        result.m[5] = cosine;
        return result;
    }

    Mat4 makeModelMatrix()
    {
        constexpr float radiansPerDegree =
            std::numbers::pi_v<float> / 180.0f;
        const float angleX = objectRotationDegrees[0] * radiansPerDegree;
        const float angleY = objectRotationDegrees[1] * radiansPerDegree;
        const float angleZ = objectRotationDegrees[2] * radiansPerDegree;

        const Mat4 manualRotation = multiply(
            makeRotationZ(angleZ),
            multiply(makeRotationY(angleY), makeRotationX(angleX)));
        return multiply(
            makeTranslation(objectPosition),
            multiply(makeRotationY(static_cast<float>(animationPhaseY)),
                     multiply(makeRotationX(static_cast<float>(animationPhaseX)),
                              multiply(manualRotation, makeScale(objectScale)))));
    }
} // namespace

namespace application
{
    bool initialize()
    {
        auto& context = graphics::internal::context;

        if (!loadShaderModule("shaders/tetra.vert.spv", vertexShader) ||
            !loadShaderModule("shaders/tetra.frag.spv", fragmentShader))
        {
            destroyPipelineSetup();
            return false;
        }

        const VkDescriptorSetLayoutBinding uniformBinding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags =
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        };

        const VkDescriptorSetLayoutCreateInfo setLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &uniformBinding,
        };

        if (vkCreateDescriptorSetLayout(
            context.device, &setLayoutInfo, nullptr,
            &globalSetLayout) != VK_SUCCESS)
        {
            std::cerr << "Cannot create descriptor set layout\n";
            destroyPipelineSetup();
            return false;
        }

        const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &globalSetLayout,
        };

        if (vkCreatePipelineLayout(
            context.device, &pipelineLayoutInfo, nullptr,
            &pipelineLayout) != VK_SUCCESS)
        {
            std::cerr << "Cannot create pipeline layout\n";
            destroyPipelineSetup();
            return false;
        }

        if (!createGraphicsPipeline())
        {
            destroyPipelineSetup();
            return false;
        }

        if (!createStaticBuffer(
                kVertices.data(),
                sizeof(kVertices),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                vertexBuffer,
                vertexAllocation) ||
            !createStaticBuffer(
                kIndices.data(),
                sizeof(kIndices),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                indexBuffer,
                indexAllocation))
        {
            std::cerr << "Cannot create mesh buffers\n";
            destroyMeshBuffers();
            destroyPipelineSetup();
            return false;
        }

        if (!createUniformResources())
        {
            std::cerr << "Cannot create uniform resources\n";
            destroyUniformResources();
            destroyMeshBuffers();
            destroyPipelineSetup();
            return false;
        }

        return true;
    }

    void shutdown()
    {
        auto& context = graphics::internal::context;
        vkQueueWaitIdle(context.graphics_queue);
        destroyUniformResources();
        destroyMeshBuffers();
        destroyPipelineSetup();
    }

    void update(double time)
    {
        if (previousUpdateTime >= 0.0 && animationPlaying)
        {
            const double elapsed = time - previousUpdateTime;
            if (elapsed > 0.0)
            {
                constexpr double fullTurn =
                    2.0 * std::numbers::pi_v<double>;
                animationPhaseY = std::fmod(
                    animationPhaseY + animationSpeedY * elapsed, fullTurn);
                animationPhaseX = std::fmod(
                    animationPhaseX + animationSpeedX * elapsed, fullTurn);
            }
        }
        previousUpdateTime = time;

        ImGui::Begin("Truncated tetrahedron");
        ImGui::Text("Vertices: %d", static_cast<int>(kVertices.size()));
        ImGui::Text("Triangles: %d", static_cast<int>(kIndices.size() / 3));
        if (ImGui::RadioButton("Perspective",
                               projectionMode == ProjectionMode::Perspective))
        {
            projectionMode = ProjectionMode::Perspective;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Orthographic",
                               projectionMode == ProjectionMode::Orthographic))
        {
            projectionMode = ProjectionMode::Orthographic;
        }
        ImGui::Separator();
        ImGui::DragFloat3("Position", objectPosition, 0.01f);
        ImGui::DragFloat3("Rotation (degrees)", objectRotationDegrees, 0.5f);
        ImGui::SliderFloat3("Scale", objectScale, 0.1f, 3.0f);
        if (ImGui::Button("Reset transform"))
        {
            for (int component = 0; component < 3; ++component)
            {
                objectPosition[component] = 0.0f;
                objectRotationDegrees[component] = 0.0f;
                objectScale[component] = 1.0f;
            }
        }
        ImGui::Separator();
        ImGui::TextUnformatted("Rotation around Y and X");
        if (ImGui::Button(animationPlaying ? "Pause" : "Play"))
        {
            animationPlaying = !animationPlaying;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset animation"))
        {
            animationPlaying = false;
            animationPhaseY = 0.0;
            animationPhaseX = 0.0;
        }
        ImGui::SliderFloat("Y speed (rad/s)", &animationSpeedY, 0.0f, 3.0f);
        ImGui::SliderFloat("X speed (rad/s)", &animationSpeedX, 0.0f, 3.0f);
        ImGui::Text("Y: %.2f rad, X: %.2f rad",
                    animationPhaseY, animationPhaseX);
        ImGui::Separator();
        ImGui::ColorEdit3("Figure color", objectColor);
        ImGui::End();
    }

    void render(const graphics::internal::FrameData& fd)
    {
        auto& context = graphics::internal::context;
        const VkExtent2D extent = context.swapchain_extent;
        if (extent.width == 0 || extent.height == 0)
        {
            std::cerr << "Cannot render a zero-sized framebuffer\n";
            std::abort();
        }

        // Lecture convention: forward is +Z, and screen +Y points down.
        Mat4 view = kIdentity;
        view.m[5] = -1.0f;
        view.m[10] = -1.0f;
        view.m[14] = kCameraDistance;

        const float aspect = static_cast<float>(extent.width) /
            static_cast<float>(extent.height);
        const float halfHeight =
            kCameraDistance * std::tan(kHalfVerticalFov);
        const Mat4 projection =
            projectionMode == ProjectionMode::Perspective
                ? makePerspective(aspect, 0.1f, 20.0f)
                : makeOrthographic(aspect, halfHeight, 0.1f, 20.0f);
        const GlobalUniforms uniforms{
            makeModelMatrix(),
            view,
            projection,
            {objectColor[0], objectColor[1], objectColor[2], 1.0f},
        };

        // prepare() has waited for the previous frame before render() is called.
        if (vmaCopyMemoryToAllocation(context.allocator, &uniforms,
                                      uniformAllocation, 0,
                                      sizeof(uniforms)) != VK_SUCCESS)
        {
            std::cerr << "Cannot update uniform buffer\n";
            std::abort();
        }

        const VkCommandBuffer commandBuffer = fd.command_buffer;
        if (vkResetCommandBuffer(commandBuffer, 0) != VK_SUCCESS)
        {
            std::cerr << "Cannot reset command buffer\n";
            std::abort();
        }

        const VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        };
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        {
            std::cerr << "Cannot begin command buffer\n";
            std::abort();
        }

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
            .renderArea = {
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
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer,
                               &vertexOffset);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0,
                             VK_INDEX_TYPE_UINT16);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1, &globalDescriptorSet,
                                0, nullptr);
        vkCmdDrawIndexed(commandBuffer,
                         static_cast<std::uint32_t>(kIndices.size()),
                         1, 0, 0, 0);

        vkCmdEndRenderPass(commandBuffer);
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        {
            std::cerr << "Cannot end command buffer\n";
            std::abort();
        }
    }
} // namespace application
