#pragma once

#include <cstdint>

#include <vulkan/vulkan_core.h>

#include <vk_mem_alloc.h>

struct GLFWwindow;

namespace graphics::internal
{
    struct Context
    {
        // PhysicalDevice описывает выбранный GPU, Device — созданное логическое устройство.
        VkPhysicalDevice physical_device;
        VkDevice device;

        VmaAllocator allocator; // Выделяет память для VkBuffer/VkImage.

        VkQueue graphics_queue; // В эту очередь отправляются command buffers.
        uint32_t graphics_queue_index;

        VkFormat swapchain_format; // Формат и размер изображения окна.
        VkExtent2D swapchain_extent;

        VkRenderPass render_pass; // Проход фигуры: цвет + depth attachment.
    };

    struct FrameData
    {
        // Конкретный framebuffer полученного swapchain image и список команд фигуры.
        VkFramebuffer framebuffer;
        VkCommandBuffer command_buffer;
    };

    extern Context context;

    bool initialize(GLFWwindow* const window);
    void shutdown();

    void resize(uint32_t width, uint32_t height);

    // Ждёт завершения прошлого кадра и получает следующее изображение swapchain.
    FrameData prepare();
    // Отправляет команды фигуры/UI и показывает результат после semaphore.
    void submitAndPresent();
} // namespace graphics::internal
