#include <cstdint>
#include <climits>

#include <iostream>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include "graphics_internal.hpp"
#include "application.hpp"

namespace
{
    constexpr int32_t default_window_width = 1280;
    constexpr int32_t default_window_height = 720;

    constexpr char default_window_title[] = "CG Lab 2 - Camera and lighting";

    GLFWwindow* glfw_window;
} // namespace

int main()
{
    // Сначала окно и CPU-интерфейс, затем Vulkan, затем ресурсы самой модели.
    int status = EXIT_SUCCESS;

    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW\n";
        return EXIT_FAILURE;
    }

    // OpenGL-контекст не нужен: изображение окна формируется через Vulkan surface.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    glfw_window = glfwCreateWindow(default_window_width, default_window_height,
                                   default_window_title, nullptr, nullptr);
    if (glfw_window == nullptr)
    {
        status = EXIT_FAILURE;
        goto err_null_window;
    }

    // Размер framebuffer может отличаться от размера окна (например, на Retina).
    glfwSetFramebufferSizeCallback(glfw_window, [](GLFWwindow*, int width, int height)
    {
        if (width == 0 || height == 0)
        {
            return;
        }

        // Только отмечаем resize; пересоздание GPU-изображений выполняет graphics.
        graphics::internal::resize(width, height);
    });

    if (ImGui::CreateContext() == nullptr)
    {
        std::cerr << "Failed to create ImGUI context\n";
        status = EXIT_FAILURE;
        goto err_imgui_init;
    }

    // Backend GLFW передаёт ImGui клавиатуру и мышь, Vulkan backend рисует UI.
    if (!ImGui_ImplGlfw_InitForVulkan(glfw_window, true))
    {
        std::cerr << "Failed to initialize ImGUI GLFW backend for Vulkan renderer\n";
        status = EXIT_FAILURE;
        goto err_imgui_glfw_init;
    }

    if (!graphics::internal::initialize(glfw_window))
    {
        std::cerr << "Failed to initialize graphics\n";
        status = EXIT_FAILURE;
        goto err_graphics_init;
    }

    if (!application::initialize())
    {
        std::cerr << "Failed to initialize application\n";
        status = EXIT_FAILURE;
        goto err_application_init;
    }

    while (!glfwWindowShouldClose(glfw_window))
    {
        // GLFW даёт время в секундах; update вычисляет delta time для вращения.
        const double time = glfwGetTime();

        glfwPollEvents();
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();

        ImGui::NewFrame();
        application::update(time);
        ImGui::Render();

        // prepare ждёт GPU-fence. Только после него render меняет UBO/SSBO и descriptors.
        graphics::internal::FrameData fd = graphics::internal::prepare();
        if (fd.command_buffer == VK_NULL_HANDLE)
        {
            status = EXIT_FAILURE;
            break;
        }
        application::render(fd);
        // Список команд фигуры + ImGui отправляется GPU, затем запрашивается present.
        graphics::internal::submitAndPresent();
    }

    // Освобождаем ресурсы в обратном порядке; метки также используются при ошибке запуска.
    application::shutdown();
err_application_init:
    graphics::internal::shutdown();
err_graphics_init:
    ImGui_ImplGlfw_Shutdown();
err_imgui_glfw_init:
    ImGui::DestroyContext();
err_imgui_init:
    glfwDestroyWindow(glfw_window);
err_null_window:
    glfwTerminate();

    return status;
}
