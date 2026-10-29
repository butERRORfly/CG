#pragma once

#include "graphics_internal.hpp"

namespace application
{
    // Создать шейдеры, pipeline, геометрию с нормалями и наборы scene/model.
    bool initialize();
    // Дождаться GPU и освободить ресурсы модели и освещения.
    void shutdown();

    // Обновить CPU-параметры через ImGui и время анимации; буферы GPU здесь не меняем.
    void update(double time);
    // После ожидания fence загрузить параметры кадра и записать команды фигуры.
    void render(const graphics::internal::FrameData& fd);
} // namespace application
