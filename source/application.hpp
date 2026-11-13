#pragma once

#include "graphics_internal.hpp"

namespace application
{
    // Создать pipeline, геометрию с UV в памяти GPU, текстуры и наборы scene/material.
    bool initialize();
    // Дождаться GPU и освободить текстуры, буферы модели и освещения.
    void shutdown();

    // Обновить CPU-параметры через ImGui и время анимации; буферы GPU здесь не меняем.
    void update(double time);
    // После ожидания fence загрузить параметры кадра и записать команды фигуры.
    void render(const graphics::internal::FrameData& fd);
} // namespace application
