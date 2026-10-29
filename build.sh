#!/bin/bash

cmake --preset debug -DFETCHCONTENT_QUIET=OFF
cmake --build build-debug --parallel
./build-debug/vulkan-starter-app
