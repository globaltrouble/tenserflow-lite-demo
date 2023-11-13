cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cptrie VERSION 1.0.0 LANGUAGES C CXX)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/tenserflow/tensorflow/lite/)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Werror -Wall -Wextra -Wno-comment -Wno-unused-parameter)

add_executable(tflite-demo 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_link_libraries(tflite-demo tensorflow-lite)

