cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cptrie VERSION 1.0.0 LANGUAGES C CXX)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/tenserflow/tensorflow/lite/)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Werror -Wall -Wextra -Wno-comment -Wno-unused-parameter)

add_custom_target(
    build_bert_tokenizer_bindings
    COMMAND cargo build --release
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/deps/bert-tokenizer-c-cpp-bindings/
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/bert-tokenizer-c-cpp-bindings/target/include)
add_library(rustbert_tokenizer STATIC IMPORTED)
set_target_properties(rustbert_tokenizer PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/deps/bert-tokenizer-c-cpp-bindings/target/release/librustbert_tokenizer.a)

add_executable(tflite-demo 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)
add_dependencies(tflite-demo  build_bert_tokenizer_bindings)


target_link_libraries(tflite-demo tensorflow-lite rustbert_tokenizer)

