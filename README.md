# TensorFlow Lite with BERT Tokenizer Demo App

This C++ demo application showcases the usage of BERT tokenizer bindings with TensorFlow Lite. The BERT tokenizer is integrated using Rust bindings, providing efficient tokenization capabilities for natural language processing tasks.

Also can be used to debug `tflite` model inputs/outputs. Set env var `TFLITE_DEMO_TRACE_MODEL=1` to be see model layers debug info.

## Prerequisites

- C++ Compiler (e.g., g++) and `cmake` build system for building the demo app.
- `rustc` compiler and `cargo` build system (for building the Rust bindings)

## Build

```
# init submodules (tflite and bert bindings)
git submodule update --init --recursive

# build
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -j8
```
## Run

```
# trace model layers
export TFLITE_DEMO_TRACE_MODEL=1

# assume run from project root
./build/tflite-demo static/model.rflite static/vocab.txt 'Your text to tokenize and process with bert'
```
