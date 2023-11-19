#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

#include "rustbert_tokenizer.h"

#include <iostream>
#include <chrono>
#include <iomanip>

struct ProfileIt {
#ifdef NDEBUG
  ProfileIt(char const * const) {};
#else
  char const * const m_name = nullptr;
  std::chrono::steady_clock::time_point m_begin;
  
  ProfileIt(char const * const name) : m_name(name), m_begin(std::chrono::steady_clock::now()) {}

  ~ProfileIt() {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_begin);
    std::cerr << m_name << ": " << std::fixed << std::setprecision(6) << (elapsed.count() * 0.000001) << " sec \n";
  }
#endif
};

int main(int argc, char const * const * argv) {
    if (argc != 4) {
        std::cerr << "Usage ./progname path/to/model path/to/vocab text";
        std::exit(1);
    }

    char const * modelFname = argv[1];
    char const * vocabFname = argv[2];
    char const * text = argv[3];

    std::cerr << "Model path: `" << modelFname << ", text: `" << text << "`\n";

    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;

    std::vector<int> const * inputs;
    std::vector<int> const * outputs;
    {

        ProfileIt loadmodel("Load model");

        // Can be build from memory
        //   tflite::FlatBufferModel::BuildFromBuffer(serialized_flatbuffer.c_str(), serialized_flatbuffer.size());
        model = tflite::FlatBufferModel::BuildFromFile(modelFname);
        if (model == nullptr) {
            std::cerr << "Can't load model\n";
            std::exit(2);
        }

        // Build the interpreter
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*model, resolver)(&interpreter);
        if (interpreter == nullptr) {
            std::cerr << "Failed to construct interpreter\n";
            std::exit(3);
        }

        // Resize input tensors, if desired.
        interpreter->AllocateTensors();

        const bool lowercase = true;
        const bool stripAccents = true;
        enum BertTokenizerInitStatus code = bert_tokenizer_init(vocabFname, lowercase, stripAccents);
        assert(code == BertTokenizerInitStatusOK);
        (void) code;

        interpreter->SetNumThreads(8);

        inputs = &interpreter->inputs();
        outputs = &interpreter->outputs();

        char const * trace_model = std::getenv("TFLITE_DEMO_TRACE_MODEL");
        if (trace_model != nullptr && strlen(trace_model) == 1 && *trace_model == '1') {
            std::cerr << "tensors size: " << interpreter->tensors_size() << "\n";
            std::cerr << "nodes size: " << interpreter->nodes_size() << "\n";
            std::cerr << "inputs: " << interpreter->inputs().size() << "\n";
            std::cerr << "input(0) name: " << interpreter->GetInputName(0) << "\n";

            int t_size = interpreter->tensors_size();
            for (int i = 0; i < t_size; i++) {
            if (interpreter->tensor(i)->name)
                std::cerr << i << ": " << interpreter->tensor(i)->name << ", "
                        << interpreter->tensor(i)->bytes << ", "
                        << interpreter->tensor(i)->type << ", "
                        << interpreter->tensor(i)->params.scale << ", "
                        << interpreter->tensor(i)->params.zero_point << "\n";
            }

            std::cerr << "number of inputs: " << inputs->size() << "\n";
            for (uint32_t i = 0; i < inputs->size(); i++) {
                int inpt = (*inputs)[i];
                std::cerr << "input[" << i << "]: " << inpt << "\n";

                TfLiteIntArray* inDims = interpreter->tensor(inpt)->dims;
                std::cerr << "Input dims: " << inDims->data[1] << "\n";
                std::cerr << "Input type: " << interpreter->tensor(inpt)->type << "\n";
            }

            std::cerr << "number of outputs: " << outputs->size() << "\n";
            for (uint32_t i = 0; i < outputs->size(); i++) {
                int outpt = (*outputs)[i];
                std::cerr << "output[" << i << "]: " << outpt << "\n";

                TfLiteIntArray* outDims = interpreter->tensor(outpt)->dims;
                std::cerr << "Output dims: " << outDims->data[1] << "\n";
                std::cerr << "Output type: " << interpreter->tensor(outpt)->type << "\n";
            }

            tflite::PrintInterpreterState(interpreter.get());
        }
        // kTfLiteNoType = 0,
        // kTfLiteFloat32 = 1,
        // kTfLiteInt32 = 2,
        // kTfLiteUInt8 = 3,
        // kTfLiteInt64 = 4,
        // kTfLiteString = 5,
        // kTfLiteBool = 6,
        // kTfLiteInt16 = 7,
        // kTfLiteComplex64 = 8,
        // kTfLiteInt8 = 9,
        // kTfLiteFloat16 = 10,
        // kTfLiteFloat64 = 11,
        // kTfLiteComplex128 = 12,
        // kTfLiteUInt64 = 13,
        // kTfLiteResource = 14,
        // kTfLiteVariant = 15,
        // kTfLiteUInt32 = 16,
        // kTfLiteUInt16 = 17,
        // kTfLiteInt4 = 18,
    }

    {
        ProfileIt preproc("Preprocess");

        size_t const idInpt = 0;
        size_t const typeInpt = 1;
        size_t const maskInpt = 2;

        assert(inputs->size() == 3);
        assert(interpreter->tensor(inputs->at(idInpt))->type == kTfLiteInt64);
        assert(interpreter->tensor(inputs->at(typeInpt))->type == kTfLiteInt64);
        assert(interpreter->tensor(inputs->at(maskInpt))->type == kTfLiteInt64);

        int64_t* tokenId = interpreter->typed_input_tensor<int64_t>(idInpt);
        int64_t* tokenType = interpreter->typed_input_tensor<int64_t>(typeInpt);
        int64_t* maskType = interpreter->typed_input_tensor<int64_t>(maskInpt);

        int const inputLen = interpreter->tensor(inputs->at(idInpt))->dims->data[1];

        assert(interpreter->tensor(inputs->at(typeInpt))->dims->data[1] == inputLen);
        assert(interpreter->tensor(inputs->at(maskInpt))->dims->data[1] == inputLen);

        enum BertTokenizerPreprocessingStatus procStatus = bert_tokenizer_process(text, tokenId, tokenType, maskType, static_cast<size_t>(inputLen));
        assert(procStatus == BertTokenizerPreprocessingStatusOK);
        (void) procStatus;
    }

    {
        ProfileIt inf("Inference: ");

        interpreter->Invoke();
    }

    std::cerr << "DOne!\n";

    return 0;
}
