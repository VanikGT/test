#include "conv_sdk.hpp"

// standard includes
#include <iostream>
#include <cassert>

namespace tensorflow {

Conv_SDK::Conv_SDK(const std::string& path) {
    model_ = tflite::FlatBufferModel::BuildFromFile(path.c_str());

    std::cout << "Successfully loaded tflite model: " << path;
    if (!model_) std::cout << "Failed to load model: " << path;

    model_->error_reporter();

    tflite::InterpreterBuilder(*model_, resolver_)(&cpu_interpreter_);
    if (!cpu_interpreter_) std::cout << "Failed to construct interpreter";

    cpu_interpreter_->SetNumThreads(1);

    if (cpu_interpreter_->AllocateTensors() != kTfLiteOk) std::cout << "Failed to allocate tensors";
    std::cout << "Successfully allocated tensors";

    const std::vector<int> input_vec = cpu_interpreter_->inputs();
    std::cout << "num inputs: " << input_vec.size();
    std::cout << "input name: " << cpu_interpreter_->GetInputName(0);

    const std::vector<int> results = cpu_interpreter_->outputs();
    std::cout << "num outputs: " << results.size();
    int i = 0;
    for (auto& v : results) {
        std::cout << "output: " << v;
        std::cout << "output name: " << cpu_interpreter_->GetOutputName(i);
        i++;
    }

    tflite::PrintInterpreterState(cpu_interpreter_.get());
}

void Conv_SDK::Preprocess(const cv::Mat& before, cv::Mat& after) {
    after = before;
}

void Conv_SDK::Postprocess(const cv::Mat& before, cv::Mat& after) {
    after = before;
}

void Conv_SDK::InitializeInput(const cv::Mat& image)
{
    cv::Mat cameraImg(h_, w_, CV_32FC3, cpu_interpreter_->typed_input_tensor<float>(0));
    cv::resize(image, image, cv::Size(w_, h_), 0, 0, CV_INTER_NN);
    cv::cvtColor(image, image, CV_BGR2RGB);
    image.convertTo(cameraImg, CV_32FC3, 1/255.0);
}

std::vector<int> Conv_SDK::Inference() {
    if (cpu_interpreter_->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke!";
    }
    cv::Mat out(1, cpu_interpreter_->outputs().size(), CV_32FC1, cpu_interpreter_->typed_output_tensor<float>(0));
    std::cout << out << "\n";
    std::vector<int> network_result;
    for (int i = 0; i < out.cols; ++i) {
        network_result.push_back(int(out.at<float>(0,i) * w_));
    }
    assert(network_result.size() % 2 == 0);
    return network_result;
}

std::vector<int> Conv_SDK::Run(const cv::Mat& image)
{
    InitializeInput(image);
    return Inference();
}


}
