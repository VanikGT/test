#ifndef CONV_SDK_
#define CONV_SDK_

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// tflite
#include "tensorflow/lite/optional_debug_tools.h"
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

#include <string>
#include <vector>

namespace tensorflow {

class Conv_SDK
{
public:
	Conv_SDK(const std::string& path);

	void InitializeInput(const cv::Mat& image);
    std::vector<int> Inference();
	std::vector<int> Run(const cv::Mat& image);

protected:
	void Preprocess(const cv::Mat& before, cv::Mat& after);
	void Postprocess(const cv::Mat& before, cv::Mat& after);

	// model, resolver and interpreter
	std::unique_ptr<tflite::Interpreter> cpu_interpreter_;
	tflite::ops::builtin::BuiltinOpResolver resolver_;
	std::unique_ptr<tflite::FlatBufferModel> model_;
	TfLiteDelegate* cpu_delegate_;

	// image size
	int w_ = 128;
	int h_ = 128;
};

}
#endif // CONV_SDK
