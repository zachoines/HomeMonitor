#pragma once
#include "ObjectDetector.h"
namespace HM {
	class CascadeDetector : public ObjectDetector
	{
	public:
		CascadeDetector(cv::dnn::Net net, std::string* class_names);
		struct DetectionData detect(cv::Mat& src, std::string target);
	private:
	};
};

