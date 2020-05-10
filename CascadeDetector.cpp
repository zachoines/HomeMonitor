#include "CascadeDetector.h"

namespace HM {
	DetectionData CascadeDetector::detect(cv::Mat& src, std::string target)
	{
		return DetectionData();
	}

	CascadeDetector::CascadeDetector(cv::dnn::Net net, std::string* class_names)
	{
		this->class_names = class_names;
		this->net = &net;
	}

}
