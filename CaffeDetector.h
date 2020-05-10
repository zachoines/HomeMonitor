#pragma once
#include "ObjectDetector.h"

namespace HM {
	class CaffeDetector : public ObjectDetector
	{
	private:
		float scale = 0.007843f;
		float mean = 127.5;
		size_t width = 300;
		size_t height = 300;
		float confidence_threshold = 0.25;

	public:
		CaffeDetector();
		CaffeDetector(cv::dnn::Net net, std::string* class_names);
		struct DetectionData detect(cv::Mat& src, std::string target);
		struct DetectionData detect(cv::Mat& src, std::string target, bool draw);

		// Getters Setters
		void setConfidenceThreshold(float ct);
	};
}

