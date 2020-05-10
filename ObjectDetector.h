#pragma once
#include <stdio.h>
#include <unistd.h>
#include <string>

#include "opencv2/opencv.hpp"
#include "opencv4/opencv2/dnn/dnn.hpp"
#include "opencv2/core/core.hpp"

namespace HM {
	struct Rect {
		int x;
		int y;
		int height;
		int width;
	};
	struct DetectionData
	{
		int targetCenterX;
		int targetCenterY;
		double confidence;
		bool found;
		std::string target;
		struct Rect boundingBox;
	};
	class ObjectDetector
	{
	protected:
		std::string* class_names = nullptr;
		cv::Mat* last_frame = nullptr;
		cv::dnn::Net* net = nullptr;

	public:
		ObjectDetector();
		virtual struct DetectionData detect(cv::Mat& src, std::string target) = 0;

	};
}

