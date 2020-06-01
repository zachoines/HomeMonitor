#include "CaffeDetector.h"

#include <vector>
#include <string>

namespace HM {

	DetectionData CaffeDetector::detect(cv::Mat& src, std::string target)
	{
	
		cv::Mat blobimg = cv::dnn::blobFromImage(src, this->scale, cv::Size(300, 300), this->mean);

		this->net->setInput(blobimg, "data");

		cv::Mat detection = this->net->forward("detection_out");

		cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		for (int i = 0; i < detectionMat.rows; i++) {

			size_t det_index = (size_t)detectionMat.at<float>(i, 1);

			if (this->class_names[det_index] == target) {

				float detect_confidence = detectionMat.at<float>(i, 2);
				if (detect_confidence > this->confidence_threshold) {

					float x1 = detectionMat.at<float>(i, 3) * src.cols;
					float y1 = detectionMat.at<float>(i, 4) * src.rows;
					float x2 = detectionMat.at<float>(i, 5) * src.cols;
					float y2 = detectionMat.at<float>(i, 6) * src.rows;

					int frameWidth = (int)(x2 - x1);
					int frameHeight = (int)(y2 - y1);

					int objX = (int)x1 + ((int)frameWidth / 2);
					int objY = (int)y1 + ((int)frameHeight / 2);

					this->last_frame = &src;
					struct Rect objectBounds = { .x = x1, .y = y1, .height = frameHeight , .width = frameWidth };
					struct DetectionData detectionResults = { .targetCenterX = objX, .targetCenterY = objY, .confidence = detect_confidence, .found = true, .target = target, .boundingBox = objectBounds };
					return detectionResults;
				}
			}
		}

		struct DetectionData detectionResults;
		detectionResults.found = false;
		return detectionResults;
	}

	// Draws bounding box and center circle onto frame. Good for debugging.
	DetectionData CaffeDetector::detect(cv::Mat& src, std::string target, bool draw)
	{
		struct DetectionData detectionResults = this->detect(src, target);
		if (draw) {
			if (detectionResults.found) {
				cv::Scalar red = cv::Scalar(0, 0, 255);
				cv::Rect rec(
					detectionResults.boundingBox.x, 
					detectionResults.boundingBox.y, 
					detectionResults.boundingBox.width, 
					detectionResults.boundingBox.height
				);
				circle(
					src, 
					cv::Point(detectionResults.targetCenterX, detectionResults.targetCenterY), 
					(int)(detectionResults.boundingBox.width + detectionResults.boundingBox.height) / 2 / 10,
					red, 2, 8, 0);
				rectangle(src, rec, red, 2, 8, 0);
				putText(
					src, 
					target, 
					cv::Point(detectionResults.boundingBox.x, detectionResults.boundingBox.y - 5), 
					cv::FONT_HERSHEY_SIMPLEX, 
					1.0, 
					red, 2, 8, 0 );
			}
		}

		return detectionResults;
	}


	void CaffeDetector::setConfidenceThreshold(float ct)
	{
		if (ct < 0) {
			throw "Confidence threshold must be a non-negative floating point value";
		}

		this->confidence_threshold = ct;
	}

	CaffeDetector::CaffeDetector(cv::dnn::Net net, std::vector<std::string> class_names)
	{
	
		this->class_names = class_names;

		this->net = &net;

		if (this->net->empty()) {
			throw "Caffe Network not initialized";
		}
	}
}