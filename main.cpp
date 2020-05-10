// Standard Libs
#include <dirent.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iostream>
#include <filesystem>

// 3rd party Dynamic libs
#include <wiringPi.h>

// OpenCV imports
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv4/opencv2/dnn/dnn.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/video/video.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

// Local classes
#include "CaffeDetector.h"

using namespace cv;
using namespace std;

const size_t width = 300;
const size_t height = 300;
const float scaleFector = 0.007843f;
const float meanVal = 127.5;

dnn::Net net;

const char* class_video_Names[] = { 
	"background",
	"aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair",
	"cow", "diningtable", "dog", "horse",
	"motorbike", "person", "pottedplant",
	"sheep", "sofa", "train", "tvmonitor" 
};

Mat detect_from_video(Mat& src)
{
	Mat blobimg = dnn::blobFromImage(src, scaleFector, Size(300, 300), meanVal);

	net.setInput(blobimg, "data");

	Mat detection = net.forward("detection_out");
	
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	const float confidence_threshold = 0.25;
	int rows = detectionMat.rows;
	for (int i = 0; i < detectionMat.rows; i++) {
		float detect_confidence = detectionMat.at<float>(i, 2);

		if (detect_confidence > confidence_threshold) {
			size_t det_index = (size_t)detectionMat.at<float>(i, 1);
			if (det_index == 15) {
				float x1 = detectionMat.at<float>(i, 3) * src.cols;
				float y1 = detectionMat.at<float>(i, 4) * src.rows;
				float x2 = detectionMat.at<float>(i, 5) * src.cols;
				float y2 = detectionMat.at<float>(i, 6) * src.rows;

				int width = (int)(x2 - x1);
				int height = (int)(y2 - y1);

				int objX = (int)x1 + ((int)width / 2);
				int objY = (int)y1 + ((int)height / 2);

				Rect rec((int)x1, (int)y1, width, height);
				circle(src, Point(objX, objY), 30,  Scalar(0, 0, 255), 2, 8, 0);
				rectangle(src, rec, Scalar(0, 0, 255), 2, 8, 0);
				putText(src, format("%s", class_video_Names[det_index]), Point(x1, y1 - 5), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8, 0);
			}
			
		}
	}
	return src;
}


bool fileExists(std::string fileName)
{
	std::ifstream test(fileName);
	return (test) ? true : false;
}


static cv::Mat GetImageFromCamera(cv::VideoCapture& camera)
{
	cv::Mat frame;
	camera >> frame;
	return frame;
}

int printDirectory(const char* path) {
	DIR* dir;
	struct dirent* ent;
	if ((dir = opendir(path)) != NULL) {
		
		while ((ent = readdir(dir)) != NULL) {
			printf("%s\n", ent->d_name);
		}
		closedir(dir);
	}
	else {
		
		perror("");
		return EXIT_FAILURE;
	}

	return 0;
}

int main(int argc, char** argv)
{
 	bool showVideo = true;
	float f;
	float FPS[16];
	int i, Fcnt = 0;
	cv::Mat frame;
	cv::Mat detection;
	chrono::steady_clock::time_point Tbegin, Tend;
	
	std::string prototextFile = "/MobileNetSSD_deploy.prototxt";
	std::string modelFile = "/MobileNetSSD_deploy.caffemodel";
	std::string path = get_current_dir_name();
	std::string prototextFilePath = path + prototextFile;
	std::string modelFilePath = path + modelFile;


	cv::VideoCapture camera;
	camera.open(1);
	sleep(3);

	if (!camera.isOpened())
	{
		cout << "Cannot open the camera!" << endl;
		exit(-1);
	}
	
	if (fileExists(modelFilePath) && fileExists(prototextFilePath)) {
		net = dnn::readNetFromCaffe(prototextFilePath, modelFilePath);
		if (net.empty()) {
			std::cout << "Error initializing caffe model" << std::endl;
			exit(-1);
		}
	} else {
		std::cout << "Error finding model and prototext files" << std::endl;
		exit(-1);
	}

	while (waitKey(1) < 0) {
		frame = GetImageFromCamera(camera);
		

		Tbegin = chrono::steady_clock::now();
		if (frame.empty()) 
		{
			std::cout << "Issue reading frame!" << std::endl; 
			sleep(1);
			continue;
			// exit(-1);
		} 
			
		detection = detect_from_video(frame);
		
		if (showVideo) {
			Tend = chrono::steady_clock::now();
			f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
			if (f > 0.0) FPS[((Fcnt++) & 0x0F)] = 1000.0 / f;
			for (f = 0.0, i = 0; i < 16; i++) { f += FPS[i]; }

			putText(frame, format("FPS %0.2f", f / 16), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255));
			//show output

			imshow("frame", detection);
		}
	}

	cv::destroyAllWindows();

	return 0;
}