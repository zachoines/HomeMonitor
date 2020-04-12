#include <stdio.h>
#include <unistd.h>
 
#include <iostream>
#include <fstream>
#include <string>
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



using namespace cv;
using namespace std;

const size_t width = 300;
const size_t height = 300;
const float scaleFector = 0.007843f;
const float meanVal = 127.5;

dnn::Net net;

const char* class_video_Names[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor" };

Mat detect_from_video(Mat& src)
{
	Mat blobimg = dnn::blobFromImage(src, scaleFector, Size(300, 300), meanVal);

	net.setInput(blobimg, "data");

	Mat detection = net.forward("detection_out");
	//	cout << detection.size[2]<<" "<< detection.size[3] << endl;
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	const float confidence_threshold = 0.25;
	for (int i = 0; i < detectionMat.rows; i++) {
		float detect_confidence = detectionMat.at<float>(i, 2);

		if (detect_confidence > confidence_threshold) {
			size_t det_index = (size_t)detectionMat.at<float>(i, 1);
			float x1 = detectionMat.at<float>(i, 3) * src.cols;
			float y1 = detectionMat.at<float>(i, 4) * src.rows;
			float x2 = detectionMat.at<float>(i, 5) * src.cols;
			float y2 = detectionMat.at<float>(i, 6) * src.rows;
			Rect rec((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
			rectangle(src, rec, Scalar(0, 0, 255), 2, 8, 0);
			putText(src, format("%s", class_video_Names[det_index]), Point(x1, y1 - 5), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8, 0);
		}
	}
	return src;
}


bool fileExists(const char* fileName)
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



int main(int argc, char** argv)
{
	float f;
	float FPS[16];
	int i, Fcnt = 0;
	Mat frame;
	chrono::steady_clock::time_point Tbegin, Tend;
	cv::VideoCapture camera(0);
	std::cout << get_current_dir_name() << std::endl;


	std::ifstream test(".\\MobileNetSSD_deploy.prototxt");
	if (!test)
	{
		std::cout << "The file doesn't exist" << std::endl;
		exit(-1);
	}

	if (fileExists("MobileNetSSD_deploy.prototxt")) {
		std::cout << "Everything works!!" << std::endl;
	}

	net = dnn::readNetFromCaffe("/MobileNetSSD_deploy.prototxt", "/MobileNetSSD_deploy.caffemodel");
	if (net.empty()) {
		std::cout << "init the model net error";
		exit(-1);
	}

	std::cout << "Start grabbing, press ESC on Live window to terminate" << endl;
	while ((cv::waitKey(1) & 0xFF) != 27) {
		cv::Mat frame = GetImageFromCamera(camera);

		Tbegin = chrono::steady_clock::now();

		detect_from_video(frame);

		Tend = chrono::steady_clock::now();
		//calculate frame rate
		f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
		if (f > 0.0) FPS[((Fcnt++) & 0x0F)] = 1000.0 / f;
		for (f = 0.0, i = 0; i < 16; i++) { f += FPS[i]; }
		putText(frame, format("FPS %0.2f", f / 16), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255));
		//show output
		// imshow("frame", frame);

		char esc = waitKey(5);
		if (esc == 27) break;
	}

	std::cout << "Closing the camera" << endl;
	cv::destroyAllWindows();
	std::cout << "Bye!" << endl;

	return 0;
}