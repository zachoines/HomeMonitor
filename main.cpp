// C++ libs
#include <iostream>
#include <fstream>
#include <string>
#include <iostream>
#include <filesystem>

// C libs
#include <dirent.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h> 
#include <sys/prctl.h>
#include <signal.h>
#include <ctime>

// 3rd party Dynamic libs
#include <wiringPi.h>
#include <wiringPiI2C.h>
#include <softPwm.h>  /* include header file for software PWM */

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
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>


// Local classes
#include "CaffeDetector.h"
#include "CascadeDetector.h"
#include "PID.h"

#define PWM0 13													// PWM0 -- RASPI pin #33 -- GPIO 18 -- WiringPi Pin #1;
#define HW_PMW false											// whether to use hardware pwm
#define PWM_CLOCK 1920											// Clock divider 
#define PWM_RANGE 200											// Number of increments
#define PWM_BASE_CLOCK 19.2e6                                   // The Raspberry Pi PWM clock has a base frequency of 19.2 MHz
#define PWM_FREQUENCY PWM_BASE_CLOCK / PWM_CLOCK / PWM_RANGE    // The resulting frequency of wave form on GPIO line
#define ARDUINO_ADDRESS 0x9

// 50Hz epected clock for a 20ms cycle on the Tower Pro SG90 servo, hense the 1920 clock divider and 200 range
// 2ms pulse will give 90 degrees, 1.5ms pulse for 0 degrees, and 1.0ms pulse for -90 degrees;
// Note: dutyCycle = pulseWidth * frequency;

using namespace cv;
using namespace std;

pid_t parent_pid;

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

int mapOutput(int x, int in_min, int in_max, int out_min, int out_max) {
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

// Return response data
void sendComand(unsigned char command, unsigned char data, int fd) {
	unsigned short finalCommand = (command << 8) + data;
	wiringPiI2CWriteReg16(fd, 0, finalCommand);	
}

// FD read that handles interrupts
ssize_t r_read(int fd, void* buf, size_t size)
{
	ssize_t retval;

	while (retval = read(fd, buf, size), retval == -1 && errno == EINTR);

	return retval;
}

// FD write that handles interrupts
ssize_t r_write(int fd, void* buf, size_t size)
{
	char* bufp;
	size_t bytestowrite;
	ssize_t byteswritten;
	size_t totalbytes;

	for (bufp = (char*)buf, bytestowrite = size, totalbytes = 0;

		bytestowrite > 0; bufp += byteswritten, bytestowrite -= byteswritten)
	{
		byteswritten = write(fd, bufp, bytestowrite);
		if ((byteswritten) == -1 && (errno != EINTR))
			return -1;

		if (byteswritten == -1)
			byteswritten = 0;

		totalbytes += byteswritten;
	}
	return totalbytes;
}

void sigquit_handler(int sig) {
	assert(sig == SIGQUIT);
	pid_t self = getpid();
	if (parent_pid != self) _exit(0);
}

int main(int argc, char** argv)
{
	// Register signal handler
	signal(SIGQUIT, sigquit_handler);
	parent_pid = getpid();

	// Create channels for communication
	int pipes[2][2];
	for (int i = 0; i < 2; i++) {
		pipe(pipes[i]);
	}

	// Setup I2C comms and devices
	if (wiringPiSetupGpio() == -1)
		exit(1);

	int fd = wiringPiI2CSetup(ARDUINO_ADDRESS);
	if (fd == -1) {
		std::cout << "Failed to init I2C communication.\n";
		return -1;
	}

	// Parent process is image recognition, child is PID and servo controller
	pid_t pid = fork();

	// Parent process
	if (pid > 0) {
		
		// user hyperparams
		float recheckChance = 0.1;
		bool useTracking = true;
		bool draw = true;
		bool showVideo = true;
		std::string target = "face";

		// program state variables
		bool rechecked = false;
		bool isTracking = false;
		bool isSearching = false;

		// Create object tracker to optimize detection performance
		cv::Rect2d roi;
		Ptr<Tracker> tracker = cv::TrackerCSRT::create();

		// Object center coordinates
		int frameCenterX = 0;
		int frameCenterY = 0;

		// Object coordinates
		int objX = 0;
		int objY = 0;

		float f;
		float FPS[16];
		int i, Fcnt = 0;
		vector<string> class_names = {
			"background",
			"aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair",
			"cow", "diningtable", "dog", "horse",
			"motorbike", "person", "pottedplant",
			"sheep", "sofa", "train", "tvmonitor"
		};

		cv::Mat frame, tmp, gray, smallImg;
		cv::Mat detection;
		chrono::steady_clock::time_point Tbegin, Tend;

		cv::VideoCapture camera(0);
		if (!camera.isOpened())
		{
			cout << "Cannot open the camera!" << endl;
			exit(-1);
		}
		
		std::string prototextFile = "/MobileNetSSD_deploy.prototxt";
		std::string modelFile = "/MobileNetSSD_deploy.caffemodel";
		std::string path = get_current_dir_name();
		std::string prototextFilePath = path + prototextFile;
		std::string modelFilePath = path + modelFile;
		dnn::Net net;
		if (fileExists(modelFilePath) && fileExists(prototextFilePath)) {
			net = dnn::readNetFromCaffe(prototextFilePath, modelFilePath);
			if (net.empty()) {
				std::cout << "Error initializing caffe model" << std::endl;
				exit(-1);
			}
		}
		else {
			std::cout << "Error finding model and prototext files" << std::endl;
			exit(-1);
		}

		//HM::CaffeDetector cd(net, class_names);
		HM::CascadeDetector cd;
		HM::DetectionData result;

		while (waitKey(1) < 0) {

			if (isSearching) {
				isSearching = false;

				// TODO:: Perform search reutine
			}

			Tbegin = chrono::steady_clock::now();

			try
			{
				frame = GetImageFromCamera(camera);

				if (frame.empty())
				{
					std::cout << "Issue reading frame!" << std::endl;
					continue;
				}

				// Convert to Gray Scale and resize
				double fx = 1 / 1.0;
				cv:resize(frame, smallImg, cv::Size(frame.cols / 2,  frame.rows / 2), fx, fx, cv::INTER_LINEAR);
				
				std::cout << "Here we are" << std::endl;
				if (!useTracking) {
					goto detect;
				}
				
				cv::cvtColor(smallImg, gray, cv::COLOR_BGR2GRAY);
				cv::equalizeHist(gray, gray);

				if (isTracking) {
					
					std::cout << "Tracking Target..." << std::endl;
					
					// Get the new tacking result
					if (!tracker->update(gray, roi)) {
						isTracking = false;
						goto detect;
					}

					// Chance to revalidate object tracking quality
					if (recheckChance >= static_cast<float>(rand()) / static_cast <float> (RAND_MAX)) {
						result = cd.detect(gray, target, draw);
						std::cout << "Rechecking quality..." << std::endl;
						if (!result.found) { 
							rechecked = true;
							goto detect;
						}
					}

					// Determine object and frame centers
					frameCenterX = static_cast<int>(smallImg.cols / 2);
					frameCenterY = static_cast<int>(smallImg.rows / 2);
					objX = roi.x + roi.width * 0.5;
					objY = roi.y + roi.height * 0.5;

					// Determin error
					double tlt_error = frameCenterY - objY;
					double pan_error = frameCenterX - objX;

					// Send to child process
					write(pipes[0][1], &tlt_error, sizeof(tlt_error));
					write(pipes[1][1], &pan_error, sizeof(pan_error));

					// draw to frame
					if (draw) {
						cv::Scalar color = cv::Scalar(255);
						cv::Rect rec(
							roi.x,
							roi.y,
							roi.width,
							roi.height
						);
						circle(
							smallImg,
							cv::Point(objX, objY),
							(int)(roi.width + roi.height) / 2 / 10,
							color, 2, 8, 0);
						rectangle(smallImg, rec, color, 2, 8, 0);
						putText(
							smallImg,
							target,
							cv::Point(roi.x, roi.y - 5),
							cv::FONT_HERSHEY_SIMPLEX,
							1.0,
							color, 2, 8, 0);
					}
				}
				else {
					
					detect: 
					if (!rechecked) {
						result = cd.detect(gray, target, draw);
						rechecked = false;
					}

					if (useTracking) {
						std::cout << "Lost Target!!" << std::endl;
					}
					
					if (result.found) {

						// Determine object and frame centers
						frameCenterX = static_cast<int>(gray.cols / 2);
						frameCenterY = static_cast<int>(gray.rows / 2);
						objX = result.targetCenterX;
						objY = result.targetCenterY;

						double tlt_error = frameCenterY - objY;
						double pan_error = frameCenterX - objX;

						write(pipes[0][1], &tlt_error, sizeof(tlt_error));
						write(pipes[1][1], &pan_error, sizeof(pan_error));
						
						
						if (useTracking) {

							roi.x = result.boundingBox.x;
							roi.y = result.boundingBox.y;
							roi.width = result.boundingBox.width;
							roi.height = result.boundingBox.height;

							if (tracker->init(gray, roi)) {
								isTracking = true;
							}
						}
					}
					else {
						isSearching = true;
					}
				}				
				
				if (showVideo) {
					if (draw) {
						Tend = chrono::steady_clock::now();
						f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
						if (f > 0.0) FPS[((Fcnt++) & 0x0F)] = 1000.0 / f;
						for (f = 0.0, i = 0; i < 16; i++) { f += FPS[i]; }
						putText(smallImg, format("FPS %0.2f", f / 16), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255));
					}
		
					imshow("frame", smallImg);
				}
			}
			catch (const std::exception&)
			{
				std::cout << "Issue detecting target from video" << std::endl;
				exit(-1);
			}
		}

		// Terminate child processes and cleanup windows
		if (showVideo) {
			cv::destroyAllWindows();
		}

		kill(-parent_pid, SIGQUIT);
		if (wait(NULL) != -1) {
			return 0;
		}
		else {
			return -1;
		}
	}

	// Child process
	else {

		prctl(PR_SET_PDEATHSIG, SIGKILL); // Kill child when parent dies
		sendComand(0x8, 0x0, fd);

		PID pan = PID(0.05, 0.04, 0.001, -75.0, 75.0);
		PID tilt = PID(0.05, 0.04, 0.001, -75.0, 75.0);
		pan.init();
		tilt.init();

		// Keep track of position information
		double tlt_error;
		double pan_error;
		int angleX;
		int angleY;
		int currentAngleX = 90;
		int currentAngleY = 90;

		while (true) {

			if (read(pipes[0][0], &tlt_error, sizeof(tlt_error)) == sizeof(tlt_error)) {
				if (read(pipes[1][0], &pan_error, sizeof(pan_error)) == sizeof(pan_error)) {
					
					// Calc new angle and update servos
					angleX = static_cast<int>(pan.update(pan_error, 0));
					angleY = static_cast<int>(tilt.update(tlt_error, 0)) * -1;

					std::cout << "Error X: ";
					std::cout << pan_error;
					std::cout << ", Error Y: ";
					std::cout << tlt_error;

					std::cout << "; X: ";
					std::cout << angleX;
					std::cout << ", Y: ";
					std::cout << angleY << std::endl;

					if (currentAngleX != angleX) {
						int mappedX = mapOutput(angleX, -90, 90, 0, 180);
						sendComand(0x2, static_cast<unsigned char>(mappedX), fd);
						currentAngleX = angleX;
					}

					if (currentAngleY != angleY) {
						int mappedY = mapOutput(angleY, -90, 90, 0, 180);
						sendComand(0x3, static_cast<unsigned char>(mappedY), fd);
						currentAngleY = angleY;
					}
					
					delay(250);
				}
			}
		}	
	}
}