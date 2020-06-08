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
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h> 
#include <sys/prctl.h>
#include <signal.h>
#include <ctime>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <malloc.h>
#include <sys/mman.h>
#include <errno.h>

// 3rd party Dynamic libs
#include <wiringPi.h>
#include <wiringPiI2C.h>
#include <softPwm.h>  /* include header file for software PWM */

// OpenCV imports
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv4/opencv2/dnn/dnn.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/video/video.hpp"
#include <opencv2/tracking.hpp>

// Local classes
#include "CaffeDetector.h"
#include "CascadeDetector.h"
#include "PID.h"

#define ARDUINO_ADDRESS 0x9

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


// Last signal caught
volatile sig_atomic_t sig_value1;

static void usr_sig_handler1(const int sig_number, siginfo_t* sig_info, void* context)
{
	// Take care of all segfaults
	if (sig_number == SIGSEGV)
	{
		perror("Address access error");
		exit(-1);
	}

	sig_value1 = sig_number;
}

void sigquit_handler(int sig) {
	assert(sig == SIGQUIT);
	pid_t self = getpid();
	if (parent_pid != self) _exit(0);
}

typedef struct parameter {
	PID* pan;
	PID* tilt;
	int* ShmPTR;
	int rate; // Updates per second
	int fd;
} param;

int msleep(long msec)
{
	struct timespec ts;
	int res;

	if (msec < 0)
	{
		errno = EINVAL;
		return -1;
	}

	ts.tv_sec = msec / 1000;
	ts.tv_nsec = (msec % 1000) * 1000000;

	do {
		res = nanosleep(&ts, &ts);
	} while (res && errno == EINTR);

	return res;
}

void* tiltThread(void* args) {

	param* parameters = (param*)args;

	PID* tilt = parameters->tilt;
	int* ShmPTR = parameters->ShmPTR;
	int milis = 1000 / parameters->rate;
	int fd = parameters->fd;

	tilt->init();
	int angleY;
	int currentAngleY = 90;


	while (true) {

		if (ShmPTR[4]) { // If the target is locked on
			if (ShmPTR[2] && ShmPTR[5] != 1) { // If we are ready to read
				angleY = static_cast<int>(tilt->update(static_cast<double>(ShmPTR[0]), 0)) * -1;
				ShmPTR[5] = 1;
			}
			else {
				continue;
			}

			std::cout << "Y: ";
			std::cout << angleY << endl;

			if (currentAngleY != angleY) {

				int mappedY = mapOutput(angleY, -90, 90, 0, 180);
				sendComand(0x3, static_cast<unsigned char>(mappedY), fd);
				currentAngleY = angleY;
			}
		}
		else {
			tilt->init();
		}
		
		msleep(milis);
	}
	
	return NULL;
}

void* panThread(void* args) {
	param* parameters = (param*)args;

	PID* pan = parameters->pan;
	int* ShmPTR = parameters->ShmPTR;
	int milis = 1000 / parameters->rate;
	int fd = parameters->fd;

	pan->init();
	int angleX;
	int currentAngleX = 90;

	while (true) {

		if (ShmPTR[4]) { // If the target is locked on
			if (ShmPTR[3] && !ShmPTR[6]) { // If the we are ready to read and not already have read it
				angleX = static_cast<int>(pan->update(static_cast<double>(ShmPTR[1]), 0));
				ShmPTR[6] = 1;
			}
			else {
				continue;
			}
		}
		else {
			pan->init();
		}
	
		std::cout << "X: "; 
		std::cout << angleX << endl;

		if (currentAngleX != angleX) {
			int mappedX = mapOutput(angleX, -90, 90, 0, 180);
			sendComand(0x2, static_cast<unsigned char>(mappedX), fd);
			currentAngleX = angleX;
		}

		msleep(milis);
	}
	
	return NULL;
}

int main(int argc, char** argv)
{
	// Register signal handler
	signal(SIGQUIT, sigquit_handler);
	parent_pid = getpid();

	// Setup I2C comms and devices
	if (wiringPiSetupGpio() == -1)
		exit(1);

	int fd = wiringPiI2CSetup(ARDUINO_ADDRESS);
	if (fd == -1) {
		std::cout << "Failed to init I2C communication.\n";
		return -1;
	}

	// Init shared memory
	int ShmID;
	int* ShmPTR;
	int status;

	ShmID = shmget(IPC_PRIVATE, 5 * sizeof(int), IPC_CREAT | 0666);

	if (ShmID < 0) {
		throw "Could not initialize shared memory";
	}

	ShmPTR = (int*)shmat(ShmID, NULL, 0);

	if ((int)ShmPTR == -1) {
		throw "Could not initialize shared memory";
	}

	// These values allow for async communication between parent process and child process's threads
	// These garentee, in a non-blocking way, that the child process will never read a value twice
	// But the childs threads may be reading old values at any given time.
	ShmPTR[0] = 0; // Tilt error
	ShmPTR[1] = 0; // Pan error
	ShmPTR[2] = 1; // Tilt lock
	ShmPTR[3] = 1; // Pan lock
	ShmPTR[4] = 1; // Reset signal
	ShmPTR[5] = 0; // Tilt read
	ShmPTR[6] = 0; // Pan read

	// Parent process is image recognition, child is PID and servo controller
	pid_t pid = fork();

	// Parent process
	if (pid > 0) {

		// user hyperparams
		float recheckChance = 0.01;
		bool useTracking = true;
		bool draw = true;
		bool showVideo = true;
		std::string target = "person";

		// program state variables
		bool rechecked = false;
		bool isTracking = false;
		bool isSearching = false;
		int lossCount = 0;
		int lossCountMax = 100;

		// Create object tracker to optimize detection performance
		cv::Rect2d roi;
		Ptr<Tracker> tracker = cv::TrackerCSRT::create();
		// Ptr<Tracker> tracker = cv::TrackerMOSSE::create();
		// Ptr<Tracker> tracker = cv::TrackerGOTURN::create();

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

		cv::Mat frame;
		cv::Mat detection;
		chrono::steady_clock::time_point Tbegin, Tend;

		cv::VideoCapture camera(0);
		if (!camera.isOpened())
		{
			throw "cannot initialize camera";
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
				throw "Error initializing caffe model";
			}
		}
		else {
			throw "Error finding model and prototext files";
		}

		// HM::CaffeDetector cd(net, class_names);
		HM::CascadeDetector cd;
		HM::DetectionData result;

		while (true) {

			if (isSearching) {
				isSearching = false;

				// TODO:: Perform search reutine
				// sendComand(0x8, 0x0, fd); // Reset servos
			}

			if (draw) {
				Tbegin = chrono::steady_clock::now();
			}

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
				cv::flip(frame, frame, -1);
				// smallImg = frame;
				// cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
				// cv::equalizeHist(gray, gray);
				// resize(frame, frame, cv::Size(frame.cols / 2,  frame.rows / 2), fx, fx, cv::INTER_LINEAR);

				if (!useTracking) {
					goto detect;
				}

				if (isTracking) {

					std::cout << "Tracking Target..." << std::endl;

					// Get the new tracking result
					if (!tracker->update(frame, roi)) {
						isTracking = false;
						std::cout << "Lost target!!" << std::endl;
						lossCount++;
						goto detect;
					}

					// Chance to revalidate object tracking quality
					if (recheckChance >= static_cast<float>(rand()) / static_cast <float> (RAND_MAX)) {
						goto detect;
					}

				validated:

					// Determine object and frame centers
					frameCenterX = static_cast<int>(frame.cols / 2);
					frameCenterY = static_cast<int>(frame.rows / 2);
					objX = roi.x + roi.width * 0.5;
					objY = roi.y + roi.height * 0.5;

					// Inform child process's threads of old data
					ShmPTR[2] = 0;
					ShmPTR[3] = 0;

					// Determine error
					ShmPTR[0] = frameCenterY - objY;
					ShmPTR[1] = frameCenterX - objX;

					// Reset read flags
					ShmPTR[5] = 0;
					ShmPTR[6] = 0;

					// Fresh data, now the child process's threads can read
					ShmPTR[2] = 1;
					ShmPTR[3] = 1;

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
							frame,
							cv::Point(objX, objY),
							(int)(roi.width + roi.height) / 2 / 10,
							color, 2, 8, 0);
						rectangle(frame, rec, color, 2, 8, 0);
						putText(
							frame,
							target,
							cv::Point(roi.x, roi.y - 5),
							cv::FONT_HERSHEY_SIMPLEX,
							1.0,
							color, 2, 8, 0);
					}
				}
				else {

				detect:
					result = cd.detect(frame, target, draw);

					if (result.found) {

						if (rechecked) {
							goto validated;
						}

						// Determine object and frame centers
						frameCenterX = static_cast<int>(frame.cols / 2);
						frameCenterY = static_cast<int>(frame.rows / 2);
						objX = result.targetCenterX;
						objY = result.targetCenterY;

						// Inform child process's threads of old data
						ShmPTR[2] = 0;
						ShmPTR[3] = 0;

						// Determine error
						ShmPTR[0] = frameCenterY - objY;
						ShmPTR[1] = frameCenterX - objX;

						// Reset read flags
						ShmPTR[5] = 0;
						ShmPTR[6] = 0;

						// Fresh data, now the child process's threads can read
						ShmPTR[2] = 1;
						ShmPTR[3] = 1;
			
						if (useTracking && !isTracking) {

							roi.x = result.boundingBox.x;
							roi.y = result.boundingBox.y;
							roi.width = result.boundingBox.width;
							roi.height = result.boundingBox.height;

							if (tracker->init(frame, roi)) {
								isTracking = true;
								std::cout << "initialized!!" << std::endl;
								ShmPTR[3];
								kill(pid, SIGUSR1);
							}
						}
					}
					else {
						lossCount++;

						if (lossCount >= lossCountMax) {
							std::cout << "No target found" << std::endl;
							isSearching = true;
							isTracking = false;
							lossCount = 0;
						}
					}
				}

				if (showVideo) {
					if (draw) {
						Tend = chrono::steady_clock::now();
						f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
						if (f > 0.0) FPS[((Fcnt++) & 0x0F)] = 1000.0 / f;
						for (f = 0.0, i = 0; i < 16; i++) { f += FPS[i]; }
						putText(frame, format("FPS %0.2f", f / 16), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255));
					}

					cv::imshow("Viewport", frame);
					waitKey(1);
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
		sendComand(0x8, 0x0, fd); // Reset servos

		// Setup threads
		PID pan = PID(0.05, 0.04, 0.001, -75.0, 75.0);
		PID tilt = PID(0.05, 0.04, 0.001, -75.0, 75.0);

		param* parameters = (param*)malloc(sizeof(param*));

		parameters->fd = fd;
		parameters->ShmPTR = ShmPTR;
		parameters->pan = &pan;
		parameters->tilt = &tilt;
		parameters->rate = 30; /* Updates per second */

		pthread_t panTid, tiltTid;
		pthread_create(&panTid, NULL, panThread, (void*)parameters);
		pthread_create(&tiltTid, NULL, tiltThread, (void*)parameters);
		pthread_detach(panTid);
		pthread_detach(tiltTid);
	
		struct sigaction sig_action;

		sigset_t oldmask;
		sigset_t newmask;
		sigset_t zeromask;

		memset(&sig_action, 0, sizeof(struct sigaction));

		sig_action.sa_flags = SA_SIGINFO;
		sig_action.sa_sigaction = usr_sig_handler1;

		sigaction(SIGHUP, &sig_action, NULL);
		sigaction(SIGINT, &sig_action, NULL);
		sigaction(SIGTERM, &sig_action, NULL);
		sigaction(SIGSEGV, &sig_action, NULL);
		sigaction(SIGUSR1, &sig_action, NULL);

		sigemptyset(&newmask);
		sigaddset(&newmask, SIGHUP);
		sigaddset(&newmask, SIGINT);
		sigaddset(&newmask, SIGTERM);
		sigaddset(&newmask, SIGSEGV);
		sigaddset(&newmask, SIGUSR1);

		sigprocmask(SIG_BLOCK, &newmask, &oldmask);
		sigemptyset(&zeromask);
		sig_value1 = 0;

		while ((sig_value1 != SIGINT) && (sig_value1 != SIGTERM))
		{
			sig_value1 = 0;

			// Sleep until signal is caught
			sigsuspend(&zeromask);

			if (sig_value1 == SIGUSR1) {
				std::cout << "Tracking reinitialized, resetting pid's..." << std::endl;
			}
		}
	}
}