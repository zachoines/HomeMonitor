// C++ libs
#include <iostream>
#include <string>
#include <vector>
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
#include <sys/ipc.h>
#include <sys/shm.h>
#include <malloc.h>
#include <sys/mman.h>
#include <errno.h>

// 3rd party libs
#include <wiringPi.h>
#include <wiringPiI2C.h>
#include <torch/torch.h>
//#include <boost/interprocess/managed_shared_memory.hpp>
//#include <boost/interprocess/containers/vector.hpp>
//#include <boost/interprocess/allocators/allocator.hpp>
#include <vector>
#include <algorithm>

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

// Local classes and files
#include "CaffeDetector.h"
#include "CascadeDetector.h"
#include "PID.h"
#include "util.h"
#include "data.h"
#include "SACAgent.h"

#define ARDUINO_ADDRESS 0x9

using namespace cv;
using namespace std;
using namespace Utility;

void* tiltThread(void* args);
void* panThread(void* args);
void* detectThread(void* args);
void* autoTuneThread(void* args);
void* nonPIDPanThread(void* args);

ED eventDataArray[2];
pthread_cond_t cond_t = PTHREAD_COND_INITIALIZER;
pthread_mutex_t lock_t = PTHREAD_MUTEX_INITIALIZER;

SACAgent* pidAutoTuner = nullptr;
cv::VideoCapture camera(0);
torch::Device device(torch::kCPU);
Buffer* trainingBuffer;


int main(int argc, char** argv)
{
	auto default_dtype = caffe2::TypeMeta::Make<double>();
	torch::set_default_dtype(default_dtype);
	param* parameters = (param*)malloc(sizeof(param));
	parameters->config = new Config();
	pidAutoTuner = new SACAgent(parameters->config->numInput, parameters->config->numHidden, parameters->config->numActions, 1.0, 0.0);
	trainingBuffer = new Buffer();
	

	// Setup Torch
	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available! Training on GPU." << std::endl;
		device = torch::kCUDA;
	}
	else {
		std::cout << "CUDA is not available! Training on CPU." << std::endl;
	}

	// Setup I2C comms and devices
	if (wiringPiSetupGpio() == -1)
		exit(1);

	int fd = wiringPiI2CSetup(ARDUINO_ADDRESS);
	if (fd == -1) {
		throw "Failed to init I2C communication.";
	}

	if (!camera.isOpened())
	{
		throw "cannot initialize camera";
	}

	cv::Mat test = GetImageFromCamera(camera);

	if (test.empty())
	{
		throw "Issue reading frame!";
	}

	int height = test.rows;
	int width = test.cols;

	// Convert to Gray Scale and resize
	double fx = 1 / 1.0;
	cv::flip(test, test, -1);

	parameters->height = height;
	parameters->width = width;

	sendComand(0x8, 0x0, fd); // Reset servos

	// Setup shared thread parameters
	PID* pan = new PID(0.05, 0.04, 0.001, -75.0, 75.0);
	PID* tilt = new PID(0.05, 0.04, 0.001, -75.0, 75.0);

	parameters->fd = fd;
	parameters->ShmPTR = eventDataArray;
	parameters->pan = pan;
	parameters->tilt = tilt;
	parameters->rate = 6; /* Updates per second */
	parameters->mutex = PTHREAD_MUTEX_INITIALIZER;
	parameters->isTraining = false;

	pthread_t panTid, tiltTid, detectTid, autoTuneTid, nonPIDPanTID;
	pthread_create(&nonPIDPanTID, NULL, nonPIDPanThread, (void*)parameters);
	// pthread_create(&panTid, NULL, panThread, (void*)parameters);
	// pthread_create(&tiltTid, NULL, tiltThread, (void*)parameters);
	pthread_create(&detectTid, NULL, detectThread, (void*)parameters);
	pthread_create(&autoTuneTid, NULL, autoTuneThread, (void*)parameters);
	// pthread_join(panTid, NULL);
	// pthread_join(tiltTid, NULL);nonPIDPanTID
	pthread_join(nonPIDPanTID, NULL);
	pthread_join(detectTid, NULL);
	pthread_join(autoTuneTid, NULL);


	// Terminate child processes and cleanup windows
	cv::destroyAllWindows();
}

void* nonPIDPanThread(void* args) {
	auto options = torch::TensorOptions().dtype(torch::kDouble).device(device);
	param* parameters = (param*)args;

	// PID* pan = parameters->pan;
	ED* ShmPTR = parameters->ShmPTR;
	int milis = 1000 / parameters->rate;
	int fd = parameters->fd;
	int angleX = 90;
	int currentAngleX = 90;
	int lastAngleX = 90;
	bool programStart = true;
	double lastTimeStep;
	SD lastState;
	ED lastPanData;

	while (true) {
		SD currentState;
		TD trainData;
		ED panData = ShmPTR[1];

		if (!panData.dirty && !panData.isOld(lastTimeStep)) { // If its not old and not already read

			lastTimeStep = panData.timestamp;

			// State data timestep (n - 1)
			currentState.objCenterOld = lastPanData.Obj;
			currentState.frameCenterOld = lastPanData.Frame;
			currentState.errorOld = lastPanData.error;
			currentState.lastAngle = static_cast<double>(lastAngleX);

			// State data timestep (n)
			currentState.objCenter = panData.Obj;
			currentState.frameCenter = panData.Frame;
			currentState.error = panData.error;
			currentState.currentAngle = static_cast<double>(currentAngleX);

			double stateArray[parameters->config->numInput];
			currentState.getStateArray(stateArray);

			// Normalize and get new PID gains
			torch::Tensor action = pidAutoTuner->get_action(torch::from_blob(stateArray, { 1, parameters->config->numInput }, options));

			double newAngle = Utility::rescaleAction(action.data().item().toDouble(), 0.0, 180.0);

			std::cout << "New pan angle: " << newAngle << std::endl;

			trainData.done = panData.done;
			trainData.reward = errorToReward(currentState.error, currentState.errorOld, parameters->width / 2, panData.done);
			trainData.actions[0] = newAngle;

			angleX = static_cast<int>(newAngle);
			if (currentAngleX != angleX) {
				sendComand(0x2, static_cast<unsigned char>(newAngle), fd);
			}

			lastAngleX = currentAngleX;
			currentAngleX = angleX;

			if (programStart) { // For when we dont have a lastState
				programStart = false;
				lastState = currentState;
				lastPanData = panData;
				goto sleep;
			}
			else {
				trainData.nextState = currentState;
				trainData.currentState = lastState;
				lastState = currentState;
				lastPanData = panData;
			}
		}
		else {
			// pan->update(0.0, 0);
			goto sleep;
		}

		if (!parameters->isTraining) {

			if (pthread_mutex_trylock(&lock_t) == 0) {

				if (!parameters->isTraining) {

					try {
						if (trainingBuffer->size() == parameters->config->maxBufferSize) {
							std::cout << "Sending a training request..." << std::endl;
							parameters->isTraining = true;
							pthread_cond_broadcast(&cond_t);
						}
						else {
							trainingBuffer->push_back(trainData);
						}
					}
					catch (...)
					{
						throw "Error in pan thread";
					}
				}

				pthread_mutex_unlock(&lock_t);
			}

			pthread_mutex_unlock(&parameters->mutex);
		}

	sleep:
		msleep(milis);
	}

	return NULL;
}

void* panThread(void* args) {
	auto options = torch::TensorOptions().dtype(torch::kDouble).device(device);
	param* parameters = (param*)args;

	PID* pan = parameters->pan;
	ED* ShmPTR = parameters->ShmPTR;
	int milis = 1000 / parameters->rate;
	int fd = parameters->fd;
	int angleX = 90;
	int currentAngleX = 90;
	int lastAngleX = 90;
	bool programStart = true;
	double lastTimeStep;
	SD lastState;
	ED lastPanData;

	pan->init();


	while (true) {
		SD currentState;
		TD trainData;
		ED panData = ShmPTR[1];

		if (!panData.dirty && !panData.isOld(lastTimeStep)) { // If its not old and not already read

			lastTimeStep = panData.timestamp;

			// State data timestep (n - 1)
			currentState.objCenterOld = lastPanData.Obj;
			currentState.frameCenterOld = lastPanData.Frame;
			currentState.errorOld = lastPanData.error;
			currentState.lastAngle = static_cast<double>(lastAngleX);

			// State data timestep (n)
			currentState.objCenter = panData.Obj;
			currentState.frameCenter = panData.Frame;
			currentState.error = panData.error;
			currentState.currentAngle = static_cast<double>(currentAngleX);

			double stateArray[parameters->config->numInput];
			currentState.getStateArray(stateArray);

			// Normalize and get new PID gains
			torch::Tensor actions = pidAutoTuner->get_action(torch::from_blob(stateArray, { 1, parameters->config->numInput }, options));
			auto actions_a = actions.accessor<double, 1>();
			double newGains[3];


			for (int i = 0; i < 3; i++) {
				// newGains[i] = actions_a[i];
				newGains[i] = Utility::rescaleAction(actions_a[i], 0.0, .1);
			}

			std::cout << "Here are the gains for pan: " << actions_a[0] << actions_a[1] << actions_a[2] << std::endl;
			std::cout << "Here are the rescaled Gains for pan: " << newGains[0] << newGains[1] << newGains[2] << std::endl;

			pan->setWeights(newGains[0], newGains[1], newGains[2]);

			trainData.done = panData.done;
			trainData.reward = errorToReward(currentState.error, currentState.errorOld, parameters->width / 2, panData.done );
			pan->getWeights(trainData.actions);

			angleX = static_cast<int>(pan->update(panData.error, 0));


			if (currentAngleX != angleX) {
				int mappedX = mapOutput(angleX, -90, 90, 0, 180);
				sendComand(0x2, static_cast<unsigned char>(mappedX), fd);
			}

			lastAngleX = currentAngleX;
			currentAngleX = angleX;

			if (programStart) { // For when we dont have a lastState
				programStart = false;
				lastState = currentState;
				lastPanData = panData;
				goto sleep;
			}
			else {
				trainData.nextState = currentState;
				trainData.currentState = lastState;
				lastState = currentState;
				lastPanData = panData;
			}

			if (trainData.done) {
				pan->init();
			}
		}
		else {
			// pan->update(0.0, 0);
			goto sleep;
		}

		if (!parameters->isTraining) {

			if (pthread_mutex_trylock(&lock_t) == 0) {

				if (!parameters->isTraining) {

					try {
						if (trainingBuffer->size() == parameters->config->maxBufferSize) {
							std::cout << "Sending a training request..." << std::endl;
							parameters->isTraining = true;
							pthread_cond_broadcast(&cond_t);
						}
						else {
							trainingBuffer->push_back(trainData);
						}
					}
					catch (...)
					{
						throw "Error in pan thread";
					}
				}

				pthread_mutex_unlock(&lock_t);
			}

			pthread_mutex_unlock(&parameters->mutex);
		}

	sleep:
		msleep(milis);
	}

	return NULL;
}

void* tiltThread(void* args) {
	auto options = torch::TensorOptions().dtype(torch::kDouble).device(device);

	param* parameters = (param*)args;

	PID* tilt = parameters->tilt;
	ED* ShmPTR = parameters->ShmPTR;
	int milis = 1000 / parameters->rate;
	int fd = parameters->fd;
	bool programStart = true;
	int angleY;
	int currentAngleY = -90;
	int lastAngleY = -90;
	double lastTimeStep;
	SD lastState;
	ED lastTiltData;

	tilt->init();

	while (true) {
		SD currentState;
		TD trainData;
		ED tiltData = ShmPTR[0];

		if (!tiltData.dirty && !tiltData.isOld(lastTimeStep)) { // If its not old and not already read

			lastTimeStep = tiltData.timestamp;

			// State data timestep (n - 1)
			currentState.objCenterOld = lastTiltData.Obj;
			currentState.frameCenterOld = lastTiltData.Frame;
			currentState.errorOld = lastTiltData.error;
			currentState.lastAngle = static_cast<double>(lastAngleY);

			// State data timestep (n)
			currentState.objCenter = tiltData.Obj;
			currentState.frameCenter = tiltData.Frame;
			currentState.error = tiltData.error;
			currentState.currentAngle = static_cast<double>(currentAngleY);

			// Normalize and get new PID gains
			double stateArray[parameters->config->numInput];
			currentState.getStateArray(stateArray);

			torch::Tensor actions = pidAutoTuner->get_action(torch::from_blob(stateArray, { 1, parameters->config->numInput }, options));
			auto actions_a = actions.accessor<double, 1>();
			double newGains[3];


			for (int i = 0; i < 3; i++) {
				//newGains[i] = actions_a[i];
				newGains[i] = Utility::rescaleAction(actions_a[i], 0.0, .1);
			}

			std::cout << "Here are the gains for pan: " << actions_a[0] << actions_a[1] << actions_a[2] << std::endl;
			std::cout << "Here are the rescaled Gains for pan: " << newGains[0] << newGains[1] << newGains[2] << std::endl;

			tilt->setWeights(newGains[0], newGains[1], newGains[2]);

			trainData.done = tiltData.done;
			trainData.reward = errorToReward(currentState.error, currentState.errorOld, parameters->height / 2, tiltData.done );
			tilt->getWeights(trainData.actions);

			angleY = static_cast<int>(tilt->update(tiltData.error, 0)) * -1;


			if (currentAngleY != angleY) {
				int mappedX = mapOutput(angleY, -90, 90, 0, 180);
				sendComand(0x2, static_cast<unsigned char>(mappedX), fd);
			}

			lastAngleY = currentAngleY;
			currentAngleY = angleY;

			if (programStart) { // For when we dont have a lastState
				programStart = false;
				lastState = currentState;
				lastTiltData = tiltData;
				goto sleep;
			}
			else {
				trainData.nextState = currentState;
				trainData.currentState = lastState;
				lastState = currentState;
				lastTiltData = tiltData;
			}

			if (trainData.done) {
				tilt->init();
			}
		}
		else {
			// tilt->update(0.0, 0);
			goto sleep;
		}

		if (!parameters->isTraining) {

			if (pthread_mutex_trylock(&lock_t) == 0) {

				if (!parameters->isTraining) {

					try {
						if (trainingBuffer->size() == parameters->config->maxBufferSize) {
							std::cout << "Sending a training request..." << std::endl;
							parameters->isTraining = true;
							pthread_cond_broadcast(&cond_t);
						}
						else {
							trainingBuffer->push_back(trainData);
						}
					}
					catch (...)
					{
						throw "Error in tilt thread";
					}
				}

				pthread_mutex_unlock(&lock_t);
			}

			pthread_mutex_unlock(&lock_t);
		}

	sleep:
		msleep(milis);
	}

	return NULL;
}

void* detectThread(void* args)
{
	
	param* parameters = (param*)args;
	int fd = parameters->fd;

	// user hyperparams
	float recheckChance = 0.01;
	bool useTracking = true;
	bool draw = false;
	bool showVideo = false;
	bool cascadeDetector = true;
	std::string target = "sheep";

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
	auto execbegin = std::chrono::high_resolution_clock::now();

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

	if (!cascadeDetector) {
		if (fileExists(modelFilePath) && fileExists(prototextFilePath)) {
			net = dnn::readNetFromCaffe(prototextFilePath, modelFilePath);
			if (net.empty()) {
				throw "Error initializing caffe model";
			}
		}
		else {
			throw "Error finding model and prototext files";
		}
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
			// cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
			// cv::equalizeHist(gray, gray);
			// resize(frame, frame, cv::Size(frame.cols / 2,  frame.rows / 2), fx, fx, cv::INTER_LINEAR);

			if (!useTracking) {
				goto detect;
			}

			if (isTracking) {

				// Get the new tracking result
				if (!tracker->update(frame, roi)) {
					isTracking = false;
					std::cout << "Lost target!!" << std::endl;
					lossCount++;
					goto detect;
				}

				// Chance to revalidate object tracking quality
				if (recheckChance >= static_cast<float>(rand()) / static_cast <float> (RAND_MAX)) {
					std::cout << "Rechecking tracking quality..." << std::endl;
					goto detect;
				}

			validated:
				ED tilt;
				ED pan;
				// Determine object and frame centers
				frameCenterX = static_cast<int>(frame.cols / 2);
				frameCenterY = static_cast<int>(frame.rows / 2);
				objX = roi.x + roi.width * 0.5;
				objY = roi.y + roi.height * 0.5;

				// Inform child process's threads of old data (race condition here, kinda washes out in end)
				eventDataArray[0].dirty = true;
				eventDataArray[1].dirty = true;

				// Determine error
				tilt.error = static_cast<double>(frameCenterY - objY);
				pan.error = static_cast<double>(frameCenterX - objX);

				// Enter State data
				double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - execbegin).count() * 1e-9;
				pan.timestamp = elapsed;
				tilt.timestamp = elapsed;
				pan.Obj = objX;
				tilt.Obj = objY;
				pan.Frame = frameCenterY;
				tilt.Frame = frameCenterX;
				pan.done = false;
				tilt.done = false;

				// Fresh data, now the child process's threads can read
				eventDataArray[0] = tilt;
				eventDataArray[1] = pan;

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

					ED tilt;
					ED pan;
					// Determine object and frame centers
					frameCenterX = static_cast<int>(frame.cols / 2);
					frameCenterY = static_cast<int>(frame.rows / 2);
					objX = roi.x + roi.width * 0.5;
					objY = roi.y + roi.height * 0.5;

					// Inform child process's threads of old data (race condition here, kinda washes out in end)
					eventDataArray[0].dirty = true;
					eventDataArray[1].dirty = true;

					// Determine error
					tilt.error = static_cast<double>(frameCenterY - objY);
					pan.error = static_cast<double>(frameCenterX - objX);

					// Other State data
					double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - execbegin).count() * 1e-9;
					pan.timestamp = elapsed;
					tilt.timestamp = elapsed;
					pan.Obj = objX;
					tilt.Obj = objY;
					pan.Frame = frameCenterY;
					tilt.Frame = frameCenterX;
					pan.done = true;
					tilt.done = true;

					// Fresh data, now the child process's threads can read
					eventDataArray[0] = tilt;
					eventDataArray[1] = pan;

					if (useTracking) {

						roi.x = result.boundingBox.x;
						roi.y = result.boundingBox.y;
						roi.width = result.boundingBox.width;
						roi.height = result.boundingBox.height;

						if (tracker->init(frame, roi)) {
							isTracking = true;
							std::cout << "initialized!!" << std::endl;
							std::cout << "Tracking Target..." << std::endl;
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

	return NULL;
}

void* autoTuneThread(void* args)
{
	param* parameters = (param*)args;

	while (true) {
		pthread_mutex_lock(&lock_t);
		while (!parameters->isTraining) {
			pthread_cond_wait(&cond_t, &lock_t);
		}

		// Perform training session
		for (int i = 0; i < parameters->config->maxTrainingSessions; i++) {
			pidAutoTuner->update(parameters->config->batchSize, trainingBuffer);
		}

		trainingBuffer->clear();
		parameters->isTraining = false;
		pthread_mutex_unlock(&lock_t);
	}

	return NULL;
}
