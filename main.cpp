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
#include "ReplayBuffer.h"

// For Servos
#include <pca9685.h>

#define PIN_BASE 300
#define MAX_PWM 4096
#define HERTZ 50
#define ARDUINO_ADDRESS 0x9

using namespace cv;
using namespace std;
using namespace Utility;

void* panTiltThread(void* args);
void* detectThread(void* args);
void* autoTuneThread(void* args);

ED eventDataArray[2];
pthread_cond_t trainCond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t trainLock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t stateDataLock = PTHREAD_MUTEX_INITIALIZER;

SACAgent* pidAutoTuner = nullptr;
ReplayBuffer* replayBuffer = nullptr;
cv::VideoCapture camera(0);
torch::Device device(torch::kCPU);


Ptr<Tracker> createOpenCVTracker(int type) {
	Ptr<Tracker> tracker;
	switch (type)
	{
	case 0:
		tracker = cv::TrackerCSRT::create();
		break;
	case 1:
		tracker = cv::TrackerMOSSE::create();
		break;
	case 2:
		tracker = cv::TrackerGOTURN::create();
		break;
	default:
		tracker = cv::TrackerCSRT::create();
		break;
	}

	return tracker;
}


int main(int argc, char** argv)
{
	auto default_dtype = caffe2::TypeMeta::Make<double>();
	torch::set_default_dtype(default_dtype);
	param* parameters = (param*)malloc(sizeof(param));
	parameters->config = new Config();
	pidAutoTuner = new SACAgent(parameters->config->numInput, parameters->config->numHidden, parameters->config->numActions, 1.0, 0.0);
	replayBuffer = new ReplayBuffer(parameters->config->maxBufferSize);

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

	int fd;
	
	if (parameters->config->useArduino) {
		fd = wiringPiI2CSetup(ARDUINO_ADDRESS);
		if (fd == -1) {
			throw "Failed to init I2C communication.";
		}
	}
	else {
		// Setup PCA9685 at address 0x40
		fd = pca9685Setup(PIN_BASE, 0x40, HERTZ);
		if (fd < 0)
		{
			throw "Failed to init I2C communication.";
		}

		pca9685PWMReset(fd);
	}

	/* 
	runServo(0, 180.0); // left
	runServo(0, 90.0); // center
	runServo(0, 0.0); // right

	runServo(1, 180.0); // down
	runServo(1, 90.0); // center
	runServo(1, 0.0); // up
	*/

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

	parameters->dims[0] = height;
	parameters->dims[1] = width;

	if (parameters->config->useArduino) {
		sendComand(0x8, 0x0, fd); // Reset servos
	}
	else {
		runServo(0, 90);
		runServo(1, 90);
	}
	

	// Setup shared thread parameters
	PID* pan = new PID(0.05, 0.04, 0.001, -75.0, 75.0);
	PID* tilt = new PID(0.05, 0.04, 0.001, -75.0, 75.0);

	parameters->fd = fd;
	parameters->eventData = eventDataArray;
	parameters->pan = pan;
	parameters->tilt = tilt;
	parameters->rate = parameters->config->updateRate; 
	parameters->isTraining = false;
	parameters->freshData = false;

	pthread_t panTid, tiltTid, detectTid, autoTuneTid, panTiltTid;

	pthread_create(&panTiltTid, NULL, panTiltThread, (void*)parameters);
	pthread_create(&detectTid, NULL, detectThread, (void*)parameters);
	pthread_create(&autoTuneTid, NULL, autoTuneThread, (void*)parameters);

	pthread_join(panTiltTid, NULL);
	pthread_join(detectTid, NULL);
	pthread_join(autoTuneTid, NULL);

	// Terminate child processes and cleanup windows
	cv::destroyAllWindows();
}

void* panTiltThread(void* args) {
	auto options = torch::TensorOptions().dtype(torch::kDouble).device(device);
	param* parameters = (param*)args;

	bool invert[2] = {
		parameters->config->invertY,
		parameters->config->invertX
	};
	
	// PIDs
	PID* pan = parameters->pan;
	PID* tilt = parameters->tilt;

	pan->init();
	tilt->init();

	PID* pids[2] = {
		tilt,
		pan
	};

	// Servo/program state variables
	int errorCounter = -1;
	double previousErrors[2][2][3] = {
		0.0
	};

	int milis = 1000 / parameters->rate;
	int fd = parameters->fd;
	bool programStart = true;
	bool addData = false;

	// I2c commands for arduino
	unsigned char commands[2];
	commands[0] = parameters->config->arduinoCommands[0];
	commands[1] = parameters->config->arduinoCommands[1];

	// Previous state data
	SD lastState[2];
	ED lastData[2];
	double lastTimeStamp[2];
	double newAngles[2] = {
			90.0
	};

	while (true) {

		errorCounter = (errorCounter + 1) % 3;

		SD currentState[2];
		TD trainData[2];
		ED eventData[2];
		

		if (pthread_mutex_lock(&stateDataLock) == 0) { 

			eventData[0] = parameters->eventData[0];
			eventData[1] = parameters->eventData[1];

			pthread_mutex_unlock(&stateDataLock);

			
			for (int i = 1; i < 2; i++) {

				if (eventData[i].timestamp == lastTimeStamp[i]) {
					lastTimeStamp[i] = eventData[i].timestamp;
					goto sleep;
				}
				else if (trainData[i].done) {
					pan->init();
					tilt->init();
					
					lastTimeStamp[i] = eventData[i].timestamp;
				}
				
				
				// Fill out the current state
				lastTimeStamp[i] = eventData[i].timestamp;
				previousErrors[i][0][errorCounter] = eventData[i].timestamp;
				previousErrors[i][1][errorCounter] = (eventData[i].error / (static_cast<double>(parameters->dims[i]) / 2.0));
				
				int k, j;
				double key;
				double key2;
				for (k = 1; k < 3; k++)
				{
					key = previousErrors[i][0][k];
					key2 = previousErrors[i][1][k];
					j = k - 1;

					while (j >= 0 && previousErrors[i][1][j] > key)
					{
						previousErrors[i][1][j + 1] = previousErrors[i][1][j];
						previousErrors[i][0][j + 1] = previousErrors[i][0][j];
						j = j - 1;
					}
					previousErrors[i][0][j + 1] = key;
					previousErrors[i][1][j + 1] = key2;
				}

				double state[3] = {
					0.0
				};

				// Error, e(t) = y′(t) - y(t)
				state[0] = previousErrors[i][1][2];

				// First order error, e(t) - e(t−1)
				state[1] = state[0] - previousErrors[i][1][1];

				// Second order error, e(t) − 2∗e(t−1) + e(t−2)
				state[2] = state[0] - 2.0 * previousErrors[i][1][1] + previousErrors[i][1][0];

				currentState[i].setStateArray(state);


				// If not end of State
				if (!trainData[i].done) {
					double stateArray[parameters->config->numInput];
					currentState[i].getStateArray(stateArray);

					// Normalize and get new PID gains
					torch::Tensor actions = pidAutoTuner->get_action(torch::from_blob(stateArray, { 1, parameters->config->numInput }, options));
					auto actions_a = actions.accessor<double, 1>();
					double scaledActions[parameters->config->numActions];

					for (int a = 0; a < parameters->config->numActions; a++) {
						scaledActions[a] = Utility::rescaleAction(actions_a[a], parameters->config->actionLow, parameters->config->actionHigh);
						trainData[i].actions[a] = actions_a[a];
					}

					// Update PID, get new angle
					pids[i]->setWeights(scaledActions[0], scaledActions[2], scaledActions[3]);
					double errorBound = static_cast<double>(parameters->dims[i]) / 2;
					// double scaledError = Utility::mapOutput(eventData[i].error, -errorBound, errorBound, 0.0, 2.0 * errorBound);
					newAngles[i] = pids[i]->update(eventData[i].error / errorBound, 0);
					// std::cout << "New angle before clamp: " << newAngles[i] << std::endl;
					newAngles[i] = Utility::mapOutput(newAngles[i], -75.0, 75.0, parameters->config->angleLow, parameters->config->angleHigh);
					
					// std::cout << "New angle: " << newAngles[i] << std::endl;	
					
					if (invert[i]) {
						newAngles[i] = parameters->config->angleHigh - newAngles[i];
					} 

					if (parameters->config->useArduino) {
						sendComand(commands[i], static_cast<unsigned char>(static_cast<int>(newAngles[i])), fd);
					}
					else {
						runServo(i, newAngles[i]);
					}		
				}

				// Done only once, for when we don't have a lastState
				if (programStart) { 
					
					lastState[i] = currentState[i];
					lastData[i] = eventData[i];

					if (i == 1) {
						programStart = false;
						goto sleep;
					}
					else {
						continue;
					}
					
				}
				else {
					// Fill out train data entry
					trainData[i].done = eventData[i].done;
					trainData[i].reward = errorToReward(eventData[i].error, lastData[i].error, parameters->dims[i] / 2, eventData[i].done);
					trainData[i].nextState = currentState[i];
					trainData[i].currentState = lastState[i];
					lastState[i] = currentState[i];
					lastData[i] = eventData[i];
				}

			}		
		}
		else {
			goto sleep;
		}

		if (!parameters->isTraining) {
			replayBuffer->add(trainData[1]);
		}
		
		if (!parameters->freshData) {

			if (pthread_mutex_lock(&trainLock) == 0) {
				if (replayBuffer->size() >= parameters->config->minBufferSize) {
					parameters->freshData = true;

					if (!parameters->config->offPolicyTrain) {
						parameters->isTraining = true;
					}
					
					pthread_cond_broadcast(&trainCond);
				}
			}

			pthread_mutex_unlock(&trainLock);
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
	float recheckChance = parameters->config->recheckChance;
	int trackerType = parameters->config->trackerType;
	bool useTracking = parameters->config->useTracking;
	bool draw = parameters->config->draw;
	bool showVideo = parameters->config->showVideo;
	bool cascadeDetector = parameters->config->cascadeDetector;
	std::string target = parameters->config->target;

	// program state variables
	bool rechecked = false;
	bool isTracking = false;
	bool isSearching = false;
	int lossCount = 0;
	int lossCountMax = parameters->config->lossCountMax;

	// Create object tracker to optimize detection performance
	cv::Rect2d roi;

	Ptr<Tracker> tracker;
	if (useTracking) {
		tracker = createOpenCVTracker(trackerType);
	}

	// Object center coordinates
	double frameCenterX = 0;
	double frameCenterY = 0;

	// Object coordinates
	double objX = 0;
	double objY = 0;

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
			// TODO:: Perform better search ruetine
			// For now servo thread detects when done and sends a reset commman to servos
		}

		if (draw) {
			Tbegin = chrono::steady_clock::now();
		}

		try
		{
			frame = GetImageFromCamera(camera);

			if (frame.empty())
			{
				// std::cout << "Issue reading frame!" << std::endl;
				continue;
			}

			// Convert to Gray Scale and resize
			double fx = 1 / 1.0;
			// cv::flip(frame, frame, -1);
			// cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
			// cv::equalizeHist(gray, gray);
			// resize(frame, frame, cv::Size(frame.cols / 2,  frame.rows / 2), fx, fx, cv::INTER_LINEAR);

			if (!useTracking) {
				goto detect;
			}

			if (isTracking) {
				isSearching = false;

				// Get the new tracking result
				if (!tracker->update(frame, roi)) {
					isTracking = false;
					// std::cout << "Lost target!!" << std::endl;
					lossCount++;
					goto detect;
				}

				// Chance to revalidate object tracking quality
				if (recheckChance >= static_cast<float>(rand()) / static_cast <float> (RAND_MAX)) {
					// std::cout << "Rechecking tracking quality..." << std::endl;
					rechecked = true;
					goto detect;
				}

			validated:
				ED tilt;
				ED pan;

				// Determine object and frame centers
				frameCenterX = static_cast<double>(frame.cols) / 2.0;
				frameCenterY = static_cast<double>(frame.rows) / 2.0;
				objX = roi.x + roi.width * 0.5;
				objY = roi.y + roi.height * 0.5;

				// Determine error
				tilt.error = frameCenterY - objY;
				pan.error = frameCenterX - objX;

				// Enter State data
				double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - execbegin).count() * 1e-9;
				pan.timestamp = elapsed;
				tilt.timestamp = elapsed;
				pan.Obj = objX;
				tilt.Obj = objY;
				pan.Frame = frameCenterX;
				tilt.Frame = frameCenterY;
				pan.done = false;
				tilt.done = false;

				// Fresh data
				if (pthread_mutex_lock(&stateDataLock) == 0) {
					eventDataArray[0] = tilt;
					eventDataArray[1] = pan;
					
					pthread_mutex_unlock(&stateDataLock);
				}

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

					// Update loop variants
					lossCount = 0;
					isSearching = false;

					if (rechecked) {
						rechecked = false;
						goto validated;
					}

					ED tilt;
					ED pan;

					// Determine object and frame centers
					frameCenterX = static_cast<double>(frame.cols) / 2.0;
					frameCenterY = static_cast<double>(frame.rows) / 2.0;
					objX = static_cast<double>(result.targetCenterX);
					objY = static_cast<double>(result.targetCenterY);

					// Determine error (negative is too far left or too far above)
					tilt.error = frameCenterY - objY;
					pan.error = frameCenterX - objX;

					// Other State data
					double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - execbegin).count() * 1e-9;
					pan.timestamp = elapsed;
					tilt.timestamp = elapsed;
					pan.Obj = objX;
					tilt.Obj = objY;
					pan.Frame = frameCenterX;
					tilt.Frame = frameCenterY;
					pan.done = false;
					tilt.done = false;

					// Fresh data
					if (pthread_mutex_lock(&stateDataLock) == 0) {
						eventDataArray[0] = tilt;
						eventDataArray[1] = pan;

						pthread_mutex_unlock(&stateDataLock);
					}

					if (useTracking) {

						roi.x = result.boundingBox.x;
						roi.y = result.boundingBox.y;
						roi.width = result.boundingBox.width;
						roi.height = result.boundingBox.height;

						tracker = createOpenCVTracker(trackerType);
						if (tracker->init(frame, roi)) {
							isTracking = true;
							// std::cout << "initialized!!" << std::endl;
							// std::cout << "Tracking Target..." << std::endl;
						}
					}
				}
				else {
					lossCount++;
					rechecked = false;

					// Target is out of sight, inform PID's, model, and servos
					if (lossCount >= lossCountMax) {
						
						isSearching = true;
						isTracking = false;
						lossCount = 0;

						ED tilt;
						ED pan;
						
						// Object not on screen
						frameCenterX = static_cast<double>(frame.cols) / 2.0;
						frameCenterY = static_cast<double>(frame.rows) / 2.0;
						objX = 0;
						objY = 0;

						// Max error
						tilt.error = frameCenterY;
						pan.error = frameCenterX;

						// Error state
						// Enter State data
						double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - execbegin).count() * 1e-9;
						pan.timestamp = elapsed;
						tilt.timestamp = elapsed;
						pan.Obj = objX;
						tilt.Obj = objY;
						pan.Frame = frameCenterX;
						tilt.Frame = frameCenterY;
						pan.done = true;
						tilt.done = true;

						// Fresh data
						if (pthread_mutex_lock(&stateDataLock) == 0) {
							eventDataArray[0] = tilt;
							eventDataArray[1] = pan;

							if (parameters->config->useArduino) {
								sendComand(0x8, 0x0, fd);
							}
							else {
								runServo(0, 90.0);
								runServo(1, 90.0);
							}

							pthread_mutex_unlock(&stateDataLock);
						}
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
	int batchSize = parameters->config->batchSize;
	bool offPolicy = parameters->config->offPolicyTrain;
	
	while (true) {
		
		// Wait until there is enough data in the training buffer
		while (!parameters->freshData) {
			pthread_cond_wait(&trainCond, &trainLock);
		}


		if (offPolicy) {
			TrainBuffer batch = replayBuffer->sample(batchSize);
			pidAutoTuner->update(batchSize, &batch);
			delay(1000);
		} 
		else {

			TrainBuffer buff = replayBuffer->getCopy();
			int numSamples = buff.size() / batchSize;
			for (int sample = 1, m = 0; sample <= numSamples; sample += 1) {
				int n = sample * batchSize - 1;
				
				TrainBuffer subBuf = slice(buff, m, n);
				pidAutoTuner->update(batchSize, &subBuf);

				m = n;
			}

			replayBuffer->clear();
			parameters->isTraining = false;
			parameters->freshData = false;
		}
	}

	return NULL;
}