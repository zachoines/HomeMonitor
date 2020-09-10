// C++ libs
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <random>

// C libs
#include <ctime>
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

// Boost
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/variate_generator.hpp>

// Local classes and files
#include "CaffeDetector.h"
#include "CascadeDetector.h"
#include "PID.h"
#include "util.h"
#include "data.h"
#include "SACAgent.h"
#include "ReplayBuffer.h"
#include "Env.h"

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

pthread_cond_t dataCond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t dataLock = PTHREAD_MUTEX_INITIALIZER;

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

	parameters->dims[1] = width;
	parameters->dims[0] = height;

	PID* pan = new PID(parameters->config->defaultGains[0], parameters->config->defaultGains[1], parameters->config->defaultGains[2], parameters->config->pidOutputLow, parameters->config->pidOutputHigh, static_cast<double>(parameters->dims[1]) / 2.0);
	PID* tilt = new PID(parameters->config->defaultGains[0], parameters->config->defaultGains[1], parameters->config->defaultGains[2], parameters->config->pidOutputLow, parameters->config->pidOutputHigh, static_cast<double>(parameters->dims[0]) / 2.0);
	parameters->pan = pan;
	parameters->tilt = tilt;
	pidAutoTuner = new SACAgent(parameters->config->numInput, parameters->config->numHidden, parameters->config->numActions, parameters->config->actionHigh, parameters->config->actionLow);
	replayBuffer = new ReplayBuffer(parameters->config->maxBufferSize);

	if (parameters->config->useArduino) {
		sendComand(0x8, 0x0, fd); // Reset servos
	}
	else {
		runServo(0, parameters->config->resetAngleY);
		runServo(1, parameters->config->resetAngleX);
	}

	parameters->fd = fd;
	parameters->eventData = eventDataArray;
	parameters->rate = parameters->config->updateRate; 
	parameters->isTraining = false;
	parameters->freshData = false;

	pthread_t panTid, tiltTid, detectTid, autoTuneTid, panTiltTid;

	pthread_create(&panTiltTid, NULL, panTiltThread, (void*)parameters);
	pthread_create(&detectTid, NULL, detectThread, (void*)parameters);

	if (parameters->config->useAutoTuning) {
		pthread_create(&autoTuneTid, NULL, autoTuneThread, (void*)parameters);
	}
	

	pthread_join(panTiltTid, NULL);
	pthread_join(detectTid, NULL);

	if (parameters->config->useAutoTuning)
	{
		pthread_join(autoTuneTid, NULL);
	}
		

	// Terminate child processes and cleanup windows
	cv::destroyAllWindows();
}

void* panTiltThread(void* args) {
	std::mt19937 eng{ std::random_device{}() };
	auto options = torch::TensorOptions().dtype(torch::kDouble).device(device);
	param* parameters = (param*)args;
	Env* servos = new Env(parameters, &dataLock, &dataCond);

	// training state variables
	bool initialRandomActions = parameters->config->initialRandomActions;
	int numInitialRandomActions = parameters->config->numInitialRandomActions;
	bool trainMode = parameters->config->trainMode;
	
	double predictedActions[2][3] = {
		0.0
	};

	SD currentState[2];
	TD trainData[2];
	ED eventData[2];

	servos->resetEnv();

	if (!servos->init()) {
		throw "Could not initialize servos and pid's";
	}

	while (true) {

		if (parameters->config->useAutoTuning) {

			if (!servos->isDone()) {
				for (int i = 0; i < 2; i++) {

					double stateArray[parameters->config->numInput];

					// Query network and get PID gains
					if (trainMode) {
						if (initialRandomActions && numInitialRandomActions >= 0) {

							numInitialRandomActions--;
							std::cout << "Random action count: " << numInitialRandomActions << std::endl;
							for (int a = 0; a < parameters->config->numActions; a++) {
								predictedActions[i][a] = std::uniform_real_distribution<double>{ parameters->config->actionLow, parameters->config->actionHigh }(eng);;
							}
						}
						else {
							currentState[i].getStateArray(stateArray);
							at::Tensor actions = pidAutoTuner->get_action(torch::from_blob(stateArray, { 1, parameters->config->numInput }, options), true);

							if (parameters->config->numActions > 1) {
								auto actions_a = actions.accessor<double, 1>();
								for (int a = 0; a < parameters->config->numActions; a++) {
									predictedActions[i][a] = actions_a[a];
								}
							}
							else {
								predictedActions[i][0] = actions.item().toDouble();
							}
							
						}
					}
					else {
						currentState[i].getStateArray(stateArray);
						at::Tensor actions = pidAutoTuner->get_action(torch::from_blob(stateArray, { 1, parameters->config->numInput }, options), false);
						if (parameters->config->numActions > 1) {
							auto actions_a = actions.accessor<double, 1>();
							for (int a = 0; a < parameters->config->numActions; a++) {
								predictedActions[i][a] = actions_a[a];
							}
						}
						else {
							predictedActions[i][0] = actions.item().toDouble();
						}
					}
					
				}

				servos->step(predictedActions);

				if (trainMode) {

					if (servos->hasData()) {
						servos->getResults(trainData);
						// replayBuffer->add(trainData[0]);
						replayBuffer->add(trainData[1]);

						if (!parameters->freshData) {
							pthread_mutex_lock(&trainLock);

							if (replayBuffer->size() >= parameters->config->minBufferSize) {
								parameters->freshData = true;
								pthread_cond_broadcast(&trainCond);
							}

							pthread_mutex_unlock(&trainLock);
						}
					}
				}
			}
			else {
				servos->ping();
			}
		}
		else {
			if (!servos->isDone()) {
				for (int i = 0; i < 2; i++) {
					predictedActions[i][0] = parameters->config->defaultGains[0];
					predictedActions[i][1] = parameters->config->defaultGains[1];
					predictedActions[i][2] = parameters->config->defaultGains[2];
				}

				servos->step(predictedActions);
			}
			else {
				servos->ping();
			}	
		}
	}
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

	double lastError = -1;

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
				pan.point = roi.x;
				tilt.point = roi.y;
				pan.size = roi.width;
				tilt.size = roi.height;
				pan.Obj = objX;
				tilt.Obj = objY;
				pan.Frame = frameCenterX;
				tilt.Frame = frameCenterY;
				pan.done = false;
				tilt.done = false;

				lastError = objX;
				
				double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - execbegin).count() * 1e-9;
				pan.timestamp = elapsed;
				tilt.timestamp = elapsed;

				// Fresh data
				if (pthread_mutex_lock(&dataLock) == 0) {
					eventDataArray[0] = tilt;
					eventDataArray[1] = pan;
					
					pthread_cond_broadcast(&dataCond);
				}
				pthread_mutex_unlock(&dataLock);

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
					/*tilt.error = objY;
					pan.error = objX;*/

					// Other State data
					double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - execbegin).count() * 1e-9;
					pan.timestamp = elapsed;
					tilt.timestamp = elapsed;

					pan.point = static_cast<double>(result.boundingBox.x);
					tilt.point = static_cast<double>(result.boundingBox.y);
					pan.size = static_cast<double>(result.boundingBox.width);
					tilt.size = static_cast<double>(result.boundingBox.height);
					pan.Obj = objX;
					tilt.Obj = objY;
					pan.Frame = frameCenterX;
					tilt.Frame = frameCenterY;
					pan.done = false;
					tilt.done = false;

					// Fresh data
					if (pthread_mutex_lock(&dataLock) == 0) {
						eventDataArray[0] = tilt;
						eventDataArray[1] = pan;

						pthread_cond_broadcast(&dataCond);
					}
					pthread_mutex_unlock(&dataLock);

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
						pan.point = 0;
						tilt.point = 0;
						pan.size = 0;
						tilt.size = 0;
						pan.Obj = objX;
						tilt.Obj = objY;
						pan.Frame = frameCenterX;
						tilt.Frame = frameCenterY;
						pan.done = true;
						tilt.done = true;

						// Fresh data
						if (pthread_mutex_lock(&dataLock) == 0) {
							eventDataArray[0] = tilt;
							eventDataArray[1] = pan;

							pthread_cond_broadcast(&dataCond);
						}

						pthread_mutex_unlock(&dataLock);
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
	int sessions = parameters->config->maxTrainingSessions;
	bool offPolicy = parameters->config->offPolicyTrain;
	double rate = parameters->config->trainRate;

	// Wait for the buffer to fill
	pthread_mutex_lock(&trainLock);
	while (!parameters->freshData) {
		pthread_cond_wait(&trainCond, &trainLock);
	}
	pthread_mutex_unlock(&trainLock);

	while (true) {

		if (offPolicy) {
			for (int i = 0; i < sessions; i += 1) {
				TrainBuffer batch = replayBuffer->sample(batchSize, false);
				pidAutoTuner->update(batch.size(), &batch);
			}

			long milis = static_cast<long>(1000.0 / rate);
			Utility::msleep(milis);
		}
		else {
			
			pthread_mutex_lock(&trainLock);
			while (!parameters->freshData) {
				pthread_cond_wait(&trainCond, &trainLock);
			}
			pthread_mutex_unlock(&trainLock);

			TrainBuffer buff = replayBuffer->getCopy();
			for (int i = 0; i < sessions; i += 1) {
				
				replayBuffer->clear();
				int numSamples = buff.size() / batchSize;
				for (int sample = 1, m = 0; sample <= numSamples; sample += 1) {
					int n = sample * batchSize - 1;

					TrainBuffer subBuf = slice(buff, m, n);
					pidAutoTuner->update(batchSize, &subBuf);

					m = n + 1;
				}

				int milis = 1000 / rate;
				Utility::msleep(milis);
			}

			parameters->freshData = false;
		}

		
	}

	return NULL;
}