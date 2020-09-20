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
			throw std::runtime_error("Failed to init I2C communication.");
		}
	}
	else {
		// Setup PCA9685 at address 0x40
		fd = pca9685Setup(PIN_BASE, 0x40, HERTZ);
		if (fd < 0)
		{
			throw std::runtime_error("Failed to init I2C communication.");
		}

		pca9685PWMReset(fd);
	}

	if (!camera.isOpened())
	{
		throw std::runtime_error("cannot initialize camera");
	}

	cv::Mat test = GetImageFromCamera(camera);

	if (test.empty())
	{
		throw std::runtime_error("Issue reading frame!");
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
	
	// Training options and record keeping
	double episodeAverageRewards = 0.0;
	double episodeAverageSteps = 0.0;
	double numEpisodes = 0.0;
	double episodeRewards = 0.0;
	double episodeSteps = 0.0;

	// training state variables
	bool initialRandomActions = parameters->config->initialRandomActions;
	int numInitialRandomActions = parameters->config->numInitialRandomActions;
	
	double predictedActions[NUM_SERVOS][NUM_ACTIONS];
	double stateArray[NUM_INPUT];
	SD currentState[NUM_SERVOS];
	TD trainData[NUM_SERVOS];
	ED eventData[NUM_SERVOS];
	RD resetResults;

	resetResults = servos->reset();	
	
	for (int servo = 0; servo < NUM_SERVOS; servo++) {
		currentState[servo] = resetResults.servos[servo];
	}

	while (true) {

		if (parameters->config->useAutoTuning) {

			if (!servos->isDone()) {
				for (int i = 0; i < 2; i++) {

					// Query network and get PID gains
					if (parameters->config->trainMode) {
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

				SR stepResults = servos->step(predictedActions);

				if (parameters->config->trainMode) {

					int size = replayBuffer->size();
					
					for (int servo = 0; servo < NUM_SERVOS; servo++) {
						trainData[servo] = stepResults.servos[servo];
						trainData[servo].currentState = currentState[servo];
						currentState[servo] = trainData[servo].nextState;

						if (replayBuffer->size() <= parameters->config->maxBufferSize) {
							replayBuffer->add(trainData[servo]);
						}
					}

					if (!trainData[1].done) {
						episodeSteps += 1.0;
						episodeRewards += trainData[1].reward;
					}
					else {
						numEpisodes += 1.0;


						// EMA of steps and rewards (With 30% weight to new episodes; or 5 episode averaging)
						double percentage = (1.0 / 3.0);
						double timePeriods = (2.0 / percentage) - 1.0;
						double emaWeight = (2.0 / (timePeriods + 1.0));

						episodeAverageRewards = (episodeRewards - episodeAverageRewards) * emaWeight + episodeAverageRewards;
						episodeAverageSteps = (episodeSteps - episodeAverageSteps) * emaWeight + episodeAverageSteps;

						std::cout << "Episode: " << numEpisodes << std::endl;
						std::cout << "Rewards were: " << episodeRewards << std::endl;
						std::cout << "Total steps were: " << episodeSteps << std::endl;
						std::cout << "EMA rewards: " << episodeAverageRewards << std::endl;
						std::cout << "EMA steps: " << episodeAverageSteps << std::endl;

						episodeSteps = 0.0;
						episodeRewards = 0.0;
					}

					if (!parameters->freshData) {
						pthread_mutex_lock(&trainLock);

							
						if (size > parameters->config->minBufferSize + 1) {
							parameters->freshData = true;
							pthread_cond_broadcast(&trainCond);
						} 

						pthread_mutex_unlock(&trainLock);
					}
	
				}
			}
			else {
				resetResults = servos->reset();
				for (int servo = 0; servo < NUM_SERVOS; servo++) {
					currentState[servo] = resetResults.servos[servo];
				}		
			}
		}
		else {
			if (!servos->isDone()) {
				for (int i = 0; i < 2; i++) {
					predictedActions[i][0] = parameters->config->defaultGains[0];
					predictedActions[i][1] = parameters->config->defaultGains[1];
					predictedActions[i][2] = parameters->config->defaultGains[2];
				}

				servos->step(predictedActions, false);
			}
			else {
				resetResults = servos->reset();
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
		throw std::runtime_error("cannot initialize camera");
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
				throw std::runtime_error("Error initializing caffe model");
			}
		}
		else {
			throw std::runtime_error("Error finding model and prototext files");
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
						pan.Obj = frameCenterX * 2;
						tilt.Obj = frameCenterY * 2; // max error
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
			throw std::runtime_error("Issue detecting target from video");
		}
	}

	return NULL;
}

void* autoTuneThread(void* args)
{
	param* parameters = (param*)args;
	int batchSize = parameters->config->batchSize;
	long maxTrainingSteps = parameters->config->maxTrainingSteps;
	long currentSteps = 0;
	int minBufferSize = parameters->config->minBufferSize;
	int maxBufferSize = parameters->config->maxBufferSize;
	int sessions = parameters->config->maxTrainingSessions;
	int numUpdates = parameters->config->numUpdates;
	bool offPolicy = parameters->config->offPolicyTrain;
	double rate = parameters->config->trainRate;

	// Annealed ERE (Emphasizing Recent Experience)
	// https://arxiv.org/pdf/1906.04009.pdf
	double N0 = 0.996;
	double NT = 1.0;
	double T = maxTrainingSteps;
	double t_i = 0;
	
start:

	parameters->freshData = false;
	replayBuffer->clear();
	
	if (sessions <= 0) {
		parameters->config->trainMode = false; // start running in eval mode
		std::cout << "Training session over!!" << std::endl;
	}
	
	// Wait for the buffer to fill
	pthread_mutex_lock(&trainLock);
	while (!parameters->freshData) {
		pthread_cond_wait(&trainCond, &trainLock);
	}
	pthread_mutex_unlock(&trainLock);

	while (true) {

		double N = static_cast<double>(replayBuffer->size());
		t_i += 1;
		double n_i = N0 + (NT - N0) * (t_i / T);
		int cmin = N - ( minBufferSize );
		
		if (offPolicy) {
			if (currentSteps >= maxTrainingSteps) {
				currentSteps = 0;
				sessions--;
				t_i = 0;
				goto start;
			}
			else {
				currentSteps += 1;
			}

			for (int k = 0; k < numUpdates; k += 1) {
				int startingRange = std::min<int>( N - N * std::pow(n_i, static_cast<double>(k) * (1000.0 / numUpdates)), cmin);

				TrainBuffer batch = replayBuffer->ere_sample(batchSize, startingRange);
				// TrainBuffer batch = replayBuffer->sample(batchSize, false);
				// TrainBuffer batch = replayBuffer->sample_transition(batchSize);
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