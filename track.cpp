//// C++ libs
//#include <iostream>
//#include <string>
//#include <vector>
//#include <filesystem>
//
//// C libs
//#include <dirent.h>
//#include <stdio.h>
//#include <unistd.h>
//#include <errno.h>
//#include <fcntl.h>
//#include <stdlib.h>
//#include <sys/stat.h>
//#include <sys/types.h>
//#include <sys/wait.h> 
//#include <sys/prctl.h>
//#include <signal.h>
//#include <sys/ipc.h>
//#include <sys/shm.h>
//#include <malloc.h>
//#include <sys/mman.h>
//#include <errno.h>
//
//// 3rd party libs
//#include <wiringPi.h>
//#include <wiringPiI2C.h>
//#include <torch/torch.h>
//#include <boost/interprocess/managed_shared_memory.hpp>
//#include <boost/interprocess/containers/vector.hpp>
//#include <boost/interprocess/allocators/allocator.hpp>
//#include <boost/interprocess/sync/interprocess_mutex.hpp>
//#include <boost/interprocess/sync/interprocess_condition.hpp>
//#include <algorithm>
//
//// OpenCV imports
//#include "opencv2/opencv.hpp"
//#include "opencv2/core/ocl.hpp"
//#include "opencv4/opencv2/dnn/dnn.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/imgcodecs/imgcodecs.hpp"
//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/videoio/videoio.hpp"
//#include "opencv2/video/video.hpp"
//#include <opencv2/tracking.hpp>
//
//// Local classes and files
//#include "CaffeDetector.h"
//#include "CascadeDetector.h"
//#include "PID.h"
//// #include "QNetwork.h"
//#include "util.h"
//#include "data.h"
//
//#define ARDUINO_ADDRESS 0x9
//
//using namespace cv;
//using namespace std;
//using namespace Utility;
//
//// Last signal caught
//volatile sig_atomic_t sig_value1; 
//pid_t parent_pid;
//
//static void usr_sig_handler1(const int sig_number, siginfo_t* sig_info, void* context);
//void* tiltThread(void* args);
//void* panThread(void* args);
//
//int track(int argc, char** argv)
//{
//	// Register signal handler
//	parent_pid = getpid();
//
//	// Setup I2C comms and devices
//	if (wiringPiSetupGpio() == -1)
//		exit(1);
//
//	int fd = wiringPiI2CSetup(ARDUINO_ADDRESS);
//	if (fd == -1) {
//		throw "Failed to init I2C communication.";
//	}
//
//	// Init shared memory
//	int ShmID;
//	ED* ShmPTRParent;
//	ED* ShmPTRChild;
//	int status;
//
//	ShmID = shmget(IPC_PRIVATE, 2 * sizeof(ED), IPC_CREAT | 0666);
//
//	if (ShmID < 0) {
//		throw "Could not initialize shared memory";
//	}
//
//	// Parent process is image recognition, child is PID and servo controller
//	pid_t pid = fork();
//
//	// Parent process
//	if (pid > 0) {
//
//		ShmPTRParent = (ED*)shmat(ShmID, 0, 0);
//
//		if ((int)ShmPTRParent == -1) {
//			throw "Could not initialize shared memory";
//		}
//
//		// user hyperparams
//		float recheckChance = 0.01;
//		bool useTracking = true;
//		bool draw = true;
//		bool showVideo = true;
//		bool cascadeDetector = true;
//		std::string target = "sheep";
//
//		// program state variables
//		bool rechecked = false;
//		bool isTracking = false;
//		bool isSearching = false;
//		int lossCount = 0;
//		int lossCountMax = 100;
//
//		// Create object tracker to optimize detection performance
//		cv::Rect2d roi;
//		Ptr<Tracker> tracker = cv::TrackerCSRT::create();
//		// Ptr<Tracker> tracker = cv::TrackerMOSSE::create();
//		// Ptr<Tracker> tracker = cv::TrackerGOTURN::create();
//
//		// Object center coordinates
//		int frameCenterX = 0;
//		int frameCenterY = 0;
//
//		// Object coordinates
//		int objX = 0;
//		int objY = 0;
//
//		float f;
//		float FPS[16];
//		int i, Fcnt = 0;
//		vector<string> class_names = {
//			"background",
//			"aeroplane", "bicycle", "bird", "boat",
//			"bottle", "bus", "car", "cat", "chair",
//			"cow", "diningtable", "dog", "horse",
//			"motorbike", "person", "pottedplant",
//			"sheep", "sofa", "train", "tvmonitor"
//		};
//
//		cv::Mat frame;
//		cv::Mat detection;
//		chrono::steady_clock::time_point Tbegin, Tend;
//		auto execbegin = std::chrono::high_resolution_clock::now();
//
//		cv::VideoCapture camera(0);
//		if (!camera.isOpened())
//		{
//			throw "cannot initialize camera";
//		}
//
//		std::string prototextFile = "/MobileNetSSD_deploy.prototxt";
//		std::string modelFile = "/MobileNetSSD_deploy.caffemodel";
//		std::string path = get_current_dir_name();
//		std::string prototextFilePath = path + prototextFile;
//		std::string modelFilePath = path + modelFile;
//		dnn::Net net;
//
//		if (!cascadeDetector) {
//			if (fileExists(modelFilePath) && fileExists(prototextFilePath)) {
//				net = dnn::readNetFromCaffe(prototextFilePath, modelFilePath);
//				if (net.empty()) {
//					throw "Error initializing caffe model";
//				}
//			}
//			else {
//				throw "Error finding model and prototext files";
//			}
//		}
//		
//
//		// HM::CaffeDetector cd(net, class_names);
//		HM::CascadeDetector cd;
//		HM::DetectionData result;
//
//		while (true) {
//
//			if (isSearching) {
//				isSearching = false;
//
//				// TODO:: Perform search reutine
//				// sendComand(0x8, 0x0, fd); // Reset servos
//			}
//
//			if (draw) {
//				Tbegin = chrono::steady_clock::now();
//			}
//
//			try
//			{
//				frame = GetImageFromCamera(camera);
//
//				if (frame.empty())
//				{
//					std::cout << "Issue reading frame!" << std::endl;
//					continue;
//				}
//
//				// Convert to Gray Scale and resize
//				double fx = 1 / 1.0;
//				cv::flip(frame, frame, -1);
//				// cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
//				// cv::equalizeHist(gray, gray);
//				// resize(frame, frame, cv::Size(frame.cols / 2,  frame.rows / 2), fx, fx, cv::INTER_LINEAR);
//
//				if (!useTracking) {
//					goto detect;
//				}
//
//				if (isTracking) {
//
//					// Get the new tracking result
//					if (!tracker->update(frame, roi)) {
//						isTracking = false;
//						std::cout << "Lost target!!" << std::endl; 
//						lossCount++;
//						goto detect;
//					}
//
//					// Chance to revalidate object tracking quality
//					if (recheckChance >= static_cast<float>(rand()) / static_cast <float> (RAND_MAX)) {
//						std::cout << "Rechecking tracking quality..." << std::endl;
//						goto detect;
//					}
//
//				validated:
//					ED tilt;
//					ED pan;
//					// Determine object and frame centers
//					frameCenterX = static_cast<int>(frame.cols / 2);
//					frameCenterY = static_cast<int>(frame.rows / 2);
//					objX = roi.x + roi.width * 0.5;
//					objY = roi.y + roi.height * 0.5;
//
//					// Inform child process's threads of old data (race condition here, kinda washes out in end)
//					ShmPTRParent[0].dirty = true;
//					ShmPTRParent[1].dirty = true;
//
//					// Determine error
//					tilt.error = static_cast<double>(frameCenterY - objY);
//					pan.error = static_cast<double>(frameCenterX - objX);
//		
//					// Enter State data
//					double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - execbegin).count() * 1e-9;
//					pan.timestamp = elapsed;
//					tilt.timestamp = elapsed;
//					pan.Obj = objX;
//					tilt.Obj = objY;
//					pan.Frame = frameCenterY;
//					tilt.Frame = frameCenterX;
//					pan.done = false;
//					tilt.done = false;
//
//					// Fresh data, now the child process's threads can read
//					ShmPTRParent[0] = tilt;
//					ShmPTRParent[1] = pan;
//
//					// draw to frame
//					if (draw) {
//						cv::Scalar color = cv::Scalar(255);
//						cv::Rect rec(
//							roi.x,
//							roi.y,
//							roi.width,
//							roi.height
//						);
//						circle(
//							frame,
//							cv::Point(objX, objY),
//							(int)(roi.width + roi.height) / 2 / 10,
//							color, 2, 8, 0);
//						rectangle(frame, rec, color, 2, 8, 0);
//						putText(
//							frame,
//							target,
//							cv::Point(roi.x, roi.y - 5),
//							cv::FONT_HERSHEY_SIMPLEX,
//							1.0,
//							color, 2, 8, 0);
//					}
//				}
//				else {
//
//				detect:
//					result = cd.detect(frame, target, draw);
//
//					if (result.found) {
//
//						if (rechecked) {
//							goto validated;
//						}
//
//						// Determine object and frame centers
//						frameCenterX = static_cast<int>(frame.cols / 2);
//						frameCenterY = static_cast<int>(frame.rows / 2);
//						objX = result.targetCenterX;
//						objY = result.targetCenterY;
//
//						ED tilt;
//						ED pan;
//						// Determine object and frame centers
//						frameCenterX = static_cast<int>(frame.cols / 2);
//						frameCenterY = static_cast<int>(frame.rows / 2);
//						objX = roi.x + roi.width * 0.5;
//						objY = roi.y + roi.height * 0.5;
//
//						// Inform child process's threads of old data (race condition here, kinda washes out in end)
//						ShmPTRParent[0].dirty = true;
//						ShmPTRParent[1].dirty = true;
//
//						// Determine error
//						tilt.error = static_cast<double>(frameCenterY - objY);
//						pan.error = static_cast<double>(frameCenterX - objX);
//
//						// Other State data
//						double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - execbegin).count() * 1e-9;
//						pan.timestamp = elapsed;
//						tilt.timestamp = elapsed;
//						pan.Obj = objX;
//						tilt.Obj = objY;
//						pan.Frame = frameCenterY;
//						tilt.Frame = frameCenterX;
//						pan.done = true;
//						tilt.done = true;
//
//						// Fresh data, now the child process's threads can read
//						ShmPTRParent[0] = tilt;
//						ShmPTRParent[1] = pan;
//			
//						if (useTracking) {
//
//							roi.x = result.boundingBox.x;
//							roi.y = result.boundingBox.y;
//							roi.width = result.boundingBox.width;
//							roi.height = result.boundingBox.height;
//
//							if (tracker->init(frame, roi)) {
//								isTracking = true;
//								std::cout << "initialized!!" << std::endl;
//								std::cout << "Tracking Target..." << std::endl;
//							} 
//						}
//					}
//					else {
//						lossCount++;
//
//						if (lossCount >= lossCountMax) {
//							std::cout << "No target found" << std::endl;
//							isSearching = true;
//							isTracking = false;
//							lossCount = 0;
//						}
//					}
//				}
//
//				if (showVideo) {
//					if (draw) {
//						Tend = chrono::steady_clock::now();
//						f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
//						if (f > 0.0) FPS[((Fcnt++) & 0x0F)] = 1000.0 / f;
//						for (f = 0.0, i = 0; i < 16; i++) { f += FPS[i]; }
//						putText(frame, format("FPS %0.2f", f / 16), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255));
//					}
//
//					cv::imshow("Viewport", frame);
//					waitKey(1);
//				}
//			}
//			catch (const std::exception&)
//			{
//				std::cout << "Issue detecting target from video" << std::endl;
//				exit(-1);
//			}
//		}
//
//		// Terminate child processes and cleanup windows
//		if (showVideo) {
//			cv::destroyAllWindows();
//		}
//
//		kill(-parent_pid, SIGQUIT);
//		if (wait(NULL) != -1) {
//			return 0;
//		}
//		else {
//			return -1;
//		}
//	}
//
//	// Child process
//	else {
//
//		// Create a memory pointer to char PID objects and 
//
//		pid_t ppid = getpid();
//		pid_t pid2 = fork();
//		
//		// Parent process
//		if (pid2 > 0) {
//
//			prctl(PR_SET_PDEATHSIG, SIGKILL); // Kill child when parent dies
//
//			sigset_t  mask;
//			siginfo_t info;
//			pid_t     child, p;
//			int       signum;
//
//			sigemptyset(&mask);
//			sigaddset(&mask, SIGINT);
//			sigaddset(&mask, SIGHUP);
//			sigaddset(&mask, SIGTERM);
//			sigaddset(&mask, SIGQUIT);
//			sigaddset(&mask, SIGUSR1);
//			sigaddset(&mask, SIGUSR2);
//			if (sigprocmask(SIG_BLOCK, &mask, NULL) == -1) {
//				throw "Cannot block SIGUSR1 or SIGUSR2";
//			}
//
//			ShmPTRChild = (ED*)shmat(ShmID, 0, 0);
//
//			if ((int)ShmPTRChild == -1) {
//				throw "Could not initialize shared memory";
//			}
//
//			param* parameters = (param*)malloc(sizeof(param));
//			parameters->maxBufferSize = 256;
//			Buffer* trainingBuffer;
//
//			try {
//				// Shared memory for training buffers
//				boost::interprocess::shared_memory_object::remove("SharedTrainingBuffer");
//				boost::interprocess::managed_shared_memory segment(boost::interprocess::create_only, "SharedTrainingBuffer", sizeof(TD) * parameters->maxBufferSize * 2);
//				ShmemAllocator alloc_inst(segment.get_segment_manager());
//				trainingBuffer = segment.construct<Buffer>("Buffer") (alloc_inst);
//			}
//			catch (...) {
//				boost::interprocess::shared_memory_object::remove("SharedTrainingBuffer");
//				std::cout << "Error encountered in servo and PID process." << std::endl;
//				throw "Issue with shared memory object";
//			}
//
//			sendComand(0x8, 0x0, fd); // Reset servos
//
//			// Setup shared thread parameters
//			PID* pan = new PID(0.05, 0.04, 0.001, -75.0, 75.0);
//			PID* tilt = new PID(0.05, 0.04, 0.001, -75.0, 75.0);
//
//			parameters->fd = fd;
//			parameters->ShmPTR = ShmPTRChild;
//			parameters->pan = pan;
//			parameters->tilt = tilt;
//			parameters->rate = 6; /* Updates per second */
//			parameters->mutex = PTHREAD_MUTEX_INITIALIZER;
//			parameters->pid = pid2;
//			parameters->isTraining = false;
//
//			pthread_t panTid, tiltTid, trainTid;
//			pthread_create(&panTid, NULL, panThread, (void*)parameters);
//			pthread_create(&tiltTid, NULL, tiltThread, (void*)parameters);
//			pthread_detach(panTid);
//			pthread_detach(tiltTid);
//
//			while (true) {
//
//				signum = sigwaitinfo(&mask, &info);
//				if (signum == -1) {
//					if (errno == EINTR)
//						continue;
//					throw "Parent process: sigwaitinfo() failed";	
//				}
//
//				// Process training data on received signal
//				if (signum == SIGUSR1 && info.si_pid == pid2) {
//					std::cout << "Received finished training data..." << std::endl;
//					if (trainingBuffer->empty()) {
//						std::cout << "Updated weights..." << std::endl;
//					}
//
//					parameters->isTraining = false;
//				}
//
//				// Break when on SIGINT
//				if (signum == SIGINT && !info.si_pid == pid2) {
//					std::cout << "Ctrl+C detected!" << std::endl;
//					break;
//				}
//			}
//
//			kill(-parent_pid, SIGQUIT);
//			if (wait(NULL) != -1) {
//				return 0;
//			}
//			else {
//				return -1;
//			}
//		}
//		else {
//
//			// Check for GPU
//			auto cuda_available = torch::cuda::is_available();
//			torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
//
//			prctl(PR_SET_PDEATHSIG, SIGKILL); // Kill child when parent dies
//
//			sleep(1); // wait for parent to fully initialize
//			Buffer* trainingBuffer;
//			
//			// Retrieve the training buffer from shared memory
//			boost::interprocess::managed_shared_memory segment(boost::interprocess::open_only, "SharedTrainingBuffer");
//			trainingBuffer = segment.find<Buffer>("Buffer").first;
//			// Train the model
//			struct sigaction sig_action;
//
//			sigset_t oldmask;
//			sigset_t newmask;
//			sigset_t zeromask;
//
//			memset(&sig_action, 0, sizeof(struct sigaction));
//
//			sig_action.sa_flags = SA_SIGINFO;
//			sig_action.sa_sigaction = usr_sig_handler1;
//
//			sigaction(SIGHUP, &sig_action, NULL);
//			sigaction(SIGINT, &sig_action, NULL);
//			sigaction(SIGTERM, &sig_action, NULL);
//			sigaction(SIGSEGV, &sig_action, NULL);
//			sigaction(SIGUSR1, &sig_action, NULL);
//
//			sigemptyset(&newmask);
//			sigaddset(&newmask, SIGHUP);
//			sigaddset(&newmask, SIGINT);
//			sigaddset(&newmask, SIGTERM);
//			sigaddset(&newmask, SIGSEGV);
//			sigaddset(&newmask, SIGUSR1);
//
//			sigprocmask(SIG_BLOCK, &newmask, &oldmask);
//			sigemptyset(&zeromask);
//			sig_value1 = 0;
//
//			while ((sig_value1 != SIGINT) && (sig_value1 != SIGTERM))
//			{
//				sig_value1 = 0;
//
//				// Sleep until signal is caught; train model on waking
//				sigsuspend(&zeromask);
//
//				if (sig_value1 == SIGUSR1) {
//					std::cout << "Performing training session..." << std::endl;
//					sleep(2);
//					// TODO Read in training session data
//					
//
//					// TODO Remove this line
//					trainingBuffer->clear();
//					
//					// TODO send updated weights to parent
//					kill(getppid(), SIGUSR1);
//				}
//			}
//		}
//	}
//}
//
//void* panThread(void* args) {
//
//	boost::interprocess::managed_shared_memory segment(boost::interprocess::open_only, "SharedTrainingBuffer");
//	Buffer* trainingBuffer;
//	param* parameters = (param*)args;
//
//	PID* pan = parameters->pan;
//	ED* ShmPTR = parameters->ShmPTR;
//	int milis = 1000 / parameters->rate;
//	int fd = parameters->fd;
//	int angleX;
//	int currentAngleX = 90;
//	bool programStart = true;
//	double lastTimeStep;
//	SD lastState;
//
//	trainingBuffer = segment.find<Buffer>("Buffer").first;
//	pan->init();
//	
//	
//	while (true) {
//		SD currentState;
//		TD trainData;
//		ED panData = ShmPTR[1];
//
//		if (!panData.dirty && !panData.isOld(lastTimeStep)) { // If its not old and not already read
//			
//			lastTimeStep = panData.timestamp;
//			angleX = static_cast<int>(pan->update(panData.error, 0));
//			currentState.Obj = panData.Obj;
//			currentState.Frame = panData.Frame;
//			currentState.Angle = angleX;
//			currentState.error = panData.error;
//			trainData.done = panData.done;
//			trainData.reward = static_cast<int>(mapOutput(std::abs(panData.error), -90, 90, 0, 100) / 100.0);
//			pan->getWeights(trainData.actions);
//
//			if (currentAngleX != angleX) {
//				int mappedX = mapOutput(angleX, -90, 90, 0, 180);
//				sendComand(0x2, static_cast<unsigned char>(mappedX), fd);
//				currentAngleX = angleX;
//			}
//
//			if (programStart) { // For when we dont have a lastState
//				programStart = false;
//				lastState = currentState;
//				goto sleep;
//			} 
//			else {
//				trainData.nextState = currentState;
//				trainData.currentState = lastState;
//				lastState = currentState;
//			}
//
//			if (trainData.done) {
//				pan->init();
//			}
//		}
//		else {
//			// pan->update(0.0, 0);
//			goto sleep;
//		}
//
//		if (!parameters->isTraining) {
//			
//			if (pthread_mutex_trylock(&parameters->mutex) == 0) {
//				
//				if (!parameters->isTraining) {
//					
//					try {
//						if (trainingBuffer->size() == parameters->maxBufferSize) {
//							std::cout << "Sending a training request..." << std::endl;
//							parameters->isTraining = true;
//							kill(parameters->pid, SIGUSR1);
//						}
//						else {
//							trainingBuffer->push_back(trainData);
//						}
//					}
//					catch (...)
//					{
//						throw "Error in pan thread";
//					}
//				}
//
//				pthread_mutex_unlock(&parameters->mutex);
//			}
//
//			pthread_mutex_unlock(&parameters->mutex);
//		}
//
//	sleep:
//		msleep(milis);
//	}
//
//	return NULL;
//}
//
//void* tiltThread(void* args) {
//
//	boost::interprocess::managed_shared_memory segment(boost::interprocess::open_only, "SharedTrainingBuffer");
//	Buffer* trainingBuffer;
//
//	param* parameters = (param*)args;
//
//	PID* tilt = parameters->tilt;
//	ED* ShmPTR = parameters->ShmPTR;
//	int milis = 1000 / parameters->rate;
//	int fd = parameters->fd;
//	bool programStart = true;
//	int angleY;
//	int currentAngleY = 90;
//	double lastTimeStep;
//	SD lastState;
//
//	tilt->init();
//	trainingBuffer = segment.find<Buffer>("Buffer").first;
//
//	while (true) {
//		SD currentState;
//		TD trainData;
//		ED tiltData = ShmPTR[0];
//
//		if (!tiltData.dirty && !tiltData.isOld(lastTimeStep)) { // If its not old and not already read
//
//			lastTimeStep = tiltData.timestamp;
//			angleY = static_cast<int>(tilt->update(tiltData.error, 0)) * -1;
//			currentState.Obj = tiltData.Obj;
//			currentState.Frame = tiltData.Frame;
//			currentState.Angle = angleY;
//			currentState.error = tiltData.error;
//			trainData.done = tiltData.done;
//			trainData.reward = tiltData.error;
//			tilt->getWeights(trainData.actions);
//
//			if (currentAngleY != angleY) {
//				int mappedY = mapOutput(angleY, -90, 90, 0, 180);
//				sendComand(0x3, static_cast<unsigned char>(mappedY), fd);
//				currentAngleY = angleY;
//			}
//
//			if (programStart) { // For when we dont have a lastState
//				programStart = false;
//				lastState = currentState;
//				goto sleep;
//			}
//			else {
//				trainData.nextState = currentState;
//				trainData.currentState = lastState;
//				lastState = currentState;
//			}
//
//			if (trainData.done) {
//				tilt->init();
//			}
//		}
//		else {
//			// tilt->update(0.0, 0);
//			goto sleep;
//		}
//
//		if (!parameters->isTraining) {
//
//			if (pthread_mutex_trylock(&parameters->mutex) == 0) {
//
//				if (!parameters->isTraining) {
//
//					try {
//						if (trainingBuffer->size() == parameters->maxBufferSize) {
//							std::cout << "Sending a training request..." << std::endl;
//							parameters->isTraining = true;
//							kill(parameters->pid, SIGUSR1);
//						}
//						else {
//							trainingBuffer->push_back(trainData);
//						}
//					}
//					catch (...)
//					{
//						throw "Error in tilt thread";
//					}
//				}
//
//				pthread_mutex_unlock(&parameters->mutex);
//			}
//
//			pthread_mutex_unlock(&parameters->mutex);
//		}
//
//	sleep:
//		msleep(milis);
//	}
//
//	return NULL;
//}
//
//static void usr_sig_handler1(const int sig_number, siginfo_t* sig_info, void* context)
//{
//	// Take care of all segfaults
//	if (sig_number == SIGSEGV)
//	{
//		perror("SIGSEV: Address access error.");
//		exit(-1);
//	}
//
//	sig_value1 = sig_number;
//}
