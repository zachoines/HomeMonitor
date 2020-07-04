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
#include <signal.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <malloc.h>
#include <sys/mman.h>
#include <errno.h>

// 3rd party libs
#include <wiringPi.h>
#include <wiringPiI2C.h>
#include <torch/torch.h>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
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
#include "Model.h"
#include "util.h"
#include "data.h"

#define ARDUINO_ADDRESS 0x9

using namespace cv;
using namespace std;
using namespace Utility;

// Last signal caught
volatile sig_atomic_t sig_value1; 
volatile sig_atomic_t sig_value2;
pid_t parent_pid;

static void usr_sig_handler1(const int sig_number, siginfo_t* sig_info, void* context);
void* tiltThread(void* args);
void* panThread(void* args);

int main(int argc, char** argv)
{
	// Register signal handler
	parent_pid = getpid();

	// Setup I2C comms and devices
	if (wiringPiSetupGpio() == -1)
		exit(1);

	int fd = wiringPiI2CSetup(ARDUINO_ADDRESS);
	if (fd == -1) {
		throw "Failed to init I2C communication.";
	}

	// Init shared memory
	int ShmID;
	int* ShmPTRParent;
	int* ShmPTRChild;
	int status;

	ShmID = shmget(IPC_PRIVATE, 7 * sizeof(int), IPC_CREAT | 0666);

	if (ShmID < 0) {
		throw "Could not initialize shared memory";
	}

	// Parent process is image recognition, child is PID and servo controller
	pid_t pid = fork();

	// Parent process
	if (pid > 0) {

		ShmPTRParent = (int*)shmat(ShmID, 0, 0);

		if ((int)ShmPTRParent == -1) {
			throw "Could not initialize shared memory";
		}

		// These values allow for async communication between parent process and child process's threads
		// These garentee, in a non-blocking way, the child process will never read a value twice.
		// But the child's threads may be reading old values at any given time.
		ShmPTRParent[0] = 0; // Tilt error
		ShmPTRParent[1] = 0; // Pan error
		ShmPTRParent[2] = 1; // Tilt lock
		ShmPTRParent[3] = 1; // Pan lock
		ShmPTRParent[4] = 1; // Reset signal
		ShmPTRParent[5] = 0; // Tilt read
		ShmPTRParent[6] = 0; // Pan read

		// user hyperparams
		float recheckChance = 0.01;
		bool useTracking = true;
		bool draw = false;
		bool showVideo = false;
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

					// Determine object and frame centers
					frameCenterX = static_cast<int>(frame.cols / 2);
					frameCenterY = static_cast<int>(frame.rows / 2);
					objX = roi.x + roi.width * 0.5;
					objY = roi.y + roi.height * 0.5;

					// Inform child process's threads of old data
					ShmPTRParent[2] = 0;
					ShmPTRParent[3] = 0;

					// Determine error
					ShmPTRParent[0] = frameCenterY - objY;
					ShmPTRParent[1] = frameCenterX - objX;

					// Reset read flags
					ShmPTRParent[5] = 0;
					ShmPTRParent[6] = 0;

					// Fresh data, now the child process's threads can read
					ShmPTRParent[2] = 1;
					ShmPTRParent[3] = 1;

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
						ShmPTRParent[2] = 0;
						ShmPTRParent[3] = 0;

						// Determine error
						ShmPTRParent[0] = frameCenterY - objY;
						ShmPTRParent[1] = frameCenterX - objX;

						// Reset read flags
						ShmPTRParent[5] = 0;
						ShmPTRParent[6] = 0;

						// Fresh data, now the child process's threads can read
						ShmPTRParent[2] = 1;
						ShmPTRParent[3] = 1;
			
						if (useTracking && !isTracking) {

							roi.x = result.boundingBox.x;
							roi.y = result.boundingBox.y;
							roi.width = result.boundingBox.width;
							roi.height = result.boundingBox.height;

							if (tracker->init(frame, roi)) {
								isTracking = true;
								std::cout << "initialized!!" << std::endl;
								std::cout << "Tracking Target..." << std::endl;
								ShmPTRParent[4] = 1;
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

		// Limit precision for calculations moving forward
		// std::setprecision(4);

		// Check for GPU
		// auto cuda_available = torch::cuda::is_available();
		// torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);

		// Create a memory pointer to char PID objects and 

		pid_t ppid = getpid();
		pid_t pid2 = fork();
		
		// Parent process
		if (pid2 > 0) {

			prctl(PR_SET_PDEATHSIG, SIGKILL); // Kill child when parent dies

			sigset_t  mask;
			siginfo_t info;
			pid_t     child, p;
			int       signum;

			sigemptyset(&mask);
			sigaddset(&mask, SIGINT);
			sigaddset(&mask, SIGHUP);
			sigaddset(&mask, SIGTERM);
			sigaddset(&mask, SIGQUIT);
			sigaddset(&mask, SIGUSR1);
			sigaddset(&mask, SIGUSR2);
			if (sigprocmask(SIG_BLOCK, &mask, NULL) == -1) {
				throw "Cannot block SIGUSR1 or SIGUSR2";
			}

			ShmPTRChild = (int*)shmat(ShmID, 0, 0);

			if ((int)ShmPTRChild == -1) {
				throw "Could not initialize shared memory";
			}

			param* parameters = (param*)malloc(sizeof(param));
			parameters->maxBufferSize = 256;
			Buffer* trainingBuffer;

			try {
				// Shared memory for training buffers
				boost::interprocess::shared_memory_object::remove("SharedTrainingBuffer");
				boost::interprocess::managed_shared_memory segment(boost::interprocess::create_only, "SharedTrainingBuffer", sizeof(TD) * parameters->maxBufferSize * 2);
				ShmemAllocator alloc_inst(segment.get_segment_manager());
				trainingBuffer = segment.construct<Buffer>("Buffer") (alloc_inst);
			}
			catch (...) {
				boost::interprocess::shared_memory_object::remove("SharedTrainingBuffer");
				std::cout << "Error encountered in servo and PID process." << std::endl;
				throw "Issue with shared memory object";
			}

			sendComand(0x8, 0x0, fd); // Reset servos

			// Setup shared thread parameters
			PID* pan = new PID(0.05, 0.04, 0.001, -75.0, 75.0);
			PID* tilt = new PID(0.05, 0.04, 0.001, -75.0, 75.0);
			// PIDAutoTuner* model = new Model();

			parameters->fd = fd;
			parameters->ShmPTR = ShmPTRChild;
			parameters->pan = pan;
			parameters->tilt = tilt;
			parameters->rate = 30; /* Updates per second */
			// parameters->model = model;
			parameters->mutex = PTHREAD_MUTEX_INITIALIZER;
			parameters->pid = pid2;
			parameters->isTraining = false;

			sleep(30);

			pthread_t panTid, tiltTid, trainTid;
			pthread_create(&panTid, NULL, panThread, (void*)parameters);
			pthread_create(&tiltTid, NULL, tiltThread, (void*)parameters);
			pthread_detach(panTid);
			pthread_detach(tiltTid);

			while (true) {

				signum = sigwaitinfo(&mask, &info);
				if (signum == -1) {
					if (errno == EINTR)
						continue;
					throw "Parent process: sigwaitinfo() failed";	
				}

				// Process training data on received signal
				if (signum == SIGUSR1 && info.si_pid == pid2) {
					std::cout << "Received finished training data..." << std::endl;
					if (trainingBuffer->empty()) {
						std::cout << "Updated weights..." << std::endl;
					}

					parameters->isTraining = false;
				}

				// Break when on SIGINT
				if (signum == SIGINT && !info.si_pid == pid2) {
					std::cout << "Ctrl+C detected!" << std::endl;
					break;
				}
			}

			kill(-parent_pid, SIGQUIT);
			if (wait(NULL) != -1) {
				return 0;
			}
			else {
				return -1;
			}
		}
		else {

			prctl(PR_SET_PDEATHSIG, SIGKILL); // Kill child when parent dies

			sleep(1); // wait for parent to fully initialize
			Buffer* trainingBuffer;
			
			// Retrieve the training buffer from shared memory
			boost::interprocess::managed_shared_memory segment(boost::interprocess::open_only, "SharedTrainingBuffer");
			trainingBuffer = segment.find<Buffer>("Buffer").first;
			// Train the model
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


			// PIDAutoTuner model;
			while ((sig_value1 != SIGINT) && (sig_value1 != SIGTERM))
			{
				sig_value1 = 0;

				// Sleep until signal is caught; train model on waking
				sigsuspend(&zeromask);

				if (sig_value1 == SIGUSR1) {
					std::cout << "Performing training session..." << std::endl;
					sleep(2);
					// Read in training session data
					

					// train on data
					trainingBuffer->clear();
					// send updated weights to parent
					kill(getppid(), SIGUSR1);
				}
			}
		}
	}
}

void* panThread(void* args) {

	boost::interprocess::managed_shared_memory segment(boost::interprocess::open_only, "SharedTrainingBuffer");
	Buffer* trainingBuffer;
	param* parameters = (param*)args;

	PID* pan = parameters->pan;
	int* ShmPTR = parameters->ShmPTR;
	int milis = 1000 / parameters->rate;
	int fd = parameters->fd;

	pan->init();
	int angleX;
	int currentAngleX = 90;

	trainingBuffer = segment.find<Buffer>("Buffer").first;

	while (true) {
		TD data;

		if (ShmPTR[4]) { // If the target is locked on
			if (ShmPTR[3] && ShmPTR[6] != 1) { // If the we are ready to read and not already have read it
				data.ref_input_1;
				data.ref_input_2;
				data.ref_output_1;
				data.ref_output_2;
				data.control_sig_1;
				data.control_sig_2;
				data.error = static_cast<double>(ShmPTR[1]);
				
				angleX = static_cast<int>(pan->update(data.error, 0));
				ShmPTR[6] = 1;
			}
			else {
				continue;
			}
		}
		else {
			pan->init();
		}

		/*std::cout << "X: ";
		std::cout << angleX << std::endl;*/

		if (currentAngleX != angleX) {
			int mappedX = mapOutput(angleX, -90, 90, 0, 180);
			sendComand(0x2, static_cast<unsigned char>(mappedX), fd);
			currentAngleX = angleX;
		}

		if (parameters->isTraining == false) {
			if (pthread_mutex_trylock(&parameters->mutex) == 0) {
				// trainingBuffer = segment.find<Buffer>("Buffer").first;
				if (parameters->isTraining == false) {
					
					

					try {
						if (trainingBuffer->size() == parameters->maxBufferSize) {
							std::cout << "Sending a training request..." << std::endl;
							parameters->isTraining = true;
							kill(parameters->pid, SIGUSR1);
						}
						else {
							trainingBuffer->push_back(data);
						}
					}
					catch (...)
					{
						std::cout << "Error in pan thread" << std::endl;
						throw;
					}
				}

				pthread_mutex_unlock(&parameters->mutex);
			}

			pthread_mutex_unlock(&parameters->mutex);
		}

		msleep(milis);
	}

	return NULL;
}

void* tiltThread(void* args) {

	boost::interprocess::managed_shared_memory segment(boost::interprocess::open_only, "SharedTrainingBuffer");
	Buffer* trainingBuffer;

	param* parameters = (param*)args;

	PID* tilt = parameters->tilt;
	int* ShmPTR = parameters->ShmPTR;
	int milis = 1000 / parameters->rate;
	int fd = parameters->fd;

	tilt->init();
	int angleY;
	int currentAngleY = 90;
	trainingBuffer = segment.find<Buffer>("Buffer").first;

	while (true) {

		if (ShmPTR[4]) { // If the target is locked on
			if (ShmPTR[2] && ShmPTR[5] != 1) { // If we are ready to read
				angleY = static_cast<int>(tilt->update(static_cast<double>(ShmPTR[0]), 0)) * -1;
				ShmPTR[5] = 1;
			}
			else {
				continue;
			}

			/*std::cout << "Y: ";
			std::cout << angleY << std::endl;*/

			if (currentAngleY != angleY) {

				int mappedY = mapOutput(angleY, -90, 90, 0, 180);
				sendComand(0x3, static_cast<unsigned char>(mappedY), fd);
				currentAngleY = angleY;
			}
		}
		else {
			tilt->init();
		}


		if (parameters->isTraining == false) {
			if (pthread_mutex_trylock(&parameters->mutex) == 0) {
				std::cout << "Here we are!!" << std::endl;
				// trainingBuffer = segment.find<Buffer>("Buffer").first;

				try {
					if (parameters->isTraining == false) {
						TD data;

						if (trainingBuffer->size() == parameters->maxBufferSize) {
							std::cout << "Sending a training request..." << std::endl;
							parameters->isTraining = true;
							kill(parameters->pid, SIGUSR1);
						}
						else {
							trainingBuffer->push_back(data);
						}
					}
				}
				catch (...)
				{
					std::cout << "Error in tilt thread" << std::endl;
					throw;
				}
				

				pthread_mutex_unlock(&parameters->mutex);
			}

			pthread_mutex_unlock(&parameters->mutex);
		}
		
		msleep(milis);
	}

	return NULL;
}

static void usr_sig_handler1(const int sig_number, siginfo_t* sig_info, void* context)
{
	// Take care of all segfaults
	if (sig_number == SIGSEGV)
	{
		perror("SIGSEV: Address access error.");
		exit(-1);
	}

	sig_value1 = sig_number;
}
