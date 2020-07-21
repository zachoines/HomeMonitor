// C++ libs
#pragma once
#include <iostream>
#include <fstream>
#include <string>
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
#include <cmath>

// 3rd party libs
#include <wiringPiI2C.h>
#include "opencv2/opencv.hpp"

namespace Utility {
	static int msleep(long msec)
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


	// FD write that handles interrupts
	static ssize_t r_write(int fd, void* buf, size_t size)
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

	// FD read that handles interrupts
	static ssize_t r_read(int fd, void* buf, size_t size)
	{
		ssize_t retval;

		while (retval = read(fd, buf, size), retval == -1 && errno == EINTR);

		return retval;
	}


	static int mapOutput(int x, int in_min, int in_max, int out_min, int out_max) {
		return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
	}

	static double mapOutput(double x, double in_min, double in_max, double out_min, double out_max) {
		return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
	}

	// Return response data
	static void sendComand(unsigned char command, unsigned char data, int fd) {
		unsigned short finalCommand = (command << 8) + data;
		wiringPiI2CWriteReg16(fd, 0, finalCommand);
	}

	static bool fileExists(std::string fileName)
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

	/* 
		If done error: -2
		Else reward is -1 to +1, plus additional reward bias for transitions that are good (and vic versa)
		Bias is a exponential func from 0.0 to 1.0 
	*/
	static double errorToReward(int errorNew, int errorOld, int absMax, bool done) {

		if (done) {
			return -2.0;
		}

		// ABS Errors
		errorOld = (errorOld < 0) ? (-errorOld) : (errorOld);
		errorNew = (errorNew < 0) ? (-errorNew) : (errorNew);
		
		// Scale errors from 0.0 to 1.0
		double errorOldScaled = static_cast<double>(errorOld) / static_cast<double>(absMax);
		double errorNewScaled = static_cast<double>(errorNew) / static_cast<double>(absMax);
		
		// Score the errors
		double errorOldScore = mapOutput(1.0 - errorOldScaled, 0.0, 1.0, -1.0, 1.0);
		double errorNewScore = mapOutput(1.0 - errorNewScaled, 0.0, 1.0, -1.0, 1.0);

		if (errorNewScore > errorOldScore) {
			double percentBetter =  1.0 - abs(errorOldScore / errorNewScore);
			double bias = ((pow(10.0, percentBetter) - 1) / (10.0 - 1.0)); // Exponential func from 0.0 to 1.0
			return errorNewScore + bias;
		}
		else if (errorNewScore < errorOldScore) {
			double percentWorse = 1.0 - abs(errorNewScore / errorOldScore);
			double bias = ((pow(10.0, percentWorse) - 1) / (10.0 - 1.0));

			return errorNewScore - bias;
		}
		else {
			return errorNewScore;
		}
	}

	static double rescaleAction(double action, double min, double max) {
		return action * (max - min) / 2.0 + (max + min) / 2.0;
	}

	static int printDirectory(const char* path) {
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
}

