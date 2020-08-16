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
#include <pca9685.h>
#include <wiringPi.h>
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

	static int calcTicks(float impulseMs, int hertz = 50, int pwm = 4096)
	{
		float cycleMs = 1000.0f / hertz;
		return (int)(pwm * impulseMs / cycleMs + 0.5f);
	}

	static int runServo(int servo, double angle, int Hz = 50, int pinBase = 300, double minAngle = 0.0, double maxAngle = 180.0, double minMs = 0.5, double maxMs = 2.5) {
		double millis = Utility::mapOutput(angle, minAngle, maxAngle, minMs, maxMs);
		int tick = calcTicks(millis, Hz);
		pwmWrite(pinBase + servo, tick);
	}

	static void calibrateServo(int Hz = 50, int pinBase = 300) {

		int i, j = 1;
		int pin;
		float millis;

		while (j)
		{
			printf("Enter servo pin [0-16]: ");
			scanf("%d", &pin);

			if (pin >= 0 && pin <= 16)
			{
				millis = 1.5f;
				i = 1;

				pwmWrite(pinBase + pin, calcTicks(millis, Hz));
				printf("Servo %d is centered at %1.2f ms\n", pin, millis);

				while (i)
				{
					printf("Enter milliseconds: ");
					scanf("%f", &millis);

					if (millis > 0 && millis <= 5)
					{
						pwmWrite(pinBase + pin, calcTicks(millis, Hz));
						delay(1000);
					}
					else
						i = 0;
				}
			}
			else
				j = 0;
		}
	}

	
	static double errorToReward(int errorNew, int errorOld, int absMax, bool done, bool alt = false) {
		// https://link.springer.com/article/10.1007/s11276-019-02225-x#Sec6

		/*if (done) {
			return -.50;
		}*/

		// if (!alt) {
		double r1, r2;

		// Scals
		double epsilon = 0.001;
		double alpha1 = .50;
		double alpha2 = .50;

		// ABS Errors
		errorOld = (errorOld < 0) ? (-errorOld) : (errorOld);
		errorNew = (errorNew < 0) ? (-errorNew) : (errorNew);

		// convert from 0 to 1
		double errorOldScaled = static_cast<double>(errorOld) / static_cast<double>(absMax);
		double errorNewScaled = static_cast<double>(errorNew) / static_cast<double>(absMax);

		if (errorNewScaled < epsilon) {
			r1 = 0.0; 
		}
		else {
			r1 = epsilon - errorNewScaled;
		}

		if (errorNewScaled <= errorOldScaled) {
			r2 = 0.0;
		}
		else {
			r2 = errorNewScaled - errorOldScaled;
		}

		return alpha1 * r1 + alpha2 * r2;
		
		//else {
		//	/*
		//		In short, reward ranges from -1.0 to 1.0
		//		If done error: -1
		//		Base reward from -0.5 to 0.5, plus additional bias for transitions that are good (and vic versa)
		//		Bias is a exponential func from 0.0 to .50

		//	*/

		//	// ABS Errors
		//	errorOld = (errorOld < 0) ? (-errorOld) : (errorOld);
		//	errorNew = (errorNew < 0) ? (-errorNew) : (errorNew);

		//	// Scale errors from 0.0 to 1.0
		//	double errorOldScaled = static_cast<double>(errorOld) / static_cast<double>(absMax);
		//	double errorNewScaled = static_cast<double>(errorNew) / static_cast<double>(absMax);

		//	// Score the errors
		//	double errorOldScore = mapOutput(1.0 - errorOldScaled, 0.0, 1.0, -0.5, .5);
		//	double errorNewScore = mapOutput(1.0 - errorNewScaled, 0.0, 1.0, -0.5, .5);

		//	if (errorNewScore > errorOldScore) {
		//		double percentBetter = 1.0 - abs(errorOldScore / errorNewScore);
		//		double bias = ((pow(10.0, percentBetter) - 1) / (10.0 - 1.0)) * 0.5; // Exponential func from 0.0 to .5
		//		return errorNewScore + bias;
		//	}
		//	else if (errorNewScore < errorOldScore) {
		//		double percentWorse = 1.0 - abs(errorNewScore / errorOldScore);
		//		double bias = ((pow(10.0, percentWorse) - 1) / (10.0 - 1.0)) * 0.5;

		//		return errorNewScore - bias;
		//	}
		//	else {
		//		return errorNewScore;
		//	}	
		//}	
	}

	static double errorToReward(int error, int absMax, bool done) {

		if (done) {
			return -1.0;
		}

		// ABS Error
		error = (error < 0) ? (-error) : (error);

		// Scale errors from 0.0 to 1.0
		double errorScaled = static_cast<double>(error) / static_cast<double>(absMax);

		// invert and scale from -1.0 to 1.0
		double inverted = 1.0 - errorScaled;
		double reward = mapOutput(inverted, 0.0, 1.0, -1.0, 1.0);

		return reward;
	}

	static double rescaleAction(double action, double min, double max) {
		action = min + (action + 1.0) * 0.5 * (max - min);
		action = std::clamp<double>(action, min, max);
		return action;
		// return action * (max - min) / 2.0 + (max + min) / 2.0;
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

	// delete sub-vector from m to n - 1. Indexes starts at 0.
	template<typename T>
	static void erase(std::vector<T> &v, int m, int n) {
		auto first = v.begin() + m;
		auto last = v.begin() + n + 1;
		v.erase(first, last);
	}

	// Concatinates two vectors together
	template<typename T>
	static std::vector<T> append(std::vector<T> &r, std::vector<T> &l) {
		std::vector<T> temp = r;
		temp.insert(temp.end(), l.begin(), l.end());
		
		return temp;
	}

	// Obtain sub-vector from m to n - 1. Indexes starts at 0.
	template<typename T>
	static std::vector<T> slice(std::vector<T> const& v, int m, int n) {
		auto first = v.cbegin() + m;
		auto last = v.cbegin() + n + 1;

		std::vector<T> vec(first, last);
		return vec;
	}
}

