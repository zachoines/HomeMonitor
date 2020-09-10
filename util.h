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
#include <chrono>
#include <thread>

// 3rd party libs
#include <wiringPiI2C.h>
#include <pca9685.h>
#include <wiringPi.h>
#include "opencv2/opencv.hpp"

namespace Utility {
	static void msleep(long msec)
	{
		
		delay(msec);
		/*struct timespec ts;
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

		return res;*/
		
		/*using namespace std;

		chrono::system_clock::time_point timePt =
			chrono::system_clock::now() + chrono::milliseconds(msec);

		this_thread::sleep_until(timePt);*/
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

	static int runServo(int servo, double angle, int Hz = 50, int pinBase = 300, double minAngle = -150.0, double maxAngle = 150.0, double minMs = 0.5, double maxMs = 2.5) {
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

	/*
		If done state: -1.
		Reward inverse of error, scaled from -1.0 to 0.0.
	*/
	static double errorToReward(double e, double max, bool d, double threshold = 0.01, bool alt = true) {

		double error = max - e;
		double absMax = max;
		double errorThreshold = threshold;
		bool done = d;

		if (alt) {

			if (done) {
				return -1.0;
			}

			double r = 0.0;
			error = std::fabs(error);

			// Scale errorsfrom 0.0 to 1.0
			double errorScaled = error / absMax;

			// If within threshold
			if (errorScaled <= errorThreshold) {
				r = .50;
			}
			else {
				r = errorThreshold - errorScaled;
			}

			return r;
		}
		else {

			if (done) {
				return -1.0;
			}

			// ABS Error
			error = std::fabs(error);

			// Scale errors from 0.0 to 1.0
			double errorScaled = error / absMax;

			// If within threshhold
			if (errorScaled <= errorThreshold) {
				return 0.0;
			}

			// invert and scale from -1.0 to 0.0
			double inverted = 1.0 - errorScaled;
			double reward = mapOutput(inverted, 0.0, 1.0, -1.0, 0.0);

			return reward;
		}
	}

	/*	
		/Base error + TD marginal increase/decrease
		A.) If overshooting: -1.
		B.) If done state: -3.
		C.) Reward inverse of error, scaled from -1.0 to 0.0, plus an
			additional bias that eponentially scales depending on percent difference. 
		B.) If new error is worse than last, then return step 'C' minus an
			additional bias that eponentially scales depending on percent difference.
			max error is -2.
	*/
	static double pidErrorToReward(double n, double o, double max, bool d, double threshold = 0.01, bool alt = true) {

		if (alt) {
			
			double absMax = max;
			double errorThreshold = threshold;
			bool done = d;

			if (done) {
				return -1.0;
			}

			bool direction_old = false;
			bool direction_new = false;
			int center = absMax;

			double r1 = 0.0;
			double r2 = 0.0;

			double targetCenterNew = n;
			double targetCenterOld = o;

			// scale from 0.0 to 1.0
			double errorOldScaled = std::fabs(center - std::fabs(o)) / center;
			double errorNewScaled = std::fabs(center - std::fabs(n)) / center;

			// If within threshold
			if (errorNewScaled <= errorThreshold) {
				r1 = .5;
			}
			else {
				r1 = errorThreshold - errorNewScaled;
			}

			// The target in ref to the center of frame. Left is F, right is T.
			if (targetCenterNew < center) { // target is left of frame center
				direction_new = false;
			}
			else { // target is right of frame center
				direction_new = true;
			}

			if (targetCenterOld < center) { // target is left of frame center
				direction_old = false;
			}
			else { // target is right of frame center
				direction_old = true;
			}

			//  Both to the right of frame center, situation #1;
			if (direction_old && direction_new) {

				double reward = std::fabs(errorNewScaled - errorOldScaled);

				if (targetCenterNew > targetCenterOld) { // frame center has moved furthure to object's left
					r2 = -reward;
				}
				else { // frame center has moved closer to object's left
					r2 = reward;
				}
			}

			// both to left of frame center, situation #2
			else if (!direction_old && !direction_new) {

				double reward = std::fabs(errorOldScaled - errorNewScaled);

				if (targetCenterNew > targetCenterOld) {  // frame center has moved closer to objects right
					r2 = reward;
				}
				else { // frame center has moved further from objects right
					r2 = -reward;
				}

			}

			// Frame center has overshot target. Old to the right and new to the left, situation #3
			else if (direction_old && !direction_new) {

				double error_old_corrected = std::fabs(std::fabs(targetCenterOld) - center);
				double error_new_corrected = std::fabs(std::fabs(targetCenterNew) - center);
				double difference = std::fabs(error_new_corrected - error_old_corrected);
				double reward = difference / center;

				if (error_old_corrected > error_new_corrected) {  // If move has resulted in be relatively closer to center
					r2 = reward;
				}
				else {
					r2 = -reward;
				}
			}
			else { // old left and new right, situation #4

				double error_old_corrected = std::fabs(std::fabs(targetCenterOld) - center);
				double error_new_corrected = std::fabs(std::fabs(targetCenterNew) - center);
				double difference = std::fabs(error_new_corrected - error_old_corrected);
				double reward = difference / center;

				if (error_old_corrected > error_new_corrected) {  // If move has resulted in be relatively closer to center
					r2 = reward;
				}
				else {
					r2 = -reward;
				}
			}

			return r1 + r2;

		} else {
			double absMax = max;
			double errorThreshold = threshold;
			bool done = d;
			double errorNew = n;
			double errorOld = o;

			// Punish done state
			if (done) {
				return -3.0;
			}

			// puinish overshooting   
			int signOld = errorOld >= 0 ? 1 : 0;
			int signNew = errorNew >= 0 ? 1 : 0;

			if (signOld - signNew != 0) {
				return -1.0;
			}
			else {

				// ABS Errors
				errorOld = abs(errorOld);
				errorNew = abs(errorNew);

				// Scale errors from 0.0 to 1.0
				double errorOldScaled = errorOld / static_cast<double>(absMax);
				double errorNewScaled = errorNew / static_cast<double>(absMax);

				// If within threshold
				if (errorNewScaled <= errorThreshold) {
					return 0.0;
				}

				// Score the errors 
				double errorOldScore = mapOutput(1.0 - errorOldScaled, 0.0, 1.0, -1.0, 0.0);
				double errorNewScore = mapOutput(1.0 - errorNewScaled, 0.0, 1.0, -1.0, 0.0);

				if (errorNewScore >= errorOldScore) {
					double percentBetter = 1.0 - abs(errorNewScore / errorOldScore);
					double bias = ((pow(10.0, percentBetter) - 1) / (10.0 - 1.0)); // Exponential func from 0.0 to 1.0

					return std::min(errorNewScore + bias, 0.0);
				}
				else {
					double percentWorse = 1.0 - abs(errorOldScore / errorNewScore);
					double bias = ((pow(10.0, percentWorse) - 1) / (10.0 - 1.0));

					return errorNewScore - bias;
				}
			}
		}
	}

	static double rescaleAction(double action, double min, double max) {

		/*double scale_factor = (max - min) / 2.0;
		double reloc_factor = max - scale_factor;
		action = (action * scale_factor) + reloc_factor;
		return std::clamp<double>(action, min, max);*/
		// return std::clamp<double>(action * ((max - min) / 2.0) + ((max + min) / 2.0), min, max);
		return action;
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

