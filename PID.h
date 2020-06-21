#pragma once
#include <chrono>
class PID
{
	public:
		PID(double kP, double kI, double kD, double min, double max);
		void init();
		double update(double error, int sleep = 0.0);
		void getWeights(double w[3]);
		void setWeights(double kP, double kI, double kD);

	private:
		double _max;
		double _min;

		double _kP;
		double _kD;
		double _kI;

		double _cP;
		double _cI;
		double _cD;

		std::chrono::steady_clock::time_point _currTime;
		std::chrono::steady_clock::time_point _prevTime;
		std::chrono::steady_clock::duration _deltTime;

		double _prevError;
		double _integral;
		double _dt;		
};

