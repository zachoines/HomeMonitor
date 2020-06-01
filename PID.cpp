#include "PID.h"
#include <chrono>
#include <wiringPi.h>

PID::PID(double kP, double kI, double kD, double min, double max) {
	_kP = kP;
	_kI = kI;
	_kD = kD;

	_max = max;
	_min = min;
}

void PID::init() {
	_currTime = std::chrono::steady_clock::now();
	_prevTime = _currTime;

	// initialize the previous error
	_prevError = 0;

	// initialize the term result variables
	_cP = 0;
	_cI = 0;
	_cD = 0;
}		

double PID::update(double error, int sleep) {
	// Delay execution
	if (sleep > 0) {
		delay(sleep);
	}
	
	// Delta time
	_currTime = std::chrono::steady_clock::now();
	_deltTime = _currTime - _prevTime;
		
	double deltaTime = double(_deltTime.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;

	// delta error
	double deltaError = error - _prevError;

	// proportional
	_cP = error;

	// integral
	_cI += error * deltaTime;

	// derivative
	(deltaTime > 0) ? (_cD = (deltaError / deltaTime)) : (_cD = 0);
	
	// save previous time and error
	_prevTime = _currTime;
	_prevError = error;

	// Cross-mult, sum and return
	double output = (_kP * _cP) + (_kI * _cI) + (_kD * _cD);
	
	if (output > _max) {
		output = _max;
	}
	else if (output < _min) {
		output = _min;
	}
		
	return output;
}


