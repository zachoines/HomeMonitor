#include "PID.h"
#include <chrono>
#include <wiringPi.h>
#include <iostream>

PID::PID(double kP, double kI, double kD, double min, double max) {
	_kP = kP;
	_kI = kI;
	_kD = kD;

	_max = max;
	_min = min;

	_windup_guard = 100; 
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

	if (_cI < -_windup_guard) {
		_cI = -_windup_guard;
	}
	else if (_cI > _windup_guard){
		_cI = _windup_guard;
	}
		

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

void PID::getPID(double w[3])
{
	w[0] = _cP;
	w[1] = _cI;
	w[2] = _cD;
}

void PID::setWindupGaurd(double guard)
{
	_windup_guard = guard;
}

void PID::getWeights(double w[3])
{
	w[0] = _kP;
	w[1] = _kI;
	w[2] = _kD;
}

void PID::setWeights(double kP, double kI, double kD)
{
	_kP = kP;
	_kI = kI;
	_kD = kD;
}


