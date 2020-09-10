#include "PID.h"
#include "util.h"
#include <chrono>
#include <wiringPi.h>
#include <iostream>
#include <cmath>

PID::PID(double kP, double kI, double kD, double min, double max, double setpoint) {
	_kP = kP;
	_kI = kI;
	_kD = kD;

	_init_kP = kP;
	_init_kI = kI;
	_init_kD = kD;

	_max = max;
	_min = min;

	_windup_guard = 1000; 
	_setpoint = setpoint;
}

void PID::init() {

	_currTime = std::chrono::steady_clock::now();
	_prevTime = _currTime;

	// initialize the previous error
	_prevError = 0.0;
	_last_input = 0.0;
	_integral = 0.0;

	// initialize the term result variables
	_cP = 0.0;
	_cI = 0.0;
	_cD = 0.0;

	// Reset the gains 
	_kP = _init_kP;
	_kI = _init_kI;
	_kD = _init_kD;

}

double PID::update(double input, double sleep) {
	
	// Delta time
	_currTime = std::chrono::steady_clock::now();
	_deltTime = _currTime - _prevTime;
	
	double deltaTime = double(_deltTime.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;

	// Delay execution to rate
	if (sleep > deltaTime * 1000.0) {
		// std::cout << "Not enough: " << deltaTime * 1000.0 << std::endl;
		double diff  = sleep - deltaTime;
		_prevTime = _currTime;
		Utility::msleep(static_cast<long>(diff));
		_currTime = std::chrono::steady_clock::now();
		_deltTime = _currTime - _prevTime;
		deltaTime = double(_deltTime.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
	}

	// Error
	double error = input - _setpoint;

	// Proportional of Error
	_cP = error;

	// Integral of error with respect to time
	_cI += (error * (_kI * deltaTime));

	// Derivative of input with respect to time
	double dInput = (_last_input - input);
	(deltaTime > 0.0) ? (_cD = (1.0 / deltaTime)) : (_cD = 0.0);

	// windup gaurd
	if (_cI < -_windup_guard) {
		_cI = -_windup_guard;
	}
	else if (_cI > _windup_guard) {
		_cI = _windup_guard;
	}

	// save previous time and error
	_prevTime = _currTime;
	_last_input = input;

	// Cross-mult, sum and return
	double output = (_kP * _cP) + (_cI) - (_kD * _cD * dInput);


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

double PID::getWindupGaurd()
{
	return _windup_guard;
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


