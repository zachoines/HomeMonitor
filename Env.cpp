﻿#include "Env.h"
#include "data.h"
#include "util.h"
#include <cmath>

Env::Env(param* parameters, pthread_mutex_t* stateDataLock, pthread_cond_t* stateDataCond)
{

	_cond = stateDataCond;
	_lock = stateDataLock;
	_params = parameters;
	_config = parameters->config;

	_pids[0] = parameters->tilt;
	_pids[1] = parameters->pan;

	_resetAngles[0] = parameters->config->resetAngleY;
	_resetAngles[1] = parameters->config->resetAngleX;

	_disableServo[0] = parameters->config->disableY;
	_disableServo[1] = parameters->config->disableX;

	_currentAngles[0] = 0.0;
	_currentAngles[1] = 0.0;

	_lastAngles[0] = 0.0;
	_lastAngles[1] = 0.0;

	_invert[0] = _params->config->invertY;
	_invert[1] = _params->config->invertX;
}


void Env::_sleep()
{
	int milis = 1000 / _params->rate;
	Utility::msleep(milis);
}

void Env::_syncEnv()
{
	// Sleep for specified time and Wait for env to respond to changes
	_sleep();

	pthread_mutex_lock(_lock);
	for (int servo = 0; servo < NUM_SERVOS; servo++) {
		
		if (_disableServo[servo]) {
			continue;
		}

		while (_params->eventData[servo].timestamp == _lastTimeStamp[servo]) {
			pthread_cond_wait(_cond, _lock);
		}

		_lastData[servo] = _currentData[servo];
		_currentData[servo] = _params->eventData[servo];
		_lastTimeStamp[servo] = _currentData[servo].timestamp;
	}

	pthread_mutex_unlock(_lock);
}

bool Env::isDone()
{
	for (int servo = 0; servo < NUM_SERVOS; servo++) {
		if (_disableServo[servo]) {
			continue;
		}
		else {
			return _currentData[servo].done;
		}
	} 

	return true;
}

void Env::_resetEnv()
{

	for (int servo = 0; servo < NUM_SERVOS; servo++) {
		if (_disableServo[servo]) {
			continue;
		}
		Utility::runServo(servo, _resetAngles[servo]);
		_lastAngles[servo] = _resetAngles[servo];
		_currentAngles[servo] = _resetAngles[servo];
		_pids[servo]->init();
	}
}

RD Env::reset()
{
	_resetEnv();
	_syncEnv();

	RD data;

	for (int servo = 0; servo < NUM_SERVOS; servo++) {
		if (_disableServo[servo]) {
			continue;
		}

		_observation[servo].pidStateData = _pids[servo]->mockUpdate(_currentData[servo].Obj);
		_observation[servo].lastAngle = _lastAngles[servo];
		_observation[servo].currentAngle = _currentAngles[servo];
		data.servos[servo] = _observation[servo];
	}

	return data;
}

// Using action, take step and return observation, reward, done, and actions for every servo. 
// Note: SR[servo].currentState is always null. Retrieve currentState from previous 'step' or 'reset' call.
SR Env::step(double actions[NUM_SERVOS][NUM_ACTIONS], bool rescale)
{
	SR stepResults;

	double randChance = static_cast<float>(rand()) / static_cast <float> (RAND_MAX);
	
	for (int servo = 0; servo < NUM_SERVOS; servo++) {

		if (_disableServo[servo]) {
			continue;
		}

		// Scale PID actions if configured
		for (int a = 0; a < NUM_ACTIONS; a++) {
			stepResults.servos[servo].actions[a] = actions[servo][a];

			if (rescale) {
				actions[servo][a] = Utility::rescaleAction(actions[servo][a], _params->config->actionLow, _params->config->actionHigh);
			}
		}

		// Print out the PID gains
		if (0.05 >= randChance) {
			std::cout << "Here is the new actions(s): ";
			for (int a = 0; a < _params->config->numActions; a++) {
				std::cout << actions[servo][a] << ", ";
			}
			std::cout << std::endl;
		}

		_pids[servo]->setWeights(actions[servo][0], actions[servo][1], actions[servo][2]);
		double newAngle = _pids[servo]->update(_currentData[servo].Obj, 1000.0 / static_cast<double>(_params->rate));
		newAngle = Utility::mapOutput(newAngle, _params->config->pidOutputLow, _params->config->pidOutputHigh, _params->config->angleLow, _params->config->angleHigh);
		if (_invert[servo]) { newAngle = _params->config->angleHigh - newAngle; }
		_lastAngles[servo] = _currentAngles[servo];
		_currentAngles[servo] = newAngle;
	
		Utility::runServo(servo, newAngle);
	}

	_syncEnv();

	for (int servo = 0; servo < NUM_SERVOS; servo++) {

		if (_disableServo[servo]) {
			continue;
		}

		double lastError = _lastData[servo].Obj;
		double currentError = _currentData[servo].Obj;

		if (_lastData[servo].done) {
			throw std::runtime_error("State must represent a complete transition");
		}
		else {
			stepResults.servos[servo].reward = Utility::pidErrorToReward(currentError, lastError, static_cast<double>(_params->dims[servo]) / 2.0, _currentData[servo].done, 0.02, true);
		}

		// Fill out the step results
		stepResults.servos[servo].nextState.pidStateData = _pids[servo]->mockUpdate(currentError);
		stepResults.servos[servo].nextState.obj = _currentData[servo].Obj / _currentData[servo].Frame;
		stepResults.servos[servo].nextState.lastAngle = _lastAngles[servo] / _params->config->angleHigh;
		stepResults.servos[servo].nextState.currentAngle = _currentAngles[servo] / _params->config->angleHigh;
		stepResults.servos[servo].done = _currentData[servo].done;
		_observation[servo] = stepResults.servos[servo].nextState;
	}

	return stepResults;
}